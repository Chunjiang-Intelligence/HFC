import asyncio
import logging
import os
from typing import Dict
import torch
import torch.distributed.rpc as rpc
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler, set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

from .rpc_server import NodeRPCServer
from .rpc_client import NodeRPCClient
from hfc.networking.rpc_init import initialize_process_groups
from hfc.training.galore_optimizer import GaLoreOptimizer
from hfc.training.data import get_dataloaders
import bitsandbytes as bnb

logger = logging.getLogger(__name__)

class NodeService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(NodeService, cls).__new__(cls)
        return cls._instance

    def __init__(self, config, node_ip, node_port, orchestrator_host, orchestrator_port):
        if not hasattr(self, '_initialized'):
            self.config = config
            self.node_ip = node_ip
            self.node_port = node_port
            self.node_id = f"node_{node_ip.replace('.', '_')}_{node_port}"
            self.orchestrator_host = orchestrator_host
            self.orchestrator_port = orchestrator_port
            
            self.world_info: Dict = {}
            self.process_groups: Dict = {}
            self.training_task = None
            self._is_training = asyncio.Event()

            # A placeholder world_size for RPC init, will be updated by orchestrator
            # We add a large number to avoid rank conflicts
            self.rpc_server = NodeRPCServer(self, self.node_id, node_ip, node_port, 100, orchestrator_host, orchestrator_port)
            self.rpc_client = NodeRPCClient(orchestrator_host, orchestrator_port)
            self._initialized = True

    async def start(self):
        self.rpc_server.start()
        await self.rpc_client.register(self.node_id, self.node_ip, self.node_port)
        logger.info(f"Node {self.node_id} started and registered. Waiting for instructions.")

    async def stop(self):
        if self.training_task:
            self.training_task.cancel()
        self.rpc_server.stop()
        logger.info(f"Node {self.node_id} stopped.")

    @staticmethod
    def _get_service():
        assert NodeService._instance is not None
        return NodeService._instance
    
    @staticmethod
    @rpc.functions.async_execution
    def update_world_info_rpc(world_info: Dict):
        service = NodeService._get_service()
        future = asyncio.Future()
        
        async def impl():
            service.world_info = world_info
            logger.info("World info updated. Re-initializing process groups...")
            try:
                # This is a blocking call, run in executor
                loop = asyncio.get_event_loop()
                service.process_groups = await loop.run_in_executor(
                    None, initialize_process_groups, service.world_info, service.node_id
                )
                logger.info("Process groups re-initialized successfully.")
                future.set_result(True)
            except Exception as e:
                logger.error(f"Failed to initialize process groups: {e}")
                future.set_exception(e)

        asyncio.create_task(impl())
        return future

    @staticmethod
    @rpc.functions.async_execution
    def start_training_rpc():
        service = NodeService._get_service()
        future = asyncio.Future()
        logger.info("Received START TRAINING command.")
        service._is_training.set()
        
        # Start the training loop in the background
        if not service.training_task or service.training_task.done():
            service.training_task = asyncio.create_task(service._train_loop())
        
        future.set_result(True)
        return future

    async def _train_loop(self):
        await self._is_training.wait()
        logger.info("--- Starting Training Loop ---")
        
        # 1. Initialize Accelerator with FSDP
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
        )
        accelerator = Accelerator(
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb" if self.config.wandb_project else "tensorboard",
        )
        
        # 2. Setup logging, seed, etc. only on main process
        if accelerator.is_main_process:
            if self.config.wandb_project:
                import wandb
                wandb.init(project=self.config.wandb_project, config=self.config)
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

        set_seed(self.config.seed)
        
        # 3. Load model and data
        model_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, config=model_config, torch_dtype=torch.bfloat16)
        train_dataloader, eval_dataloader = get_dataloaders(self.config, accelerator)

        # 4. Create Optimizer
        optim_cls = bnb.optim.AdamW8bit if self.config.use_8bit_adam else torch.optim.AdamW
        if self.config.use_galore:
            optimizer = GaLoreOptimizer(model.parameters(), base_optimizer_cls=optim_cls, lr=self.config.learning_rate, rank=self.config.galore_rank)
            # hooks are injected later after model is wrapped by FSDP
        else:
            optimizer = optim_cls(model.parameters(), lr=self.config.learning_rate)

        # 5. Prepare with Accelerator (this will wrap the model with FSDP)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
        
        # Now inject hooks if using GaLore
        if self.config.use_galore:
            # FSDP requires accessing the optimizer instance on the unwrapped model
            # This is complex. For now, we assume GaLore is applied before FSDP wrapping
            # which is what `accelerator.prepare` does. Let's find the optimizer and inject.
            # This is a bit of a hack to get the underlying GaLore optimizer instance
            if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, GaLoreOptimizer):
                 optimizer.optimizer.inject_hooks()
            elif isinstance(optimizer, GaLoreOptimizer):
                optimizer.inject_hooks()
            else:
                 logger.error("Could not find GaLoreOptimizer instance to inject hooks.")
                 return

        # 6. Scheduler and progress bar
        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(self.config.max_train_steps * self.config.warmup_ratio),
            num_training_steps=self.config.max_train_steps,
        )
        lr_scheduler = accelerator.prepare(lr_scheduler)

        progress_bar = tqdm(range(self.config.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        logger.info("***** Running training *****")
        model.train()
      
        for epoch in range(self.config.num_train_epochs):
             for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    
                    # Instead of a complex DAG, we do synchronous communication after grad sync
                    if accelerator.sync_gradients:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        
                        progress_bar.update(1)
                        completed_steps += 1
                        
                        if completed_steps % self.config.logging_steps == 0:
                            logger.info(f"Step {completed_steps}, Loss: {loss.item()}")

                if completed_steps >= self.config.max_train_steps:
                    break
        
        logger.info("Training finished.")
