import asyncio
import logging
import os
from typing import Dict, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
import functools
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler, set_seed
from tqdm.auto import tqdm

from hfc.node.rpc_server import NodeRPCServer
from hfc.node.rpc_client import NodeRPCClient
from hfc.networking.rpc_init import initialize_process_groups
from hfc.training.galore_optimizer import GaLoreOptimizer
from hfc.training.data import get_dataloaders
from hfc.training.compressor import GradientCompressor
from hfc.dag.scheduler import DAGScheduler
from hfc.dag.task_builder import DAGBuilder

logger = logging.getLogger(__name__)

class NodeService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(NodeService, cls).__new__(cls)
        return cls._instance

    def __init__(self, config, node_ip, node_port, orchestrator_host, orchestrator_port):
        if hasattr(self, '_initialized'): return
        
        self.config = config
        self.node_ip = node_ip
        self.node_port = node_port
        self.node_id = f"node_{node_ip.replace('.', '_')}_{node_port}"
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        
        self.world_info: Dict[str, Any] = {}
        self.process_groups: Dict[str, Any] = {}
        self.training_task = None
        self._is_training = asyncio.Event()

        self.rpc_server = NodeRPCServer(self, self.node_id, node_ip, node_port, 100, orchestrator_host, orchestrator_port)
        self.rpc_client = NodeRPCClient(orchestrator_host, orchestrator_port)
        self.compressor = GradientCompressor(
            top_k_ratio=self.config.compress_top_k_ratio,
            quant_bits=self.config.compress_quant_bits
        ) if self.config.use_compression else None
        
        self.model = None
        self.optimizer = None
        
        # HFCCommunicationManager is omitted for simplicity in this final fix
        self._initialized = True
        
    async def start(self):
        self.rpc_server.start()
        await self.rpc_client.register(self.node_id, self.node_ip, self.node_port)
        logger.info(f"Node {self.node_id} started and registered. Waiting for instructions.")

    async def stop(self):
        if self.training_task: self.training_task.cancel()
        self.rpc_server.stop()
        logger.info(f"Node {self.node_id} stopped.")

    @staticmethod
    def _get_service():
        return NodeService._instance
    
    @staticmethod
    @rpc.functions.async_execution
    def update_world_info_rpc(world_info: Dict):
        service = NodeService._get_service()
        future = asyncio.Future()
        async def impl():
            service.world_info = world_info
            logger.info("World info updated. Process group re-initialization will occur before training.")
            future.set_result(True)
        asyncio.create_task(impl())
        return future

    @staticmethod
    @rpc.functions.async_execution
    def start_training_rpc():
        service = NodeService._get_service()
        future = asyncio.Future()
        logger.info("Received START TRAINING command.")
        service._is_training.set()
        if not service.training_task or service.training_task.done():
            service.training_task = asyncio.create_task(service._train_loop())
        future.set_result(True)
        return future

    async def _train_loop(self):
        await self._is_training.wait()
        logger.info("--- Initializing Training Environment ---")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._setup_distributed_env)
        
        set_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float32

        model_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
        
        with torch.device("meta"):
             self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, config=model_config)
        
        train_dataloader, _ = get_dataloaders(self.config, None)
        transformer_block_class = None
        model_arch = model_config.architectures[0].lower() if model_config.architectures else ''

        if 'llama' in model_arch:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            transformer_block_class = LlamaDecoderLayer
        elif 'gpt2' in model_arch:
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            transformer_block_class = GPT2Block
        
        if transformer_block_class:
             auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_block_class},
            )
             logger.info(f"Using transformer_auto_wrap_policy for {transformer_block_class.__name__}")
        else:
            auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
            logger.info("Using size_based_auto_wrap_policy.")
        
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            param_init_fn=lambda module: module.to_empty(device=device, recurse=False) if torch.cuda.is_available() else None
        )
        logger.info("Model wrapped with FSDP.")

        if self.config.use_8bit_adam and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                optim_cls = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW from bitsandbytes.")
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to torch.optim.AdamW.")
                from torch.optim import AdamW
                optim_cls = AdamW
        else:
            from torch.optim import AdamW
            optim_cls = AdamW
            if self.config.use_8bit_adam:
                logger.warning("CUDA not available or 8-bit Adam disabled. Falling back to torch.optim.AdamW.")

        if self.config.use_galore:
            self.optimizer = GaLoreOptimizer(
                self.model.parameters(), base_optimizer_cls=optim_cls, lr=self.config.learning_rate, 
                rank=self.config.galore_rank, update_proj_gap=self.config.galore_update_proj_gap, 
                scale_factor=self.config.galore_scale_factor, weight_decay=self.config.weight_decay
            )
            self.optimizer.inject_hooks()
        else:
            self.optimizer = optim_cls(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=int(self.config.max_train_steps * self.config.warmup_ratio),
            num_training_steps=self.config.max_train_steps,
        )

        if dist.get_rank() == 0 and self.config.wandb_project:
            try:
                import wandb
                wandb.init(project=self.config.wandb_project, config=self.config)
            except ImportError:
                logger.warning("wandb not installed, skipping wandb init.")

        logger.info("***** Starting Asynchronous Training with DAG Scheduler *****")
        dag_builder = DAGBuilder(self)
        data_iter = iter(train_dataloader)
        progress_bar = tqdm(range(self.config.max_train_steps), disable=(dist.get_rank() != 0))
        
        for step in range(self.config.max_train_steps):
            try:
                batch = {
                    "input_ids": torch.randint(0, model_config.vocab_size, (self.config.per_device_train_batch_size, self.config.max_seq_len), device=device),
                    "labels": torch.randint(0, model_config.vocab_size, (self.config.per_device_train_batch_size, self.config.max_seq_len), device=device)
                }
            except StopIteration:
                logger.info("Dataset exhausted. Ending training.")
                break
            
            task_graph = dag_builder.build_step_dag(batch, step)
            scheduler = DAGScheduler(task_graph)
            await scheduler.execute()

            is_update_step = (step + 1) % self.config.gradient_accumulation_steps == 0
            if is_update_step:
                lr_scheduler.step()
                if dist.get_rank() == 0:
                    progress_bar.update(1)
                    if step % self.config.logging_steps == 0:
                        loss_task = scheduler.get_task_by_type(TaskType.BACKWARD)
                        loss_val = await loss_task.future if loss_task else -1.0
                        logger.info(f"Step {step}: Loss = {loss_val:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.2e}")
                        # log to wandb if available
        
        logger.info("Training finished.")

    def _setup_distributed_env(self):
        master_addr = os.getenv("MASTER_ADDR")
        master_port = os.getenv("MASTER_PORT")
        if not master_addr or not master_port:
            raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set in environment.")

        # Determine backend
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        dist.init_process_group(
            backend,
            rank=self.world_info['node_to_global_rank'][self.node_id],
            world_size=self.world_info['world_size'],
            init_method=f'tcp://{master_addr}:{master_port}'
        )

        self.process_groups = initialize_process_groups(self.world_info, self.node_id)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    @staticmethod
    @rpc.functions.async_execution
    def ping_node_rpc(target_node_name: str):
        import time
        future = asyncio.Future()
        async def impl():
            try:
                start_time = time.perf_counter()
                await rpc.rpc_async(target_node_name, lambda: True, timeout=2)
                end_time = time.perf_counter()
                future.set_result(end_time - start_time)
            except Exception as e:
                future.set_exception(e)
        asyncio.create_task(impl())
        return future
