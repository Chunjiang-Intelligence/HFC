import asyncio
import logging
import os
from typing import Dict, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler, set_seed, LlamaDecoderLayer
from tqdm.auto import tqdm

from .rpc_server import NodeRPCServer
from .rpc_client import NodeRPCClient
from hfc.networking.rpc_init import initialize_process_groups
from hfc.training.galore_optimizer import GaLoreOptimizer
from hfc.training.data import get_dataloaders
from hfc.training.compressor import GradientCompressor
from hfc.dag.scheduler import DAGScheduler
from hfc.dag.task_builder import DAGBuilder

logger = logging.getLogger(__name__)

HFC_COMM_HOOK_MANAGER = None

class HFCCommunicationManager:
    """管理 FSDP 的通信 hook，將其替換為 HFC 的分層通信。"""
    def __init__(self, node_service):
        self.node_service = node_service
        self.original_all_reduce = dist.all_reduce
        self.is_hooked = False

    def hfc_all_reduce_hook(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        """我們的自定義 all_reduce 邏輯。"""
        # 如果 FSDP 調用時指定了 group，或者該 group 不是全局的，我們就遵循它
        # 這確保了 FSDP 內部的其他同步操作不受影響
        if group is not None and group != dist.group.WORLD:
            return self.original_all_reduce(tensor, op=op, group=group, async_op=async_op)
        
        # --- 這是我們攔截 FSDP 梯度同步的地方 ---
        # 1. 組內 All-Reduce (使用高帶寬)
        group_pg = self.node_service.process_groups.get('group')
        if group_pg:
            dist.all_reduce(tensor, op=op, group=group_pg, async_op=False)

        # 2. 如果是 Leader，進行組間通信
        if self.node_service.is_leader():
            leaders_pg = self.node_service.process_groups.get('leaders')
            if leaders_pg:
                dist.all_reduce(tensor, op=op, group=leaders_pg, async_op=False)

        # 3. 組內廣播最終結果
        if group_pg:
            leader_global_rank = self.node_service.get_group_leader_rank()
            dist.broadcast(tensor, src=leader_global_rank, group=group_pg, async_op=False)
        
        # FSDP 期望一個 handle，我們返回一個已完成的操作
        class DummyWork:
            def wait(self): pass
        
        if async_op:
            return DummyWork()
        else:
            return None


    def install_hook(self):
        if not self.is_hooked:
            logger.info("Installing HFC communication hook on dist.all_reduce...")
            dist.all_reduce = self.hfc_all_reduce_hook
            self.is_hooked = True

    def remove_hook(self):
        if self.is_hooked:
            logger.info("Removing HFC communication hook.")
            dist.all_reduce = self.original_all_reduce
            self.is_hooked = False


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
        
        global HFC_COMM_HOOK_MANAGER
        HFC_COMM_HOOK_MANAGER = HFCCommunicationManager(self)
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

    def is_leader(self) -> bool:
        my_rank = self.world_info.get('node_to_global_rank', {}).get(self.node_id)
        return my_rank is not None and my_rank in self.world_info.get('leader_ranks', [])

    def get_group_leader_rank(self) -> int:
        my_rank = self.world_info.get('node_to_global_rank', {}).get(self.node_id)
        for group_data in self.world_info.get('groups', {}).values():
            if my_rank in group_data['ranks']:
                return group_data['leader_rank']
        raise RuntimeError("Could not find group leader for current node.")

    async def _train_loop(self):
        await self._is_training.wait()
        logger.info("--- Initializing Training Environment ---")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._setup_distributed_env)
        
        set_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float32

        model_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
        
        # FSDP recommends creating model on CPU first
        with torch.device("meta"):
             self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, config=model_config)
        
        train_dataloader, _ = get_dataloaders(self.config, None)
        
        # FSDP wrapping policy
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        self.model = FSDP(
            self.model,
            auto_wrap_policy=llama_auto_wrap_policy,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        )
        logger.info("Model wrapped with FSDP.")

        optim_cls = bnb.optim.AdamW8bit if self.config.use_8bit_adam else torch.optim.AdamW
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
            import wandb
            wandb.init(project=self.config.wandb_project, config=self.config)

        # HFC_COMM_HOOK_MANAGER.install_hook() # Monkey-patching can be risky, use with caution

        logger.info("***** Starting Asynchronous Training with DAG Scheduler *****")
        dag_builder = DAGBuilder(self)
        data_iter = iter(train_dataloader)
        progress_bar = tqdm(range(self.config.max_train_steps), disable=(dist.get_rank() != 0))
        
        for step in range(self.config.max_train_steps):
            try:
                batch = next(data_iter)
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            except StopIteration:
                logger.info("Dataset exhausted. Ending training.")
                break
            
            # Build and execute the DAG for one step
            task_graph = dag_builder.build_step_dag(batch, step)
            scheduler = DAGScheduler(task_graph)
            await scheduler.execute()

            # Post-step updates
            is_update_step = (step + 1) % self.config.gradient_accumulation_steps == 0
            if is_update_step:
                lr_scheduler.step()
                if dist.get_rank() == 0:
                    progress_bar.update(1)
                    if step % self.config.logging_steps == 0:
                        loss_task = scheduler.get_task_by_type(TaskType.BACKWARD)
                        loss_val = await loss_task.future if loss_task else -1.0
                        logger.info(f"Step {step}: Loss = {loss_val:.4f}, LR = {lr_scheduler.get_last_lr()[0]:.2e}")
                        if self.config.wandb_project:
                            wandb.log({"train_loss": loss_val, "lr": lr_scheduler.get_last_lr()[0]}, step=step)

        # HFC_COMM_HOOK_MANAGER.remove_hook()
        logger.info("Training finished.")

    def _setup_distributed_env(self):
        master_addr = os.getenv(self.config.master_addr_env)
        master_port = os.getenv(self.config.master_port_env)
        if not master_addr or not master_port:
            raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set in environment.")

        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            rank=self.world_info['node_to_global_rank'][self.node_id],
            world_size=self.world_info['world_size'],
            init_method=f'tcp://{master_addr}:{master_port}'
        )

        self.process_groups = initialize_process_groups(self.world_info, self.node_id)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
