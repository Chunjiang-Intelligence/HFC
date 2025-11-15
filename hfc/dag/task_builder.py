import torch
import torch.distributed as dist
from .task import Task, TaskType
import logging

logger = logging.getLogger(__name__)

class DAGBuilder:
    """為一個訓練步驟（step）構建完整的、可執行的異步任務圖。"""
    def __init__(self, node_service):
        self.node_service = node_service
        self.model = node_service.model
        self.optimizer = node_service.optimizer
        self.compressor = node_service.compressor
        
    def build_step_dag(self, batch_data: dict, step: int):
        """
        構建一個 step 的 DAG。
        """
        ns = self.node_service # shorthand

        async def init_fn():
            return batch_data

        async def forward_fn(batch):
            logger.debug("FWD: Starting forward pass.")
            outputs = self.model(**batch)
            logger.debug("FWD: Forward pass completed.")
            return outputs

        async def backward_fn(fwd_result):
            loss = fwd_result.loss
            scaled_loss = loss / ns.config.gradient_accumulation_steps
            logger.debug(f"BWD: Starting backward pass for loss {scaled_loss.item():.4f}.")
            # 在 FSDP 中，backward 會觸發梯度計算和同步
            scaled_loss.backward()
            logger.debug("BWD: Backward pass completed.")
            return loss.item()

        async def comm_prepare_fn(bwd_loss):
            # 在 GaLore hook 模式下，投影已經在 backward 中完成
            # 這一步主要是為了未來可能的、手動的梯度壓縮
            logger.debug("COMM_PREPARE: Preparing gradients for communication.")
            # 在 HFC 模式下，FSDP 的默認 all-reduce 被我們的 hook 攔截
            # 所以這裡不需要做額外操作，梯度同步發生在 backward 內部
            return "Gradients are ready for HFC communication within FSDP hooks."
        
        async def comm_execute_fn(prepare_signal):
            # 實際通信由 FSDP 的 post-backward hook 觸發的 all_reduce 完成
            # 我們的 hook (HFCCommunicationManager) 會攔截它
            # 這個任務只是一個邏輯上的佔位符，確保依賴關係正確
            logger.debug("COMM_EXECUTE: HFC communication is handled by FSDP hooks.")
            await asyncio.sleep(0) # yield control
            return "Communication logic executed within hooks."

        async def comm_finalize_fn(comm_signal):
            logger.debug("COMM_FINALIZE: Finalizing communication.")
            return "Communication finalized."

        async def optimizer_step_fn(finalize_signal):
            is_update_step = (step + 1) % ns.config.gradient_accumulation_steps == 0
            if is_update_step:
                logger.debug("OPTIMIZER: Performing optimizer step.")
                if ns.config.gradient_clipping:
                    self.model.clip_grad_norm_(ns.config.gradient_clipping)
                self.optimizer.step()
                self.optimizer.zero_grad()
                return "Optimizer step performed."
            return "Optimizer step skipped (gradient accumulation)."

        # --- 組裝 DAG ---
        init_task = Task(TaskType.INIT, init_fn, name=f"Step{step}-Init")
        fwd_task = Task(TaskType.FORWARD, forward_fn, dependencies=[init_task], name=f"Step{step}-Forward")
        bwd_task = Task(TaskType.BACKWARD, backward_fn, dependencies=[fwd_task], name=f"Step{step}-Backward")
        
        # 在 FSDP + hook 模式下，通信與 BWD 緊密耦合，因此依賴關係很直接
        comm_prepare_task = Task(TaskType.COMM_PREPARE, comm_prepare_fn, dependencies=[bwd_task], name=f"Step{step}-CommPrepare")
        comm_execute_task = Task(TaskType.COMM_EXECUTE, comm_execute_fn, dependencies=[comm_prepare_task], name=f"Step{step}-CommExecute")
        comm_finalize_task = Task(TaskType.COMM_FINALIZE, comm_finalize_fn, dependencies=[comm_execute_task], name=f"Step{step}-CommFinalize")
        
        optimizer_task = Task(TaskType.OPTIMIZER_STEP, optimizer_step_fn, dependencies=[comm_finalize_task], name=f"Step{step}-Optimize")

        all_tasks = [
            init_task, fwd_task, bwd_task, 
            comm_prepare_task, comm_execute_task, comm_finalize_task,
            optimizer_task
        ]
        
        return all_tasks
