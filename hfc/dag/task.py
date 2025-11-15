import asyncio
from enum import Enum, auto
from typing import List, Callable, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    INIT = auto()
    LOAD_BATCH = auto()
    FORWARD = auto()
    BACKWARD = auto()
    COMM_PREPARE = auto()      # 準備通信數據 (GaLore投影, 壓縮)
    COMM_EXECUTE = auto()      # 執行 P2P 或集體通信
    COMM_FINALIZE = auto()     # 後處理通信數據 (解壓, 廣播)
    OPTIMIZER_STEP = auto()
    FINISH = auto()

class Task:
    """DAG 中的一個節點，代表一個計算或通信任務。"""
    _task_id_counter = 0

    def __init__(self, task_type: TaskType, execute_fn: Callable, dependencies: Optional[List['Task']] = None, name: Optional[str] = None):
        self.task_id = Task._task_id_counter
        Task._task_id_counter += 1
        
        self.task_type = task_type
        self.name = name or f"{task_type.name}-{self.task_id}"
        self.execute_fn = execute_fn
        self.dependencies = dependencies if dependencies else []
        self.children: List['Task'] = []
        
        for dep in self.dependencies:
            dep.children.append(self)
            
        self.future = asyncio.Future()
        self.status = "PENDING"
        self.start_time = 0.0
        self.end_time = 0.0

    def __repr__(self):
        dep_ids = [d.task_id for d in self.dependencies]
        return f"Task(id={self.task_id}, name={self.name}, deps={dep_ids}, status={self.status})"

    async def run(self):
        """執行任務。首先等待所有依賴完成。"""
        try:
            # 等待依賴任務的 future 完成
            dependency_results = await asyncio.gather(*[dep.future for dep in self.dependencies])
            
            self.status = "RUNNING"
            self.start_time = time.perf_counter()
            logger.debug(f"Task {self.name} started.")

            # 異步執行本任務的具體邏輯
            # 我們假設 execute_fn 是一個協程 (async def)
            result = await self.execute_fn(*dependency_results)
            
            self.end_time = time.perf_counter()
            self.status = "COMPLETED"
            logger.debug(f"Task {self.name} completed in {self.end_time - self.start_time:.4f}s.")
            
            self.future.set_result(result)

        except Exception as e:
            logger.error(f"Task {self.name} failed: {e}", exc_info=True)
            self.status = "FAILED"
            self.future.set_exception(e)
