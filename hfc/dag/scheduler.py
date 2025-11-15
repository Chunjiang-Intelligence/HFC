import asyncio
from typing import List
from .task import Task
import logging

logger = logging.getLogger(__name__)

class DAGScheduler:
    """異步執行基於 Task 的有向無環圖。"""
    def __init__(self, all_tasks: List[Task]):
        self.all_tasks = all_tasks
        self.entry_points = self._find_entry_points()
        if not self.entry_points:
            raise ValueError("DAG has no entry points (tasks with no dependencies). It might contain a cycle.")
            
    def _find_entry_points(self) -> List[Task]:
        """找到沒有任何依賴的任務，作為 DAG 的入口。"""
        return [task for task in self.all_tasks if not task.dependencies]

    async def execute(self):
        """
        並行執行整個 DAG，並等待所有任務完成。
        """
        logger.debug(f"Scheduler starting with {len(self.entry_points)} entry points.")
        
        # 創建所有任務的協程，但不立即執行
        task_coroutines = {task.task_id: task.run() for task in self.all_tasks}
        
        # 我們只需要等待 DAG 的最終節點（沒有子節點的任務）
        final_tasks = [task for task in self.all_tasks if not task.children]
        
        if not final_tasks:
            logger.warning("DAG has no final tasks. Will wait for all tasks.")
            final_tasks = self.all_tasks

        # 等待所有最終任務的 future 完成
        # 由於任務間的依賴關係，這將確保整個圖被執行
        await asyncio.gather(*[task.future for task in final_tasks])
        
        logger.debug("DAG execution complete.")

    def get_task_by_type(self, task_type):
        """輔助函數，用於獲取特定類型的第一個任務。"""
        for task in self.all_tasks:
            if task.task_type == task_type:
                return task
        return None
