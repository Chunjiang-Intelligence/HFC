import asyncio
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import torch.distributed.rpc as rpc
from hfc.networking.topology import TopologyMonitor
from hfc.networking.grouping import GroupingAlgorithm

logger = logging.getLogger(__name__)

class OrchestratorService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OrchestratorService, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if not hasattr(self, '_initialized'):
            self.config = config
            self.nodes: Dict[str, Dict] = {} # {node_id: {"addr": (ip, port), "status": "alive", "info": RpcInfo}}
            self.topology_monitor = TopologyMonitor()
            self.grouper = GroupingAlgorithm(strategy=self.config.grouping_strategy)
            self.training_started = False
            self.global_rank_counter = 0
            self._initialized = True

    async def run(self):
        logger.info("Orchestrator service running.")
        asyncio.create_task(self.heartbeat_check())
        while not self.training_started:
            if len(self.nodes) >= self.config.min_nodes_to_start:
                logger.info("Minimum number of nodes reached. Finalizing groups and starting training...")
                await self.update_topology_and_groups()
                await self.start_training()
            else:
                logger.info(f"Waiting for nodes... {len(self.nodes)}/{self.config.min_nodes_to_start} registered.")
            await asyncio.sleep(10)

    @staticmethod
    def _get_service():
        assert OrchestratorService._instance is not None
        return OrchestratorService._instance

    @staticmethod
    @rpc.functions.async_execution
    def register_node_rpc(node_id: str, node_ip: str, node_port: int):
        service = OrchestratorService._get_service()
        future = asyncio.Future()

        async def impl():
            node_info = rpc.get_worker_info(node_id)
            if node_id not in service.nodes:
                service.nodes[node_id] = {
                    "addr": (node_ip, node_port),
                    "status": "alive",
                    "info": node_info
                }
                logger.info(f"Registered new node: {node_id} at {node_ip}:{node_port}")
                # Trigger a regroup if training hasn't started
                if not service.training_started:
                     asyncio.create_task(service.update_topology_and_groups())
            future.set_result(True)
        
        asyncio.create_task(impl())
        return future

    async def heartbeat_check(self):
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            dead_nodes = []
            alive_nodes = list(self.nodes.keys())
            for node_id in alive_nodes:
                try:
                    await rpc.rpc_async(self.nodes[node_id]["info"], lambda: True, timeout=5)
                except Exception:
                    logger.warning(f"Node {node_id} failed heartbeat. Marking as dead.")
                    dead_nodes.append(node_id)
            
            if dead_nodes:
                for node_id in dead_nodes:
                    self.nodes.pop(node_id, None)
                if self.training_started:
                    logger.warning("Node failure during training detected. Re-grouping and recovery not yet implemented.")
                    # In a full production system, this would trigger a recovery and regrouping process.
                else:
                    await self.update_topology_and_groups()
    
    async def update_topology_and_groups(self):
        node_ids = list(self.nodes.keys())
        if len(node_ids) < self.config.min_nodes_to_start:
            return

        logger.info("Measuring network topology...")
        ping_matrix = await self.topology_monitor.measure_latency_rpc(self.nodes)
        
        logger.info("Forming new groups...")
        groups = self.grouper.form_groups(node_ids, ping_matrix, k=self.config.nodes_per_group)
        
        world_info = self._prepare_world_info(groups)
        logger.info(f"Distributing new world info to {len(self.nodes)} nodes.")
        
        tasks = [
            rpc.rpc_async(self.nodes[node_id]["info"], "hfc.node.service.NodeService.update_world_info_rpc", args=(world_info,))
            for node_id in node_ids
        ]
        for fut in tasks:
            await fut

    def _prepare_world_info(self, groups: List[List[str]]) -> Dict:
        world_size = len(self.nodes)
        node_to_global_rank = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        
        group_info = {}
        for i, group_nodes in enumerate(groups):
            leader_id = group_nodes[0] # Simple leader election
            leader_rank = node_to_global_rank[leader_id]
            group_ranks = [node_to_global_rank[nid] for nid in group_nodes]
            group_info[i] = {
                "leader_id": leader_id,
                "leader_rank": leader_rank,
                "nodes": group_nodes,
                "ranks": sorted(group_ranks)
            }
        
        leader_ranks = sorted([g["leader_rank"] for g in group_info.values()])

        return {
            "world_size": world_size,
            "node_to_global_rank": node_to_global_rank,
            "groups": group_info,
            "leader_ranks": leader_ranks
        }

    async def start_training(self):
        if self.training_started or len(self.nodes) < self.config.min_nodes_to_start:
            return
        
        logger.info("--- Sending START TRAINING command to all nodes ---")
        self.training_started = True
        tasks = [
            rpc.rpc_async(node_info["info"], "hfc.node.service.NodeService.start_training_rpc", args=())
            for node_info in self.nodes.values()
        ]
        for fut in tasks:
            await fut
