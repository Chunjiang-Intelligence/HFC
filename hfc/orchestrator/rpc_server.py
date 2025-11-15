import logging
import torch.distributed.rpc as rpc
import os

logger = logging.getLogger(__name__)

class NodeRPCServer:
    def __init__(self, service, node_id: str, node_ip: str, node_port: int, orchestrator_host: str, orchestrator_port: int):
        self.service = service # 接收一个 service 实例
        self.node_id = node_id
        self.node_ip = node_ip
        self.node_port = node_port
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        
        self.init_method = f"tcp://{self.orchestrator_host}:{self.orchestrator_port + 1}"
        
    def start(self):
        logger.info(f"Initializing RPC for node {self.node_id} via {self.init_method}")
        
        world_size = int(os.environ.get("WORLD_SIZE", "2"))
        rank = self.node_port - 29600 + 1
        
        rpc.init_rpc(
            self.node_id,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=self.init_method,
                _transports=["uv"]
            )
        )
        logger.info(f"Node {self.node_id} RPC initialized with rank {rank}/{world_size}.")

    def stop(self):
        logger.info(f"Shutting down RPC for node {self.node_id}")
        rpc.shutdown()
