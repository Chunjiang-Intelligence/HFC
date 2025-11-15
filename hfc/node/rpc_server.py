import logging
import torch.distributed.rpc as rpc
from .service import NodeService

logger = logging.getLogger(__name__)

class NodeRPCServer:
    def __init__(self, service: NodeService, node_id: str, node_ip: str, node_port: int, world_size: int, orchestrator_host: str, orchestrator_port: int):
        self.service = service
        self.node_id = node_id
        self.node_ip = node_ip
        self.node_port = node_port
        self.world_size = world_size # This is a placeholder, actual world_size comes from orchestrator
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        
        # All nodes connect to the orchestrator to initialize the RPC framework
        self.init_method = f"tcp://{self.orchestrator_host}:{self.orchestrator_port + 1}"
        
    def start(self):
        logger.info(f"Initializing RPC for node {self.node_id} via {self.init_method}")
        rpc.init_rpc(
            self.node_id,
            # Rank must be unique. Here we use a temporary placeholder.
            # A more robust system might use a dynamic rank assignment service.
            # For simplicity, we assume node_port can serve as a unique identifier for now.
            rank=self.node_port - 29501 + 1, # Hacky way to get a unique rank for init
            world_size=self.world_size + 1, # All nodes + orchestrator
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=self.init_method,
                _transports=["uv"]
            )
        )
        logger.info(f"Node {self.node_id} RPC initialized.")

    def stop(self):
        logger.info(f"Shutting down RPC for node {self.node_id}")
        rpc.shutdown()
