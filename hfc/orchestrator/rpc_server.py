import asyncio
import logging
import torch.distributed.rpc as rpc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .service import OrchestratorService


logger = logging.getLogger(__name__)

class OrchestratorRPCServer:
    def __init__(self, service: OrchestratorService, host: str, port: int):
        self.service = service
        self.host = host
        self.port = port
        self.rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            init_method=f'tcp://{self.host}:{self.port + 1}', # Use a different port for setup
            _transports=["uv"]
        )

    async def start(self):
        logger.info(f"Starting Orchestrator RPC server on {self.host}:{self.port}")
        rpc.init_rpc(
            "orchestrator",
            rank=0,
            world_size=1, # Orchestrator is a singleton
            rpc_backend_options=self.rpc_backend_options
        )
        logger.info("Orchestrator RPC initialized.")
        # Start the main service logic in the background
        asyncio.create_task(self.service.run())

    async def stop(self):
        logger.info("Shutting down Orchestrator RPC server.")
        rpc.shutdown()
