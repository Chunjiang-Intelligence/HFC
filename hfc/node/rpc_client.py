import torch.distributed.rpc as rpc
import logging

logger = logging.getLogger(__name__)

class NodeRPCClient:
    def __init__(self, orchestrator_host, orchestrator_port):
        self.orchestrator_name = "orchestrator"
        self.orchestrator_addr = f"{orchestrator_host}:{orchestrator_port}"

    async def register(self, node_id: str, node_ip: str, node_port: int):
        try:
            logger.info(f"Attempting to register with orchestrator at {self.orchestrator_addr}")
            await rpc.rpc_async(
                self.orchestrator_name,
                "hfc.orchestrator.service.OrchestratorService.register_node_rpc",
                args=(node_id, node_ip, node_port)
            )
            logger.info(f"Successfully registered as {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register with orchestrator: {e}")
            return False
