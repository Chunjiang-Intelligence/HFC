import asyncio
import numpy as np
import time
import torch.distributed.rpc as rpc
import logging

logger = logging.getLogger(__name__)

# A simple RPC function for pinging
def ping_rpc_target():
    return True

class TopologyMonitor:
    async def measure_latency_rpc(self, nodes: dict) -> np.ndarray:
        node_ids = list(nodes.keys())
        n = len(node_ids)
        ping_matrix = np.full((n, n), float('inf'))
        np.fill_diagonal(ping_matrix, 0)

        for i, src_id in enumerate(node_ids):
            for j, dst_id in enumerate(node_ids):
                if i == j: continue
                
                try:
                    start_time = time.perf_counter()
                    # RPC call from orchestrator to src, asking src to ping dst
                    # This is complex. A simpler way is for orchestrator to ping all nodes.
                    # Let's simplify: orchestrator pings each node. This is not a full matrix
                    # but a latency vector from orchestrator to nodes.
                    # A true P2P matrix requires each node to ping each other.
                    
                    # For a workable solution, we make nodes ping each other via RPC.
                    src_info = nodes[src_id]['info']
                    dst_info = nodes[dst_id]['info']
                    
                    # This function needs to be defined on the NodeService
                    latency = await rpc.rpc_async(src_info, "hfc.node.service.NodeService.ping_node_rpc", args=(dst_info.name,))
                    ping_matrix[i, j] = latency * 1000 # convert to ms
                except Exception as e:
                    logger.warning(f"Could not ping from {src_id} to {dst_id}: {e}")
        
        # Symmetrize and fill gaps
        ping_matrix = np.minimum(ping_matrix, ping_matrix.T)
        ping_matrix[np.isinf(ping_matrix)] = np.nan
        mean_latency = np.nanmean(ping_matrix)
        if np.isnan(mean_latency): mean_latency = 500 # A high default
        ping_matrix[np.isnan(ping_matrix)] = mean_latency * 2
        
        return ping_matrix
