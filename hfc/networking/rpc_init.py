import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

def initialize_process_groups(world_info, node_id):
    global_rank = world_info['node_to_global_rank'][node_id]
    world_size = world_info['world_size']
    
    if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
         raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set.")

    logger.info(f"[{node_id}] Initializing default process group. Rank {global_rank}/{world_size}")
    dist.init_process_group(
        backend='gloo', # Gloo is better for CPU/heterogeneous environments
        rank=global_rank,
        world_size=world_size
    )
    logger.info(f"[{node_id}] Default process group initialized.")

    process_groups = {'world': dist.group.WORLD}
    my_group_id = -1
    my_group_ranks = []

    for group_id, group_data in world_info['groups'].items():
        if node_id in group_data['nodes']:
            my_group_id = group_id
            my_group_ranks = group_data['ranks']
        
        group = dist.new_group(ranks=group_data['ranks'])
        if node_id in group_data['nodes']:
            process_groups['group'] = group
            
    # Create inter-group (leaders) communicator
    leader_ranks = world_info['leader_ranks']
    if leader_ranks:
        leader_group = dist.new_group(ranks=leader_ranks)
        if global_rank in leader_ranks:
            process_groups['leaders'] = leader_group
            
    logger.info(f"[{node_id}] Created process groups: {list(process_groups.keys())}")
    return process_groups
