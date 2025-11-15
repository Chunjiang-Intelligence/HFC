import numpy as np
import logging
from sklearn.cluster import SpectralClustering
import random

logger = logging.getLogger(__name__)

class GroupingAlgorithm:
    def __init__(self, strategy='spectral'):
        if strategy not in ['spectral', 'random']:
            raise ValueError("Unsupported grouping strategy")
        self.strategy = strategy

    def form_groups(self, node_ids: list, ping_matrix: np.ndarray, k: int) -> list:
        num_nodes = len(node_ids)
        if num_nodes == 0:
            return []
        
        num_groups = max(1, num_nodes // k)
        
        if self.strategy == 'spectral':
            try:
                # We use affinity matrix (inverse of latency)
                affinity_matrix = np.exp(-ping_matrix / ping_matrix.mean())
                
                clustering = SpectralClustering(
                    n_clusters=num_groups,
                    assign_labels='kmeans',
                    affinity='precomputed',
                    random_state=0
                )
                labels = clustering.fit_predict(affinity_matrix)
                
                groups = [[] for _ in range(num_groups)]
                for i, node_id in enumerate(node_ids):
                    groups[labels[i]].append(node_id)
                
                # Filter out empty groups if any
                return [g for g in groups if g]

            except Exception as e:
                logger.warning(f"Spectral clustering failed: {e}. Falling back to random grouping.")
                return self._random_grouping(node_ids, num_groups)
        else:
            return self._random_grouping(node_ids, num_groups)

    def _random_grouping(self, node_ids: list, num_groups: int) -> list:
        random.shuffle(node_ids)
        groups = [[] for _ in range(num_groups)]
        for i, node_id in enumerate(node_ids):
            groups[i % num_groups].append(node_id)
        return groups
