import logging

import numpy as np
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def compute_cohesion(embeddings, labels, max_exact_cluster_size: int = 1000):
    """
    Compute average intra-cluster cohesion (compactness).

    Hybrid strategy:
      • For small clusters (size <= max_exact_cluster_size):
          Use exact mean pairwise distance (O(n^2)).
      • For large clusters:
          Approximate using mean distance to cluster centroid (O(n)).

    Returns:
        float: Average cohesion across clusters (lower = tighter).
    """
    if embeddings is None or labels is None:
        logger.warning("Cohesion called with None inputs.")
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0

    total = 0.0
    count = 0

    for lbl in unique_labels:
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        n = len(idx)
        if n < 2:
            continue

        cluster = embeddings[idx]

        if n <= max_exact_cluster_size:
            # exact pairwise distance
            D = pairwise_distances(cluster)
            cohesion = D.mean()
        else:
            # approximate: distance to centroid
            centroid = cluster.mean(axis=0)
            cohesion = np.linalg.norm(cluster - centroid, axis=1).mean()

        total += cohesion
        count += 1

    if count == 0:
        return 0.0

    result = total / count
    logger.info(f"Cohesion (hybrid) computed over {count} clusters: {result:.4f}")
    return float(result)
