import logging

import numpy as np
from sklearn.metrics import pairwise_distances

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)


def compute_dbcv(embeddings, labels):
    """
    Compute an approximate Density-Based Cluster Validation (DBCV) score.

    DBCV measures how well-defined density-based clusters are.
    Returns a float in [-1, 1], where higher is better.

    This implementation approximates DBCV using intra-cluster vs. nearest
    inter-cluster distances. Noise points (label == -1) are ignored.
    """

    if embeddings is None or labels is None:
        logger.warning("DBCV called with None inputs.")
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        logger.warning("DBCV: only one cluster found; returning 0.0.")
        return 0.0

    # Move to CPU for sklearn pairwise_distances
    if cp and isinstance(embeddings, cp.ndarray):
        X = cp.asnumpy(embeddings)
    else:
        X = embeddings

    try:
        D = pairwise_distances(X)
    except Exception as e:
        logger.error(f"DBCV distance computation failed: {e}")
        return 0.0

    intra_dists = []
    inter_dists = []

    for lbl in unique_labels:
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue

        # intra-cluster mean distance
        intra = D[np.ix_(idx, idx)].mean()

        # nearest inter-cluster distance (min distance to other clusters)
        others = np.where(labels != lbl)[0]
        inter = D[np.ix_(idx, others)].min()

        intra_dists.append(intra)
        inter_dists.append(inter)

    if not intra_dists or not inter_dists:
        logger.warning("DBCV: insufficient clusters for computation.")
        return 0.0

    mean_intra = np.mean(intra_dists)
    mean_inter = np.mean(inter_dists)

    score = 1.0 - (mean_intra / (mean_inter + 1e-9))
    return float(np.clip(score, -1.0, 1.0))
