import logging

import numpy as np
from sklearn.metrics import pairwise_distances

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)


def compute_dbcv(embeddings, labels, max_samples=5000):
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

    # Subsample if needed
    n = len(embeddings)
    if n > max_samples:
        idx = np.random.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        logger.info(f"DBCV: subsampled {max_samples}/{n} points.")

    if cp and isinstance(embeddings, cp.ndarray):
        embeddings = cp.asnumpy(embeddings)

    try:
        D = pairwise_distances(embeddings)
    except Exception as e:
        logger.error(f"DBCV distance computation failed: {e}")
        return 0.0

    intra_dists = []
    inter_dists = []
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        intra = D[np.ix_(idx, idx)].mean()
        others = np.where(labels != lbl)[0]
        inter = D[np.ix_(idx, others)].min()
        intra_dists.append(intra)
        inter_dists.append(inter)

    if not intra_dists or not inter_dists:
        return 0.0

    mean_intra = np.mean(intra_dists)
    mean_inter = np.mean(inter_dists)
    return float(np.clip(1.0 - (mean_intra / (mean_inter + 1e-9)), -1.0, 1.0))
