import logging

import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def compute_silhouette(embeddings, labels):
    """
    Compute mean silhouette score for all non-noise samples.

    Returns:
        float in [-1, 1], higher is better.
        Returns 0.0 if there's only one cluster or all points are noise.
    """
    if embeddings is None or labels is None:
        logger.warning("Silhouette called with None inputs.")
        return 0.0

    labels = np.asarray(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        logger.warning("Silhouette: fewer than 2 clusters; returning 0.0.")
        return 0.0

    # Exclude noise points (label == -1)
    mask = labels != -1
    if mask.sum() == 0:
        logger.warning("Silhouette: all points are noise; returning 0.0.")
        return 0.0

    try:
        score = silhouette_score(embeddings[mask], labels[mask])
        logger.info(f"Silhouette score computed: {score:.4f}")
        return float(score)
    except Exception as e:
        logger.error(f"Silhouette computation failed: {e}")
        return 0.0
