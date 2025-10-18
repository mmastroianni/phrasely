import logging

from .cohesion import compute_cohesion
from .dbcv_score import compute_dbcv
from .silhouette_score import compute_silhouette

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Unified evaluator for clustering metrics.

    Example:
        evaluator = ClusterEvaluator(embeddings, labels)
        results = evaluator.evaluate_all()
    """

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def evaluate_all(self, metrics=None):
        """
        Compute all (or selected) evaluation metrics.

        Args:
            metrics (list[str], optional):
                Subset of metrics to compute, e.g. ["silhouette", "dbcv"]

        Returns:
            dict[str, float]: Metric names → values
        """
        available = {
            "dbcv": compute_dbcv,
            "cohesion": compute_cohesion,
            "silhouette": compute_silhouette,
        }

        if metrics is None:
            metrics = list(available.keys())

        results = {}
        for name in metrics:
            func = available.get(name)
            if func is None:
                logger.warning(f"Unknown metric '{name}', skipping.")
                continue

            try:
                value = func(self.embeddings, self.labels)
            except Exception as e:
                logger.error(f"Metric '{name}' failed: {e}")
                value = 0.0
            results[name] = value

        msg = ", ".join(f"{k}: {v:.3f}" for k, v in results.items())
        logger.info(f"Evaluation metrics → {msg}")
        return results
