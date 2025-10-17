import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------- GPU / CPU DBCV availability ----------------
try:
    from cuml.metrics import density_based_cluster_validity
    _HAS_GPU_DBCV = True
except Exception:
    _HAS_GPU_DBCV = False

try:
    from hdbscan.validity import validity_index as cpu_validity_index
    _HAS_CPU_DBCV = True
except Exception:
    _HAS_CPU_DBCV = False


class ClusterEvaluator:
    """
    Evaluates clustering quality using the Density-Based Cluster Validity (DBCV) score.

    Supports both GPU (cuML) and CPU (hdbscan) backends, automatically choosing
    the most efficient available option.

    Parameters
    ----------
    use_gpu : bool, default=True
        Prefer GPU backend if available.
    metric : str, default="euclidean"
        Distance metric for validity calculation.
    """

    def __init__(self, use_gpu: bool = True, metric: str = "euclidean"):
        self.use_gpu = bool(use_gpu and _HAS_GPU_DBCV)
        self.metric = metric

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the DBCV (Density-Based Cluster Validity) score.

        Parameters
        ----------
        X : np.ndarray
            Feature or embedding matrix (n_samples, n_features).
        labels : np.ndarray
            Cluster labels from HDBSCAN (noise labeled as -1).

        Returns
        -------
        float
            DBCV score in range [-1, 1]; higher is better.
            NaN if computation is not possible.
        """
        if X is None or labels is None:
            logger.warning("DBCV evaluation skipped: missing inputs.")
            return float("nan")

        # Skip degenerate cases: all noise or one cluster
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            logger.info("DBCV skipped: too few clusters.")
            return float("nan")

        try:
            if self.use_gpu:
                logger.info("Computing DBCV using GPU cuML backend...")
                score = float(
                    density_based_cluster_validity(X, labels, metric=self.metric)
                )
            elif _HAS_CPU_DBCV:
                logger.info("Computing DBCV using CPU hdbscan.validity_index...")
                score = float(cpu_validity_index(X, labels, metric=self.metric))
            else:
                logger.warning("No DBCV backend available.")
                score = float("nan")
        except Exception as e:
            logger.warning(f"DBCV computation failed: {type(e).__name__}: {e}")
            score = float("nan")

        logger.info(f"DBCV score = {score:.3f}")
        return score


# ------------------------- Convenience wrapper -------------------------
def compute_dbcv(X: np.ndarray, labels: np.ndarray, use_gpu: bool = True) -> float:
    """
    Quick helper function for standalone use outside of the ClusterEvaluator class.
    """
    evaluator = ClusterEvaluator(use_gpu=use_gpu)
    return evaluator.evaluate(X, labels)
