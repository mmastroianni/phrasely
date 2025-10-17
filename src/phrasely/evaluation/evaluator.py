import logging

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from phrasely.pipeline_result import PipelineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional imports (DBCV support)
# ---------------------------------------------------------------------
try:
    import hdbscan

    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

try:
    from cuml.cluster.hdbscan import compute_dbcv as gpu_dbcv  # RAPIDS version

    _HAS_GPU_DBCV = True
except Exception:
    _HAS_GPU_DBCV = False


class ClusterEvaluator:
    """
    Computes quality metrics and descriptive stats for clustered embeddings.

    Provides:
      • Silhouette Score (cosine or euclidean)
      • DBCV (Density-Based Clustering Validation) – via hdbscan or cuML
      • Basic size + noise statistics
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    # ---------------------------------------------------
    def evaluate(self, result: PipelineResult) -> dict:
        """Compute evaluation metrics for a pipeline result."""
        labels = np.asarray(result.labels)
        reduced = np.asarray(result.reduced)
        n_total = len(labels)

        # drop noise for metrics
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        noise_frac = 1 - np.mean(mask)

        metrics = {
            "n_total": n_total,
            "n_clusters": int(n_clusters),
            "noise_fraction": float(noise_frac),
        }

        # --- Silhouette ---
        if n_clusters > 1 and np.sum(mask) > 1:
            try:
                metrics["silhouette"] = float(
                    silhouette_score(reduced[mask], labels[mask], metric=self.metric)
                )
            except Exception as e:
                logger.warning(f"Silhouette score failed: {e}")
                metrics["silhouette"] = np.nan
        else:
            metrics["silhouette"] = np.nan

        # --- DBCV ---
        metrics["dbcv"] = self._compute_dbcv(reduced, labels)

        # --- Cluster size stats ---
        unique, counts = np.unique(labels, return_counts=True)
        size_df = pd.DataFrame({"label": unique, "size": counts}).sort_values(
            "size", ascending=False
        )

        metrics["cluster_size_summary"] = {
            "mean": float(size_df["size"].mean()),
            "median": float(size_df["size"].median()),
            "max": int(size_df["size"].max()),
            "min": int(size_df["size"].min()),
        }

        return metrics

    # ---------------------------------------------------
    def _compute_dbcv(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute DBCV score using hdbscan (CPU) or cuML (GPU)."""
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return float("nan")

        try:
            if _HAS_GPU_DBCV:
                import cupy as cp

                X_gpu = cp.asarray(X)
                labels_gpu = cp.asarray(labels)
                score = float(gpu_dbcv(X_gpu, labels_gpu, metric=self.metric))
                logger.info(f"DBCV (GPU): {score:.4f}")
                return score

            elif _HAS_HDBSCAN:
                score = float(
                    hdbscan.validity.validity_index(X, labels, metric=self.metric)
                )
                logger.info(f"DBCV (CPU): {score:.4f}")
                return score

            else:
                logger.warning("DBCV unavailable (no hdbscan or RAPIDS installed).")
                return float("nan")

        except Exception as e:
            logger.warning(f"DBCV computation failed: {e}")
            return float("nan")

    # ---------------------------------------------------
    def size_distribution(self, labels: np.ndarray) -> pd.DataFrame:
        """Return sorted cluster size DataFrame."""
        unique, counts = np.unique(labels, return_counts=True)
        df = pd.DataFrame({"label": unique, "size": counts})
        df = df.sort_values("size", ascending=False).reset_index(drop=True)
        return df
