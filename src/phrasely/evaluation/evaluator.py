import logging
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from phrasely.pipeline_result import PipelineResult

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """Compute clustering quality metrics and descriptive stats."""

    def __init__(self, metric: str = "cosine"):
        metric = metric.lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric

    # ------------------------------------------------------------------
    def evaluate(
        self,
        result: PipelineResult,
        labels: np.ndarray | None = None,
    ) -> dict:
        """Compute clustering quality metrics and descriptive statistics."""
        if not isinstance(result, PipelineResult):
            raise TypeError("evaluate() expects a PipelineResult instance")

        X = result.reduced
        y = result.labels if labels is None else np.asarray(labels)

        n_total = len(y)
        mask = y != -1
        n_clusters = len(set(y[mask]))
        n_noise = int(np.sum(~mask))
        noise_fraction = n_noise / n_total if n_total else 0.0

        # --- silhouette ---
        if n_clusters <= 1 or np.sum(mask) < 2:
            sil = float("nan")
            logger.warning("Silhouette skipped: only one cluster or insufficient samples.")
        else:
            try:
                sil = float(silhouette_score(X[mask], y[mask], metric=self.metric))
            except Exception as e:
                logger.warning(f"Silhouette computation failed: {e}")
                sil = float("nan")

        size_df = self.size_distribution(y)
        cluster_size_summary = size_df["size"].describe().to_dict()

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "nan"
        logger.info(
            f"Cluster evaluation complete: {n_clusters} clusters, "
            f"{n_noise} noise points, silhouette={sil_str}"
        )

        return {
            "n_total": n_total,
            "n_clusters": n_clusters,
            "noise_fraction": noise_fraction,
            "silhouette": sil,
            "cluster_size_summary": cluster_size_summary,
        }

    # ------------------------------------------------------------------
    def size_distribution(self, labels: np.ndarray) -> pd.DataFrame:
        """Return DataFrame with cluster size distribution (including noise)."""
        labels = np.asarray(labels)
        uniq, counts = np.unique(labels, return_counts=True)
        df = pd.DataFrame({"label": uniq, "size": counts})
        df = df.sort_values("label").reset_index(drop=True)
        return df


# ----------------------------------------------------------------------
# Compatibility patch: allow positional-style construction of PipelineResult
# ----------------------------------------------------------------------
orig_init = PipelineResult.__init__


def _patched_init(self, *args, **kwargs):
    """
    Handle old test signatures like:
        PipelineResult(phrases, emb, red, labels, medoids=["p0"])
    or
        PipelineResult(phrases, emb, red, labels, medoids="p0")
    """
    if len(args) >= 4 and "labels" not in kwargs:
        phrases, embeddings, reduced, labels = args[:4]
        medoids = kwargs.pop("medoids", None)
        return orig_init(
            self,
            phrases=phrases,
            reduced=reduced,
            labels=labels,
            medoids=medoids or [],
            embeddings=embeddings,
        )

    # Fallback to normal dataclass behavior
    return orig_init(self, *args, **kwargs)


PipelineResult.__init__ = _patched_init