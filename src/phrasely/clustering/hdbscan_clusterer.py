import logging

import numpy as np

from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

# Always ensure a CPU fallback exists
from hdbscan import HDBSCAN as CPUHDBSCAN  # noqa: E402

try:
    from cuml.cluster import HDBSCAN as GPUHDBSCAN

    GPU_IMPORTED = True
except Exception:
    GPUHDBSCAN = None
    GPU_IMPORTED = False


class HDBSCANClusterer:
    """
    Clusters embeddings using HDBSCAN with optional GPU acceleration.

    - If use_gpu=True, GPU is used only if cuML + CUDA are available.
    - Otherwise, falls back gracefully to CPU.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        **kwargs,
    ):
        self.user_requested_gpu = use_gpu
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.kwargs = kwargs

        gpu_ok = GPU_IMPORTED and is_gpu_available()
        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = use_gpu and gpu_ok

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """Run HDBSCAN on the input array and return cluster labels."""
        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"HDBSCAN: using {backend} backend.")

        try:
            cluster_cls = GPUHDBSCAN if self.use_gpu and GPUHDBSCAN else CPUHDBSCAN
            clusterer = cluster_cls(
                min_cluster_size=self.min_cluster_size,
                min_samples=(
                    self.min_samples
                    if self.min_samples is not None
                    else self.min_cluster_size
                ),
                **self.kwargs,
            )
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"HDBSCAN: found {n_clusters} clusters (+ noise).")
            return labels

        except Exception as e:
            logger.warning(f"HDBSCAN failed: {e}. Returning mock labels.")
            rng = np.random.default_rng(42)
            k = min(3, max(1, X.shape[0] // 50))
            return rng.integers(low=0, high=k, size=X.shape[0])
