import logging
import numpy as np
from phrasely.utils.gpu_utils import is_gpu_available
from hdbscan import HDBSCAN as CPUHDBSCAN

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import HDBSCAN as GPUHDBSCAN
    GPU_IMPORTED = True
except Exception:
    GPUHDBSCAN = None
    GPU_IMPORTED = False


class HDBSCANClusterer:
    """
    Clusters embeddings using HDBSCAN with optional GPU acceleration.

    - GPU used if available and requested.
    - Falls back gracefully with clear logging.
    - Validates 2D input and returns all -1s on failure instead of random noise.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        **kwargs,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.user_requested_gpu = use_gpu
        self.kwargs = kwargs

        gpu_ok = GPU_IMPORTED and is_gpu_available()
        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = use_gpu and gpu_ok

    def cluster(self, X: np.ndarray) -> np.ndarray:
        # --- Validation ---
        if not isinstance(X, np.ndarray):
            raise TypeError(f"HDBSCANClusterer expected numpy.ndarray, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(
                "HDBSCANClusterer expected 2D array, got "
                + f"shape {getattr(X, 'shape', None)}"
            )

        n_samples = X.shape[0]
        if n_samples < 2:
            logger.warning(
                "HDBSCANClusterer: input too small for clustering "
                f"(samples={n_samples}). Returning all -1s."
            )
            return np.full(n_samples, -1)

        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"HDBSCANClusterer: using {backend} backend.")

        try:
            cluster_cls = GPUHDBSCAN if (self.use_gpu and GPUHDBSCAN) else CPUHDBSCAN
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
            logger.info(f"HDBSCANClusterer: found {n_clusters} clusters (+ noise).")
            return labels

        except Exception as e:
            logger.warning(f"HDBSCANClusterer failed: {e}. Returning all -1 labels.")
            return np.full(n_samples, -1)
