import logging
import numpy as np
from hdbscan import HDBSCAN as CPUHDBSCAN

from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cuml.cluster import HDBSCAN as GPUHDBSCAN

    GPU_IMPORTED = True
except Exception as e:
    GPUHDBSCAN = None
    cp = None
    GPU_IMPORTED = False
    logger.warning(f"cuML HDBSCAN unavailable ({e}) → using CPU only.")


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

        # --- Robust GPU detection ---
        gpu_ok = GPU_IMPORTED
        if not gpu_ok:
            logger.warning("cuML HDBSCAN not importable → CPU only.")
        else:
            # Even if is_gpu_available() says False, cuML may still work
            try:
                import cupy
                n_devices = cupy.cuda.runtime.getDeviceCount()
                gpu_ok = n_devices > 0
            except Exception as e:
                logger.warning(f"GPU availability check failed: {e}")
                gpu_ok = False

        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = use_gpu and gpu_ok

        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"HDBSCANClusterer initialized with {backend} backend.")

    # ------------------------------------------------------------------
    def cluster(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"HDBSCANClusterer expected numpy.ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(
                f"HDBSCANClusterer expected 2D array, got shape={X.shape}"
            )

        n_samples = X.shape[0]
        if n_samples < 2:
            logger.warning(
                f"HDBSCANClusterer: input too small for clustering (samples={n_samples})."
            )
            return np.full(n_samples, -1, dtype=int)

        if X.dtype == np.float16:
            logger.info("Converting float16 → float32 for cuML compatibility.")
            X = X.astype(np.float32)

        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"HDBSCANClusterer: using {backend} backend.")

        try:
            if self.use_gpu and GPUHDBSCAN is not None:
                # --- GPU path ---
                logger.info("Running cuML HDBSCAN on GPU...")
                X_gpu = cp.asarray(X)
                clusterer = GPUHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=(
                        self.min_samples
                        if self.min_samples is not None
                        else self.min_cluster_size
                    ),
                    **self.kwargs,
                )
                labels_gpu = clusterer.fit_predict(X_gpu)
                labels = cp.asnumpy(labels_gpu)
            else:
                # --- CPU path ---
                clusterer = CPUHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=(
                        self.min_samples
                        if self.min_samples is not None
                        else self.min_cluster_size
                    ),
                    core_dist_n_jobs=-1,
                    **self.kwargs,
                )
                labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"HDBSCANClusterer: found {n_clusters} clusters (+ noise).")
            return labels

        except Exception as e:
            # Handle out-of-memory or any GPU failure
            if self.use_gpu and "CUDA" in str(e).upper():
                logger.warning(
                    f"HDBSCANClusterer GPU OOM or error: {e}. Falling back to CPU."
                )
                return self._fallback_cpu(X)

            logger.warning(f"HDBSCANClusterer failed: {e}. Returning all -1 labels.")
            return np.full(n_samples, -1, dtype=int)

    # ------------------------------------------------------------------
    def _fallback_cpu(self, X: np.ndarray) -> np.ndarray:
        """CPU fallback after GPU failure."""
        try:
            clusterer = CPUHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=(
                    self.min_samples
                    if self.min_samples is not None
                    else self.min_cluster_size
                ),
                core_dist_n_jobs=-1,
                **self.kwargs,
            )
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(
                f"HDBSCANClusterer (fallback): found {n_clusters} clusters (+ noise)."
            )
            return labels
        except Exception as e:
            logger.error(f"HDBSCANClusterer CPU fallback failed: {e}")
            return np.full(X.shape[0], -1, dtype=int)
