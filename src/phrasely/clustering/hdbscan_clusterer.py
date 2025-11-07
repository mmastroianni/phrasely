import logging
from typing import Optional

import numpy as np
from hdbscan import HDBSCAN as CPUHDBSCAN

from phrasely.utils import gpu_utils
# Expose for monkeypatching in tests
is_gpu_available = gpu_utils.is_gpu_available

logger = logging.getLogger(__name__)

# --- Optional GPU backend ------------------------------------------------------

try:
    import cupy as cp
    from cuml.cluster import HDBSCAN as GPUHDBSCAN  # type: ignore

    GPU_IMPORTED = True
except Exception as e:
    cp = None  # type: ignore
    GPUHDBSCAN = None
    GPU_IMPORTED = False
    logger.warning(f"cuML HDBSCAN unavailable ({e}) → using CPU only.")


# ==============================================================================
#   HDBSCANClusterer
# ==============================================================================


class HDBSCANClusterer:
    """
    Unified HDBSCAN clustering interface with optional GPU acceleration.

    Behavior:
    ---------
    • If use_gpu=True and cuML + CuPy available → run GPU HDBSCAN
    • Otherwise → run CPU hdbscan.HDBSCAN
    • On GPU OOM or failure → fallback to CPU
    """

    def __init__(
        self,
        use_gpu: bool = False,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        **kwargs,
    ):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples) if min_samples is not None else self.min_cluster_size
        self.kwargs = kwargs

        # --- Robust GPU detection ---
        try:
            gpu_ok = bool(GPU_IMPORTED and gpu_utils.is_gpu_available())
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            gpu_ok = False

        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = bool(use_gpu and gpu_ok)

        logger.info(
            "HDBSCANClusterer initialized with %s backend.",
            "GPU" if self.use_gpu else "CPU",
        )

    # ----------------------------------------------------------------------
    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict HDBSCAN clusters.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Input reduced embeddings.

        Returns
        -------
        labels : np.ndarray (N,)
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"HDBSCANClusterer expected numpy.ndarray, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"HDBSCANClusterer expected 2D array, got shape={X.shape}")

        n_samples = X.shape[0]
        if n_samples < 2:
            logger.warning(
                "HDBSCANClusterer: input too small to cluster (samples=%d).",
                n_samples,
            )
            return np.full(n_samples, -1, dtype=int)

        # cuML dislikes fp16 → promote to fp32
        if X.dtype == np.float16:
            logger.info("Converting float16 → float32 for cuML compatibility.")
            X = X.astype(np.float32)

        # =====================================================
        #   GPU PATH
        # =====================================================
        if self.use_gpu and GPUHDBSCAN is not None:
            try:
                logger.info("Running cuML HDBSCAN on GPU...")

                xp = cp if cp is not None else np
                X_gpu = xp.asarray(X)

                clusterer = GPUHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    **self.kwargs,
                )
                labels_gpu = clusterer.fit_predict(X_gpu)

                labels = (
                    cp.asnumpy(labels_gpu)
                    if (cp is not None and hasattr(cp, "asnumpy"))
                    else np.array(labels_gpu)
                )

                self._log_cluster_count(labels)
                return labels

            except Exception as e:
                msg = str(e).lower()
                if "cuda" in msg or "oom" in msg:
                    logger.warning(
                        f"HDBSCANClusterer GPU OOM or CUDA error: {e}. " "Falling back to CPU."
                    )
                else:
                    logger.warning(f"HDBSCANClusterer GPU error: {e}. " "Falling back to CPU.")
                # GPU failure → fall through to CPU

        # =====================================================
        #   CPU PATH
        # =====================================================
        return self._cluster_cpu(X)

    # ----------------------------------------------------------------------
    def _cluster_cpu(self, X: np.ndarray) -> np.ndarray:
        """
        CPU fallback (or primary path).
        """
        try:
            logger.info("Running CPython hdbscan on CPU...")

            clusterer = CPUHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                core_dist_n_jobs=-1,
                **self.kwargs,
            )
            labels = clusterer.fit_predict(X)

            self._log_cluster_count(labels)
            return labels

        except Exception as e:
            logger.error(f"HDBSCANClusterer CPU fallback failed: {e}")
            return np.full(X.shape[0], -1, dtype=int)

    # ----------------------------------------------------------------------
    @staticmethod
    def _log_cluster_count(labels: np.ndarray) -> None:
        """Report number of discovered clusters."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(
            "HDBSCANClusterer: found %d clusters (+ noise).",
            n_clusters,
        )
