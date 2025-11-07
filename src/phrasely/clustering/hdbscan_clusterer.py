import logging
from typing import Optional

import numpy as np
from hdbscan import HDBSCAN as CPUHDBSCAN

from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

# Try GPU backend
try:
    import cupy as cp
    from cuml.cluster import HDBSCAN as GPUHDBSCAN

    GPU_OK = True
except Exception as e:
    logger.warning(f"cuML HDBSCAN import failed: {e}")
    cp = None
    GPUHDBSCAN = None
    GPU_OK = False


class HDBSCANClusterer:
    """
    GPU-first HDBSCAN clusterer.
    Clean backend selection, aggressive diagnostics, stable fallback.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        force_gpu: bool = False,
        **kwargs,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.kwargs = kwargs
        self.force_gpu = force_gpu

        # ===================
        # GPU Eligibility
        # ===================
        logger.info("=== HDBSCAN GPU ELIGIBILITY ===")
        logger.info(f"use_gpu requested: {use_gpu}")
        logger.info(f"force_gpu: {force_gpu}")
        logger.info(f"GPU_OK (import): {GPU_OK}")

        try:
            avail = is_gpu_available()
        except Exception as e:
            logger.warning(f"is_gpu_available() failed: {e}")
            avail = False

        logger.info(f"is_gpu_available(): {avail}")

        gpu_eligible = GPU_OK and avail

        self.use_gpu = bool(use_gpu and gpu_eligible or force_gpu)

        logger.info(f"Final backend selection: {'GPU' if self.use_gpu else 'CPU'}")
        logger.info("================================")

    # -------------------------------------------------------------

    def cluster(self, X: np.ndarray) -> np.ndarray:
        logger.info(f"HDBSCANClusterer.cluster(): X={X.shape}, dtype={X.dtype}")

        if X.ndim != 2:
            raise ValueError(f"HDBSCAN expects 2D matrix, got {X.shape}")

        # cuML requires float32
        if self.use_gpu:
            if X.dtype != np.float32:
                logger.info("Converting embeddings to float32 for GPU compatibility…")
                X = X.astype(np.float32)

        try:
            if self.use_gpu:
                return self._gpu_cluster(X)
            else:
                return self._cpu_cluster(X)
        except Exception as e:
            logger.error(f"GPU clustering failed: {e} → Falling back to CPU.")
            return self._cpu_cluster(X)

    # -------------------------------------------------------------

    def _gpu_cluster(self, X: np.ndarray) -> np.ndarray:
        logger.info("Running cuML HDBSCAN (GPU)…")

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
            if cp and hasattr(cp, "asnumpy")
            else np.array(labels_gpu)
        )

        logger.info(
            f"HDBSCAN(GPU): {len(set(labels)) - (1 if -1 in labels else 0)} clusters"
        )
        return labels

    # -------------------------------------------------------------

    def _cpu_cluster(self, X: np.ndarray) -> np.ndarray:
        logger.info("Running CPU HDBSCAN…")

        clusterer = CPUHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            core_dist_n_jobs=-1,
            **self.kwargs,
        )
        labels = clusterer.fit_predict(X)

        logger.info(
            f"HDBSCAN(CPU): {len(set(labels)) - (1 if -1 in labels else 0)} clusters"
        )
        return labels
