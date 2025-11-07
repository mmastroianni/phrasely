import logging
from typing import Any

import numpy as np

try:
    from cuml.decomposition import TruncatedSVD as GPUSVD
    from cuml.manifold import UMAP as GPUUMAP

    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

from sklearn.decomposition import TruncatedSVD as CPUSVD
from umap import UMAP as CPUUMAP

from phrasely.utils.gpu_utils import get_device_info
from phrasely.reduction.reducer_protocol import ReducerProtocol

logger = logging.getLogger(__name__)


class TwoStageReducer(ReducerProtocol):
    """
    Two-stage dimensionality reducer:
        1. Linear reduction via TruncatedSVD
        2. Non-linear reduction via UMAP

    Works on GPU (cuML) when available, falls back to CPU otherwise.
    """

    def __init__(
        self,
        svd_components: int = 100,
        umap_components: int = 15,
        use_gpu: bool = True,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        **kwargs: Any,
    ):
        self.svd_components = svd_components
        self.umap_components = umap_components
        self.use_gpu = bool(use_gpu and _HAS_CUML)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.extra = kwargs  # reserved for future

        # For compatibility with the protocol:
        self.n_components = umap_components

    # ------------------------------------------------------------------

    def reduce(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SVD then UMAP.

        Ensures dtype float32 for GPU compatibility.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape={X.shape}")

        n_rows, n_cols = X.shape
        vram = get_device_info().get("total", 0)

        logger.info(
            f"TwoStageReducer: X=({n_rows}, {n_cols}), "
            f"GPU={self.use_gpu}, SVD={self.svd_components}, UMAP={self.umap_components}, "
            f"VRAM≈{vram:.1f} GB"
        )

        # ---- Ensure float32 or float64 ----
        if X.dtype not in (np.float32, np.float64):
            logger.info(
                f"Converting input from {X.dtype} → float32 for GPU compatibility."
            )
            X = X.astype(np.float32)

        # ==========================================================
        # Stage 1 — SVD (GPU or CPU)
        # ==========================================================
        if self.use_gpu:
            try:
                svd = GPUSVD(n_components=self.svd_components)
                X_svd = svd.fit_transform(X)
                logger.info(
                    f"Stage 1 (GPU SVD): reduced {n_cols} → {self.svd_components}"
                )
            except Exception as e:
                logger.warning(f"GPU SVD failed ({e}) → falling back to CPU.")
                self.use_gpu = False
                svd = CPUSVD(n_components=self.svd_components)
                X_svd = svd.fit_transform(X)
                logger.info(
                    f"Stage 1 (CPU SVD): reduced {n_cols} → {self.svd_components}"
                )
        else:
            svd = CPUSVD(n_components=self.svd_components)
            X_svd = svd.fit_transform(X)
            logger.info(
                f"Stage 1 (CPU SVD): reduced {n_cols} → {self.svd_components}"
            )

        # Ensure correct dtype before UMAP
        if X_svd.dtype not in (np.float32, np.float64):
            X_svd = X_svd.astype(np.float32)

        # ==========================================================
        # Stage 2 — UMAP (GPU or CPU)
        # ==========================================================
        if self.use_gpu:
            try:
                umap = GPUUMAP(
                    n_components=self.umap_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    verbose=False,
                )
                X_umap = umap.fit_transform(X_svd)
                logger.info(
                    f"Stage 2 (GPU UMAP): reduced "
                    f"{self.svd_components} → {self.umap_components}"
                )
            except Exception as e:
                logger.warning(f"GPU UMAP failed ({e}) → falling back to CPU.")
                umap = CPUUMAP(
                    n_components=self.umap_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    verbose=False,
                )
                X_umap = umap.fit_transform(X_svd)
                logger.info(
                    f"Stage 2 (CPU UMAP): reduced "
                    f"{self.svd_components} → {self.umap_components}"
                )
        else:
            umap = CPUUMAP(
                n_components=self.umap_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                verbose=False,
            )
            X_umap = umap.fit_transform(X_svd)
            logger.info(
                f"Stage 2 (CPU UMAP): reduced "
                f"{self.svd_components} → {self.umap_components}"
            )

        # Guarantee numpy return
        X_umap = np.asarray(X_umap, dtype=np.float32)
        return X_umap
