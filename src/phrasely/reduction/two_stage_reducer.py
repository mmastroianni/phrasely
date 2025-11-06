# two_stage_reducer.py
import logging

import numpy as np

try:
    from cuml.decomposition import TruncatedSVD as GPUSVD
    from cuml.manifold import UMAP as GPUUMAP

    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

from sklearn.decomposition import TruncatedSVD as CPUSVD
from umap import UMAP as CPUUMAP

logger = logging.getLogger(__name__)


class TwoStageReducer:
    """
    SVD → UMAP reducer.

    Stage 1: TruncatedSVD (GPU or CPU)
    Stage 2: UMAP (GPU or CPU)

    Parameters
    ----------
    svd_components : int
        Number of linear SVD components.
    umap_components : int
        Final non-linear embedding dim.
    use_gpu : bool
        Attempt GPU both stages.
    n_neighbors : int
    min_dist : float
    metric : str
    """

    def __init__(
        self,
        svd_components: int = 100,
        umap_components: int = 15,
        use_gpu: bool = True,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
    ):
        self.svd_components = svd_components
        self.umap_components = umap_components
        self.use_gpu = use_gpu and _HAS_CUML
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

    # ----------------------------------------------------------
    def reduce(self, X: np.ndarray) -> np.ndarray:
        n_rows, n_cols = X.shape
        logger.info(
            f"TwoStageReducer: X={X.shape}, GPU={self.use_gpu}, "
            f"SVD={self.svd_components}, UMAP={self.umap_components}"
        )

        # ensure dtype
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float32)

        # ============================
        # Stage 1: SVD
        # ============================
        if self.use_gpu:
            try:
                svd = GPUSVD(n_components=self.svd_components)
                X_svd = svd.fit_transform(X)
                logger.info(f"Stage 1 GPU SVD: {n_cols} → {self.svd_components}")
            except Exception as e:
                logger.warning(f"GPU SVD failed ({e}); falling back to CPU.")
                self.use_gpu = False
                svd = CPUSVD(n_components=self.svd_components)
                X_svd = svd.fit_transform(X)
                logger.info(f"Stage 1 CPU SVD: {n_cols} → {self.svd_components}")
        else:
            svd = CPUSVD(n_components=self.svd_components)
            X_svd = svd.fit_transform(X)
            logger.info(f"Stage 1 CPU SVD: {n_cols} → {self.svd_components}")

        if X_svd.dtype not in (np.float32, np.float64):
            X_svd = X_svd.astype(np.float32)

        # ============================
        # Stage 2: UMAP
        # ============================
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
                    f"Stage 2 GPU UMAP: {self.svd_components} → {self.umap_components}"
                )
                return X_umap
            except Exception as e:
                logger.warning(f"GPU UMAP failed ({e}); falling back to CPU.")
                self.use_gpu = False

        umap = CPUUMAP(
            n_components=self.umap_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            verbose=False,
        )
        X_umap = umap.fit_transform(X_svd)
        logger.info(f"Stage 2 CPU UMAP: {self.svd_components} → {self.umap_components}")
        return X_umap
