import logging
from time import perf_counter

import numpy as np

try:
    from cuml.decomposition import TruncatedSVD as GPUSVD
    from cuml.manifold import UMAP as GPUUMAP

    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False

from sklearn.decomposition import TruncatedSVD as CPUSVD
from umap import UMAP as CPUUMAP

from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


class TwoStageReducer:
    """
    Two-stage dimensionality reducer: linear SVD followed by non-linear UMAP.

    Stage 1: TruncatedSVD (GPU or CPU)
    Stage 2: UMAP (GPU or CPU)

    Parameters
    ----------
    svd_components : int, default=100
        Number of linear SVD components.
    umap_components : int, default=15
        Number of non-linear UMAP components.
    use_gpu : bool, default=True
        Whether to attempt GPU acceleration (requires RAPIDS/cuML).
    n_neighbors : int, default=15
        Number of neighbors for UMAP graph.
    min_dist : float, default=0.1
        Minimum distance between UMAP embeddings.
    metric : str, default="cosine"
        Distance metric for UMAP.
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
        """Apply SVD → UMAP reduction pipeline."""
        n_rows, n_cols = X.shape
        vram_gb = get_device_info().get("total", 0)
        logger.info(
            f"HybridReducer: input shape={X.shape}, "
            f"GPU={self.use_gpu}, VRAM≈{vram_gb:.1f} GB"
        )

        t0 = perf_counter()

        # --- Stage 1: SVD ---
        if self.use_gpu:
            try:
                svd = (
                    GPUSVD(n_components=self.svd_components))
                X_svd = svd.fit_transform(X)
                logger.info(
                    f"Stage 1 (GPU SVD): reduced {n_cols} → "
                    f"{self.svd_components} dims"
                )
            except Exception as e:
                logger.warning(f"GPU SVD failed ({e}); falling back to CPU.")
                self.use_gpu = False
                svd = CPUSVD(n_components=self.svd_components)
                X_svd = svd.fit_transform(X)
                logger.info(
                    f"Stage 1 (CPU SVD): reduced {n_cols} → "
                    f"{self.svd_components} dims"
                )
        else:
            svd = CPUSVD(n_components=self.svd_components)
            X_svd = svd.fit_transform(X)
            logger.info(
                f"Stage 1 (CPU SVD): reduced {n_cols} → {self.svd_components} dims"
            )

        # --- Stage 2: UMAP ---
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
                    f"{self.svd_components}"
                    f" → {self.umap_components} dims"
                )
            except Exception as e:
                logger.warning(f"GPU UMAP failed ({e}); falling back to CPU.")
                umap = CPUUMAP(
                    n_components=self.umap_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    verbose=False,
                )
                X_umap = umap.fit_transform(X_svd)
                logger.info(
                    f"Stage 2 (CPU UMAP): reduced {self.svd_components} "
                    f"→ {self.umap_components} dims"
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
                f"Stage 2 (CPU UMAP): reduced {self.svd_components}"
                f" → {self.umap_components} dims"
            )

        t_total = perf_counter() - t0
        logger.info(f"HybridReducer complete in {t_total:.2f}s.")
        return np.asarray(X_umap)
