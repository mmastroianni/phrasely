# src/phrasely/reduction/two_stage_reducer.py

import logging
from typing import Tuple, Type

import numpy as np

from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.utils import gpu_utils

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cuml.manifold import UMAP as GPUUMAP

    _GPU_UMAP_IMPORTED = True
except Exception:
    _GPU_UMAP_IMPORTED = False
    GPUUMAP = None
    cp = None

try:
    import umap

    CPUUMAP = umap.UMAP
except Exception:
    CPUUMAP = None


class TwoStageReducer:
    def __init__(
        self,
        svd_components: int = 100,
        umap_components: int = 15,
        use_gpu: bool = True,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        self.svd_components = int(svd_components)
        self.umap_components = int(umap_components)
        self.use_gpu = bool(use_gpu)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = metric
        self.random_state = int(random_state)

        self.n_components = self.umap_components

    def _select_umap_backend(self) -> Tuple[str, Type | None]:
        gpu_ok = self.use_gpu and _GPU_UMAP_IMPORTED and gpu_utils.is_gpu_available()
        if gpu_ok:
            return "GPU", GPUUMAP
        return "CPU", CPUUMAP

    def reduce(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"TwoStageReducer expected numpy.ndarray, got {type(X)}")

        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        logger.info(
            "TwoStageReducer: X=%s, use_gpu=%s, SVD=%d → UMAP=%d, seed=%d",
            tuple(X.shape),
            self.use_gpu,
            self.svd_components,
            self.umap_components,
            self.random_state,
        )

        svd = SVDReducer(
            n_components=self.svd_components,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )
        X_svd = svd.reduce(X)

        backend_name, UMAPClass = self._select_umap_backend()
        logger.info("TwoStageReducer: UMAP stage on %s backend…", backend_name)

        if UMAPClass is None:
            raise RuntimeError(f"UMAP backend {backend_name} unavailable.")

        try:
            if backend_name == "GPU":
                X_gpu = cp.asarray(X_svd) if cp is not None else X_svd
                um = UMAPClass(
                    n_components=self.umap_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=self.random_state,
                )
                Y_gpu = um.fit_transform(X_gpu)
                Y = (
                    cp.asnumpy(Y_gpu)
                    if cp is not None and hasattr(cp, "asnumpy")
                    else np.array(Y_gpu)
                )
                return Y.astype(np.float32, copy=False)

            if CPUUMAP is None:
                raise RuntimeError("CPU UMAP unavailable.")

            um = CPUUMAP(
                n_components=self.umap_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
            )
            Y = um.fit_transform(X_svd)
            return Y.astype(np.float32, copy=False)

        except Exception as e:
            logger.warning(
                "UMAP %s backend failed (%s). Falling back to CPU UMAP…",
                backend_name,
                e,
            )

            if CPUUMAP is None:
                raise RuntimeError("UMAP unavailable on both GPU and CPU.") from e

            um = CPUUMAP(
                n_components=self.umap_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
            )
            Y = um.fit_transform(X_svd)
            return Y.astype(np.float32, copy=False)
