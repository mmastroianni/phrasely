# src/phrasely/reduction/svd_reducer.py

import logging
from typing import Tuple, Type

import numpy as np
from sklearn.decomposition import TruncatedSVD as CPUSVD

from phrasely.utils import gpu_utils

logger = logging.getLogger(__name__)

# Optional GPU backend (cuML)
try:
    import cupy as cp
    from cuml.decomposition import TruncatedSVD as GPUSVD
    _GPU_SVD_IMPORTED = True
except Exception:
    _GPU_SVD_IMPORTED = False
    GPUSVD = None
    cp = None


class SVDReducer:
    """
    Dimensionality reduction via TruncatedSVD with GPU/CPU fallback.

    Test expectations:
      • If input too small → log "input too small" → return unchanged
      • If GPU requested but unavailable → warn "falling back to CPU"
      • If n_components > n_features → clamp to n_features - 1 with log
    """

    n_components: int

    def __init__(
        self,
        n_components: int = 100,
        use_gpu: bool = True,
        random_state: int = 42,
    ):
        self.n_components = int(n_components)
        self.use_gpu = bool(use_gpu)
        self.random_state = int(random_state)

    def _select_backend(self) -> Tuple[str, Type | None]:
        gpu_ok = self.use_gpu and _GPU_SVD_IMPORTED and gpu_utils.is_gpu_available()
        if gpu_ok:
            return "GPU", GPUSVD
        return "CPU", CPUSVD

    def reduce(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"SVDReducer expected numpy.ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"SVDReducer expected 2D array, got shape: {X.shape}")

        # Normalize dtype
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # ✅ 1. Too few samples → return unchanged (test_svd_reducer_too_few_samples)
        # ------------------------------------------------------------------
        if X.shape[0] < 2 or X.shape[1] < 2:
            logger.warning("SVDReducer: input too small; returning unchanged.")
            return X

        # ------------------------------------------------------------------
        # ✅ 2. Clamp n_components > n_features (test_svd_reducer_component_clamping)
        # Clamp to n_features - 1 (ensuring >= 1)
        # ------------------------------------------------------------------
        if self.n_components >= X.shape[1]:
            new_components = max(1, X.shape[1] - 1)
            logger.warning(
                "SVDReducer: reducing n_components from %d to %d",
                self.n_components, new_components
            )

            n_components = new_components
        else:
            n_components = self.n_components

        # ------------------------------------------------------------------
        # Select backend
        # ------------------------------------------------------------------
        backend_name, Backend = self._select_backend()
        if Backend is None:
            raise RuntimeError(f"SVD backend {backend_name} unavailable.")

        # ------------------------------------------------------------------
        # ✅ 3. GPU fallback log if GPU requested but backend is CPU
        # (test_svd_reducer_gpu_fallback)
        # ------------------------------------------------------------------
        if self.use_gpu and backend_name == "CPU":
            logger.warning("SVDReducer: falling back to CPU")

        # Test also accepts: "using GPU backend"
        logger.info("SVDReducer: using %s backend", backend_name)

        # ------------------------------------------------------------------
        # Reduction logic
        # ------------------------------------------------------------------
        try:
            svd = Backend(
                n_components=n_components,
                random_state=self.random_state,
            )

            if backend_name == "GPU":
                X_gpu = cp.asarray(X) if cp is not None else X
                Y_gpu = svd.fit_transform(X_gpu)
                Y = (
                    cp.asnumpy(Y_gpu)
                    if cp is not None and hasattr(cp, "asnumpy")
                    else np.array(Y_gpu)
                )
                return Y.astype(np.float32, copy=False)

            # CPU path
            Y = svd.fit_transform(X)
            return Y.astype(np.float32, copy=False)

        except Exception as e:
            # Secondary fallback (rare)
            logger.warning("SVDReducer: falling back to CPU (%s)", e)
            svd = CPUSVD(
                n_components=n_components,
                random_state=self.random_state,
            )
            Y = svd.fit_transform(X.astype(np.float32, copy=False))
            return Y.astype(np.float32, copy=False)
