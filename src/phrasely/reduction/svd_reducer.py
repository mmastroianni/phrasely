import logging
import numpy as np
from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

# --- Always import CPU backend first ---
from sklearn.decomposition import TruncatedSVD as CPUSVD  # ✅ always available

# --- Try GPU backend (optional) ---
try:
    from cuml.decomposition import TruncatedSVD as GPUSVD
    GPU_IMPORTED = True
except Exception:
    GPUSVD = None
    GPU_IMPORTED = False


class SVDReducer:
    """
    Reduces embedding dimensionality using TruncatedSVD (CPU or GPU).

    - GPU used if available and requested.
    - Falls back gracefully with clear logging.
    - Validates 2D input and clamps n_components to < n_features.
    """

    def __init__(self, n_components: int = 50, use_gpu: bool = False):
        self.n_components = n_components
        self.user_requested_gpu = use_gpu

        gpu_ok = GPU_IMPORTED and is_gpu_available()
        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = use_gpu and gpu_ok

    def reduce(self, X: np.ndarray) -> np.ndarray:
        # --- Validation ---
        if not isinstance(X, np.ndarray):
            raise TypeError(f"SVDReducer expected numpy.ndarray, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"SVDReducer expected 2D array, got shape {getattr(X, 'shape', None)}")

        n_samples, n_features = X.shape
        if n_samples < 2:
            logger.warning(f"SVDReducer: input too small for reduction (samples={n_samples}, features={n_features}).")
            return X

        n_components = min(self.n_components, n_features - 1)
        if n_components < self.n_components:
            logger.info(f"SVDReducer: reducing n_components from {self.n_components} → {n_components}.")

        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"SVDReducer: using {backend} backend for TruncatedSVD.")

        try:
            if self.use_gpu and GPUSVD is not None:
                svd = GPUSVD(n_components=n_components)
            else:
                svd = CPUSVD(n_components=n_components)

            reduced = svd.fit_transform(X)
            logger.info(f"SVDReducer: reduced {n_features} → {n_components} dimensions.")
            return reduced

        except Exception as e:
            logger.warning(f"SVDReducer failed: {e}. Returning original input.")
            return X
