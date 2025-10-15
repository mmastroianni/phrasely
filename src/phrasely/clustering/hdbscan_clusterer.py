import logging
import numpy as np
from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

try:
    from cuml.decomposition import TruncatedSVD as GPUTSVD
    GPU_IMPORTED = True
except Exception:
    from sklearn.decomposition import TruncatedSVD as CPUTSVD
    GPU_IMPORTED = False


class SVDReducer:
    """
    Reduces embedding dimensionality using TruncatedSVD (CPU or GPU).

    - User can request GPU (`use_gpu=True`)
    - GPU is used only if available and cuML is importable
    - Automatically clamps n_components to valid range
    """

    def __init__(self, n_components: int = 50, use_gpu: bool = False):
        self.n_components = n_components
        self.user_requested_gpu = use_gpu

        # enable GPU only if explicitly requested AND supported AND available
        self.use_gpu = use_gpu and GPU_IMPORTED and is_gpu_available()

        if use_gpu and not self.use_gpu:
            logger.warning("GPU requested but unavailable → falling back to CPU.")

    def reduce(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"SVDReducer expected 2D ndarray, got {type(X)} with shape {getattr(X, 'shape', None)}")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for dimensionality reduction.")

        # Clamp n_components
        n_components = min(self.n_components, n_features - 1)
        if n_components < self.n_components:
            logger.info(f"Reducing n_components from {self.n_components} → {n_components} to fit data shape.")

        logger.info(f"Running TruncatedSVD on {'GPU' if self.use_gpu else 'CPU'} with n_components={n_components}.")

        try:
            if self.use_gpu:
                svd = GPUTSVD(n_components=n_components)
            else:
                svd = CPUTSVD(n_components=n_components)

            reduced = svd.fit_transform(X)
        except Exception as e:
            logger.warning(f"SVD failed: {e}. Returning original data.")
            reduced = X

        return reduced
