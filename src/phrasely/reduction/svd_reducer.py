import logging
import numpy as np
from sklearn.decomposition import TruncatedSVD as CPUSVD

logger = logging.getLogger(__name__)

try:
    from cuml.decomposition import TruncatedSVD as GPUSVD
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class SVDReducer:
    """Performs dimensionality reduction using TruncatedSVD (GPU if available)."""

    def __init__(self, n_components: int = 50, use_gpu: bool = False):
        self.n_components = n_components
        self.use_gpu = use_gpu and GPU_AVAILABLE

    def reduce(self, X: np.ndarray) -> np.ndarray:
        # --- Validation ---
        if not isinstance(X, np.ndarray):
            raise TypeError(f"SVDReducer expected numpy.ndarray, got {type(X)}")

        if X.ndim != 2:
            logger.warning(f"SVDReducer: input must be 2D, got shape {X.shape}. Returning original.")
            return X

        n_samples, n_features = X.shape
        if n_samples < 2 or n_features < 2:
            logger.warning(f"SVDReducer: input too small for reduction (samples={n_samples}, features={n_features}).")
            return X

        # --- Adjust component count safely ---
        n_comps = min(self.n_components, n_features - 1, n_samples - 1)
        if n_comps < self.n_components:
            logger.info(f"SVDReducer: reducing n_components from {self.n_components} → {n_comps}.")
        self.n_components = n_comps

        # --- Select backend ---
        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"SVDReducer: using {backend} backend for TruncatedSVD.")

        try:
            if self.use_gpu:
                svd = GPUSVD(n_components=self.n_components, random_state=42)
            else:
                svd = CPUSVD(n_components=self.n_components, random_state=42)

            X_reduced = svd.fit_transform(X)
            logger.info(f"SVDReducer: reduced {X.shape[1]} → {X_reduced.shape[1]} dimensions.")
            return X_reduced

        except Exception as e:
            # --- Graceful fallback ---
            if self.use_gpu:
                logger.warning(f"SVDReducer GPU failed: {e}. Falling back to CPU.")
                try:
                    svd = CPUSVD(n_components=self.n_components, random_state=42)
                    X_reduced = svd.fit_transform(X)
                    logger.info(f"SVDReducer: reduced {X.shape[1]} → {X_reduced.shape[1]} dimensions (CPU fallback).")
                    return X_reduced
                except Exception as e2:
                    logger.warning(f"SVDReducer CPU fallback also failed: {e2}. Returning original input.")
            else:
                logger.warning(f"SVDReducer failed: {e}. Returning original input.")

            return X
