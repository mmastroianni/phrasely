import logging
import numpy as np

from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)

# --- Try GPU backend (RAPIDS cuML) ---
try:
    import cupy as cp
    from cuml.manifold import UMAP as GPUUMAP

    GPU_IMPORTED = True
except Exception as e:
    GPUUMAP = None
    cp = None
    GPU_IMPORTED = False
    logger.warning(f"cuML UMAP import failed: {e}")

# --- CPU fallback ---
try:
    import umap
    from umap import UMAP as CPUUMAP
except Exception as e:
    CPUUMAP = None
    logger.warning(f"UMAP (CPU) import failed: {e}")


class UMAPReducer:
    """
    Reduces embedding dimensionality using UMAP (GPU or CPU).

    - Uses cuML.manifold.UMAP when GPU is available and requested.
    - Falls back cleanly to umap-learn on CPU.
    - Handles conversion between NumPy and CuPy automatically.

    Example:
        reducer = UMAPReducer(n_neighbors=15, n_components=2, use_gpu=True)
        reduced = reducer.reduce(embeddings)
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.random_state = random_state

        gpu_ok = GPU_IMPORTED and is_gpu_available()
        if use_gpu and not gpu_ok:
            logger.warning("GPU requested but unavailable → falling back to CPU.")
        self.use_gpu = use_gpu and gpu_ok

        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"UMAPReducer initialized with {backend} backend.")

    # ------------------------------------------------------------------
    def reduce(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"UMAPReducer expected numpy.ndarray, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"UMAPReducer expected 2D array, got shape={X.shape}")

        try:
            if self.use_gpu and GPUUMAP is not None:
                logger.info("Running cuML UMAP on GPU...")
                X_gpu = cp.asarray(X)
                reducer = GPUUMAP(
                    n_neighbors=self.n_neighbors,
                    n_components=self.n_components,
                    min_dist=self.min_dist,
                    random_state=self.random_state,
                )
                X_reduced = reducer.fit_transform(X_gpu)
                reduced = cp.asnumpy(X_reduced)
            else:
                logger.info("Running umap-learn on CPU...")
                reducer = CPUUMAP(
                    n_neighbors=self.n_neighbors,
                    n_components=self.n_components,
                    min_dist=self.min_dist,
                    random_state=self.random_state,
                )
                reduced = reducer.fit_transform(X)

            logger.info(f"UMAPReducer: reduced to {self.n_components} dimensions.")
            return reduced

        except Exception as e:
            logger.warning(f"UMAPReducer failed: {e}. Returning original input.")
            return X
