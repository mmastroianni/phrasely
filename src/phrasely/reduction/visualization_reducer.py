import logging

import numpy as np

try:
    import cuml.manifold.umap as cuml_umap

    HAS_GPU_UMAP = True
except Exception:
    HAS_GPU_UMAP = False

import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class VisualizationReducer:
    """
    Performs low-dimensional reduction for visualization (2D).

    Supports:
    - UMAP (GPU or CPU)
    - t-SNE
    - PCA fallback

    Parameters
    ----------
    method : {"umap", "tsne", "pca"}
        Reduction method to use.
    n_components : int, default=2
        Target dimension for visualization.
    use_gpu : bool, default=False
        Whether to prefer GPU backends (for UMAP if available).
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : dict
        Additional parameters passed to the underlying reducer.
    """

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        use_gpu: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        self.method = method.lower()
        self.n_components = n_components
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.kwargs = kwargs

        logger.info(
            f"VisualizationReducer: method={self.method}, "
            f"GPU={self.use_gpu}, seed={self.random_state}"
        )

    # ------------------------------------------------------------------

    def _init_umap(self):
        """Initialize UMAP reducer (GPU or CPU)."""
        clean_kwargs = {k: v for k, v in self.kwargs.items() if k != "n_components"}

        if self.use_gpu and HAS_GPU_UMAP:
            try:
                reducer = cuml_umap.UMAP(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    **clean_kwargs,
                )
                logger.warning(
                    "⚠️  GPU UMAP may be nondeterministic even with a fixed "
                    + "random_state."
                )
                return reducer
            except Exception as e:
                logger.warning(f"GPU UMAP unavailable ({e}); using CPU fallback.")

        return umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            **clean_kwargs,
        )

    # ------------------------------------------------------------------

    def reduce(self, X: np.ndarray) -> np.ndarray:
        """Run dimensionality reduction and return a 2D embedding."""
        try:
            if self.method == "umap":
                reducer = self._init_umap()
                Y = reducer.fit_transform(X)
            elif self.method == "tsne":
                reducer = TSNE(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    **self.kwargs,
                )
                Y = reducer.fit_transform(X)
            elif self.method == "pca":
                reducer = PCA(
                    n_components=self.n_components, random_state=self.random_state
                )
                Y = reducer.fit_transform(X)
            else:
                raise ValueError(f"Unknown visualization method: {self.method}")

        except Exception as e:
            logger.warning(
                f"Visualization reducer '{self.method}' failed ({e}); "
                + "falling back to PCA."
            )
            reducer = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
            Y = reducer.fit_transform(X)

        # Normalize for consistent plotting
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-9)
        logger.info(
            f"VisualizationReducer: reduced {X.shape[1]} → {self.n_components} dims, "
            f"mean={Y.mean(axis=0)}, std={Y.std(axis=0)}"
        )
        return Y
