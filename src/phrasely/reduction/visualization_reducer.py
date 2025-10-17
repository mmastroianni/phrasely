import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class VisualizationReducer:
    """Performs 2D or 3D visualization-oriented dimensionality reduction.

    Supports UMAP, t-SNE, and fallback to PCA.
    """

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        use_gpu: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters
        ----------
        method : str
            Reduction method ('umap', 'tsne', or 'pca' fallback).
        n_components : int
            Number of components to reduce to (usually 2 or 3).
        use_gpu : bool
            Whether to prefer GPU-based reducers (if available).
        random_state : int
            Random seed for reproducibility.
        **kwargs :
            Additional keyword arguments passed to t-SNE (e.g. n_iter, perplexity).
        """
        self.method = method
        self.n_components = n_components
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.kwargs = kwargs  # extra params for TSNE etc.

        logger.info(
            f"VisualizationReducer: method={method}, GPU={use_gpu}, seed={random_state}"
        )

    # ------------------------------------------------------------------
    def _init_umap(self):
        """Return a configured UMAP reducer instance."""
        if self.use_gpu:
            try:
                from cuml.manifold import UMAP
            except ImportError:
                from umap import UMAP
        else:
            from umap import UMAP

        return UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            metric="cosine",
        )

    # ------------------------------------------------------------------
    def _run_umap(self, X: np.ndarray) -> np.ndarray:
        """Run UMAP; kept separate for easier test monkeypatching."""
        reducer = self._init_umap()
        return reducer.fit_transform(X)

    # ------------------------------------------------------------------
    def reduce(self, X: np.ndarray) -> np.ndarray:
        """Perform visualization reduction with fallback to PCA."""
        try:
            if self.method == "umap":
                return self._run_umap(X)

            elif self.method == "tsne":
                tsne = TSNE(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    **self.kwargs,  # allow n_iter, perplexity, etc.
                )
                return tsne.fit_transform(X)

            else:
                raise ValueError(f"Unknown visualization method: {self.method}")

        except Exception as e:
            logger.warning(f"VisualizationReducer fallback to PCA due to {e}")
            pca = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )
            return pca.fit_transform(X)
