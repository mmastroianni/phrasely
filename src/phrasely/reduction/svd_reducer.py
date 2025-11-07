"""
SVDReducer – Linear dimensionality reduction with GPU/CPU fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from cuml.decomposition import TruncatedSVD as GPUSVD

    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

from sklearn.decomposition import TruncatedSVD as CPUSVD

from phrasely.utils.gpu_utils import is_gpu_available

logger = logging.getLogger(__name__)


class SVDReducer:
    """
    Linear dimensionality reduction via TruncatedSVD.

    Auto-selects GPU (cuML) if available and permitted,
    falls back to scikit-learn CPU otherwise.

    Parameters
    ----------
    n_components : int
        Number of reduced dimensions.
    use_gpu : bool
        Whether to attempt GPU acceleration.
    """

    def __init__(self, n_components: int = 100, use_gpu: bool = True):
        self.n_components = int(n_components)
        self.use_gpu = bool(use_gpu and _HAS_CUML and is_gpu_available())

    # ------------------------------------------------------------------
    def reduce(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SVD reduction.

        Returns
        -------
        np.ndarray  shape = (n_samples, n_components)
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"SVDReducer expects numpy.ndarray, got {type(X)}")

        # cuML requires float32/float64
        if X.dtype not in (np.float32, np.float64):
            logger.info(f"Converting input {X.dtype} → float32 for SVD.")
            X = X.astype(np.float32, copy=False)

        if self.use_gpu:
            try:
                logger.info("SVDReducer: using GPU SVD.")
                svd = GPUSVD(n_components=self.n_components)
                return svd.fit_transform(X)
            except Exception as e:
                logger.warning(f"GPU SVD failed ({e}); falling back to CPU.")
                # Fall through to CPU path

        logger.info("SVDReducer: using CPU SVD.")
        svd = CPUSVD(n_components=self.n_components)
        return svd.fit_transform(X)
