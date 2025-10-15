import numpy as np
import pytest
from phrasely.reduction.svd_reducer import SVDReducer


def test_svd_reducer_reduces_dimensions():
    """Basic reduction should decrease the number of features."""
    X = np.random.rand(100, 50).astype(np.float32)
    reducer = SVDReducer(n_components=10, use_gpu=False)
    X_reduced = reducer.reduce(X)
    assert X_reduced.shape[1] == 10


def test_svd_reducer_too_few_samples():
    """If samples/features too small, should return unchanged."""
    X = np.random.rand(1, 5)
    reducer = SVDReducer(n_components=3)
    X_reduced = reducer.reduce(X)
    assert X_reduced.shape == X.shape


def test_svd_reducer_invalid_input():
    """Should raise TypeError if input is not a NumPy array."""
    reducer = SVDReducer()
    with pytest.raises(TypeError):
        reducer.reduce([[1, 2], [3, 4]])


def test_svd_reducer_gpu_flag(monkeypatch):
    """Should fall back to CPU gracefully if GPU path fails."""
    X = np.random.rand(10, 10).astype(np.float32)

    # Monkeypatch GPUSVD to throw error
    from phrasely import reduction
    svd_module = reduction.svd_reducer
    original_gpu = svd_module.GPU_AVAILABLE
    svd_module.GPU_AVAILABLE = True

    class MockGPUSVD:
        def __init__(self, *_, **__): ...
        def fit_transform(self, *_): raise RuntimeError("GPU failed")

    svd_module.GPUSVD = MockGPUSVD

    reducer = svd_module.SVDReducer(n_components=5, use_gpu=True)
    X_reduced = reducer.reduce(X)

    assert isinstance(X_reduced, np.ndarray)
    assert X_reduced.shape[1] <= X.shape[1]

    # Restore GPU flag
    svd_module.GPU_AVAILABLE = original_gpu
