import numpy as np
import pytest

from phrasely.reduction.svd_reducer import SVDReducer


def test_svd_reducer_reduces_dimensions():
    """Basic reduction should decrease the number of features."""
    X = np.random.rand(100, 50).astype(np.float32)
    reducer = SVDReducer(n_components=10, use_gpu=False)
    X_reduced = reducer.reduce(X)
    assert X_reduced.shape[1] == 10


def test_svd_reducer_too_few_samples(caplog):
    """
    If samples/features too small, should log a warning and
    return unchanged.
    """
    X = np.random.rand(1, 5).astype(np.float32)
    reducer = SVDReducer(n_components=3)
    X_reduced = reducer.reduce(X)
    assert X_reduced.shape == X.shape
    assert "input too small" in caplog.text


def test_svd_reducer_invalid_input():
    """Should raise TypeError if input is not a NumPy array."""
    reducer = SVDReducer()
    with pytest.raises(TypeError):
        reducer.reduce([[1, 2], [3, 4]])


def test_svd_reducer_gpu_fallback(monkeypatch, caplog):
    """If GPU requested, should either use GPU or fall back gracefully."""
    X = np.random.rand(10, 10).astype(np.float32)

    import phrasely.utils.gpu_utils as gpu_utils

    monkeypatch.setattr(gpu_utils, "is_gpu_available", lambda: False)

    reducer = SVDReducer(n_components=5, use_gpu=True)
    X_reduced = reducer.reduce(X)

    assert X_reduced.shape[0] == 10
    # Accept either a fallback or GPU log
    assert "falling back to CPU" in caplog.text or "using GPU backend" in caplog.text


def test_svd_reducer_component_clamping(caplog):
    """If n_components > n_features, it should clamp and log it."""
    X = np.random.rand(20, 8).astype(np.float32)
    reducer = SVDReducer(n_components=50, use_gpu=False)
    X_reduced = reducer.reduce(X)
    assert X_reduced.shape[1] == 7  # n_features - 1
    # Match lowercase log wording
    assert "reducing n_components" in caplog.text
