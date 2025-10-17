import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer


@pytest.fixture
def small_data():
    np.random.seed(0)
    c1 = np.random.normal(0, 0.1, (10, 2))
    c2 = np.random.normal(5, 0.1, (10, 2))
    return np.vstack([c1, c2])


# --- Basic CPU path ----------------------------------------------------------

def test_cpu_clustering_basic(small_data):
    clusterer = HDBSCANClusterer(use_gpu=False, min_cluster_size=3)
    labels = clusterer.cluster(small_data)
    assert labels.shape == (20,)
    assert (labels >= -1).all()


# --- Validation and edge cases ----------------------------------------------

def test_invalid_type():
    clusterer = HDBSCANClusterer()
    with pytest.raises(TypeError):
        clusterer.cluster([[1, 2], [3, 4]])  # not np.ndarray


def test_invalid_dimensionality():
    clusterer = HDBSCANClusterer()
    with pytest.raises(ValueError):
        clusterer.cluster(np.random.rand(10, 5, 2))


def test_too_small_input():
    data = np.random.rand(1, 2)
    clusterer = HDBSCANClusterer()
    labels = clusterer.cluster(data)
    assert np.all(labels == -1)


# --- GPU behavior ------------------------------------------------------------

def test_gpu_requested_but_unavailable(monkeypatch, small_data):
    monkeypatch.setattr("phrasely.clustering.hdbscan_clusterer.GPU_IMPORTED", False)
    monkeypatch.setattr("phrasely.utils.gpu_utils.is_gpu_available", lambda: False)

    clusterer = HDBSCANClusterer(use_gpu=True)
    assert not clusterer.use_gpu, "Should fall back to CPU"
    labels = clusterer.cluster(small_data)
    assert labels.shape == (20,)


def test_gpu_available_and_used(monkeypatch, small_data):
    """Simulate GPU path using a fake GPUHDBSCAN class."""
    fake_gpu = MagicMock()
    fake_gpu_instance = MagicMock()
    fake_gpu_instance.fit_predict.return_value = np.arange(len(small_data))
    fake_gpu.return_value = fake_gpu_instance

    monkeypatch.setattr("phrasely.clustering.hdbscan_clusterer.GPUHDBSCAN", fake_gpu)
    monkeypatch.setattr("phrasely.clustering.hdbscan_clusterer.GPU_IMPORTED", True)
    # ✅ Patch inside the same module namespace so it affects the imported reference
    monkeypatch.setattr(
        "phrasely.clustering.hdbscan_clusterer.is_gpu_available", lambda: True
    )

    clusterer = HDBSCANClusterer(use_gpu=True)
    assert clusterer.use_gpu, "GPU path should be activated"
    labels = clusterer.cluster(small_data)
    assert np.array_equal(labels, np.arange(len(small_data)))
    fake_gpu.assert_called_once()


# --- Fallback behavior -------------------------------------------------------

def test_backend_failure_returns_all_minus_one(monkeypatch, small_data):
    """If the backend raises, return all -1."""
    def raise_error(*args, **kwargs):
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr("phrasely.clustering.hdbscan_clusterer.CPUHDBSCAN", raise_error)
    clusterer = HDBSCANClusterer(use_gpu=False)
    labels = clusterer.cluster(small_data)
    assert np.all(labels == -1)
