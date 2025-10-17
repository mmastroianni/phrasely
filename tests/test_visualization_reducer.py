import numpy as np

from phrasely.reduction.visualization_reducer import VisualizationReducer


def test_umap_reducer_shape_cpu():
    """UMAP (CPU) should reduce correctly to 2D."""
    X = np.random.rand(300, 50).astype(np.float32)
    reducer = VisualizationReducer(method="umap", n_components=2, use_gpu=False)
    Y = reducer.reduce(X)
    assert Y.shape == (300, 2)
    assert not np.isnan(Y).any()


def test_tsne_reducer_shape_cpu():
    """t-SNE (CPU) should reduce correctly to 2D."""
    X = np.random.rand(200, 30).astype(np.float32)
    reducer = VisualizationReducer(
        method="tsne", n_components=2, use_gpu=False, n_iter=250
    )
    Y = reducer.reduce(X)
    assert Y.shape == (200, 2)
    assert not np.isnan(Y).any()


def test_fallback_to_pca(monkeypatch):
    """Force an exception and ensure fallback to PCA executes cleanly."""
    X = np.random.rand(100, 20).astype(np.float32)

    reducer = VisualizationReducer(method="umap", n_components=2, use_gpu=False)

    # Monkeypatch the internal UMAP call to raise an exception
    monkeypatch.setattr(
        reducer,
        "_run_umap",
        lambda X: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    Y = reducer.reduce(X)
    # Should still succeed using PCA fallback
    assert Y.shape == (100, 2)
    assert not np.isnan(Y).any()
