import numpy as np

from phrasely.reduction.visualization_reducer import UMAPReducer


def test_umap_reducer_shape_cpu():
    """UMAP (CPU) should reduce correctly to 2D."""
    X = np.random.rand(300, 50).astype(np.float32)
    reducer = UMAPReducer(n_components=2, use_gpu=False)
    Y = reducer.reduce(X)
    assert Y.shape == (300, 2)
    assert not np.isnan(Y).any()
