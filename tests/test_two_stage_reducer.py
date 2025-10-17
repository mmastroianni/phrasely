import numpy as np

from phrasely.reduction.hybrid_reducer import HybridReducer


def test_hybrid_reducer_shapes():
    X = np.random.rand(300, 384)
    reducer = HybridReducer(svd_components=50, umap_components=10, use_gpu=False)
    Y = reducer.reduce(X)
    assert Y.shape == (300, 10)
