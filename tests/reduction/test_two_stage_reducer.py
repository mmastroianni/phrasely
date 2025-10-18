import numpy as np

from phrasely.reduction.two_stage_reducer import TwoStageReducer


def test_two_stage_reducer_shapes():
    X = np.random.rand(300, 384)
    reducer = TwoStageReducer(svd_components=50, umap_components=10, use_gpu=False)
    Y = reducer.reduce(X)
    assert Y.shape == (300, 10)
