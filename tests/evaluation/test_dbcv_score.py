import numpy as np
from sklearn.datasets import make_blobs

from phrasely.evaluation.dbcv_score import compute_dbcv


def test_dbcv_random_data_negative():
    X = np.random.rand(10, 4)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    score = compute_dbcv(X, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_dbcv_well_separated_clusters_positive():
    X, labels = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=0)
    score = compute_dbcv(X, labels)
    # Expect positive score for clear clusters
    assert score > 0.4
    assert score <= 1.0
