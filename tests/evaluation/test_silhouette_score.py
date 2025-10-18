import numpy as np
from sklearn.datasets import make_blobs

from phrasely.evaluation.silhouette_score import compute_silhouette


def test_silhouette_positive_for_clear_clusters():
    X, labels = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=0)
    score = compute_silhouette(X, labels)
    assert -1.0 <= score <= 1.0
    assert score > 0.3  # clear clusters => positive silhouette


def test_silhouette_zero_for_single_cluster():
    X = np.random.rand(20, 5)
    labels = np.zeros(20, dtype=int)
    assert compute_silhouette(X, labels) == 0.0


def test_silhouette_zero_for_all_noise():
    X = np.random.rand(20, 5)
    labels = np.full(20, -1)
    assert compute_silhouette(X, labels) == 0.0
