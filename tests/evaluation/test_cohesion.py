import numpy as np
from sklearn.datasets import make_blobs

from phrasely.evaluation.cohesion import compute_cohesion


def test_cohesion_basic_behavior():
    X, labels = make_blobs(n_samples=60, centers=3, cluster_std=0.3, random_state=0)
    cohesion_value = compute_cohesion(X, labels)
    assert isinstance(cohesion_value, float)
    assert cohesion_value > 0.0
    assert cohesion_value < 2.0  # typical upper bound for normalized data


def test_cohesion_returns_zero_for_single_cluster():
    X = np.random.rand(10, 4)
    labels = np.zeros(10, dtype=int)
    assert compute_cohesion(X, labels) == 0.0
