import numpy as np
from sklearn.datasets import make_blobs

from phrasely.evaluation import ClusterEvaluator


def test_evaluator_all_metrics_run():
    X, labels = make_blobs(n_samples=90, centers=3, cluster_std=0.3, random_state=0)
    ev = ClusterEvaluator(X, labels)
    result = ev.evaluate_all()
    assert set(result.keys()) == {"dbcv", "cohesion", "silhouette"}
    assert all(isinstance(v, float) for v in result.values())


def test_evaluator_subset_metrics():
    X = np.random.rand(10, 3)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    ev = ClusterEvaluator(X, labels)
    result = ev.evaluate_all(metrics=["cohesion"])
    assert list(result.keys()) == ["cohesion"]
