import numpy as np
import pytest

from phrasely.evaluation.evaluator import ClusterEvaluator
from phrasely.pipeline_result import PipelineResult


@pytest.fixture
def fake_result():
    phrases = [f"phrase {i}" for i in range(10)]
    embeddings = np.random.randn(10, 5)
    reduced = np.random.randn(10, 2)
    # 3 clusters + noise
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, -1])
    medoids = ["phrase 0", "phrase 3", "phrase 5"]
    return PipelineResult(
        phrases=phrases,
        embeddings=embeddings,
        reduced=reduced,
        labels=labels,
        medoids=medoids,
    )


def test_evaluate_returns_expected_keys(fake_result):
    evaluator = ClusterEvaluator(metric="cosine")
    metrics = evaluator.evaluate(fake_result)

    expected_keys = {
        "n_total",
        "n_clusters",
        "noise_fraction",
        "silhouette",
        "cluster_size_summary",
    }
    assert expected_keys.issubset(metrics.keys())


def test_silhouette_fallback_for_single_cluster():
    phrases = [f"p{i}" for i in range(5)]
    emb = np.random.randn(5, 3)
    red = np.random.randn(5, 2)
    labels = np.array([0, 0, 0, 0, -1])  # only one cluster
    result = PipelineResult(phrases, emb, red, labels, medoids=["p0"])
    evaluator = ClusterEvaluator(metric="cosine")

    metrics = evaluator.evaluate(result)
    assert np.isnan(metrics["silhouette"]) or isinstance(metrics["silhouette"], float)


def test_size_distribution_matches_counts(fake_result):
    evaluator = ClusterEvaluator()
    df = evaluator.size_distribution(fake_result.labels)

    # total rows = total phrases
    assert df["size"].sum() == len(fake_result.labels)
    # noise cluster included
    assert -1 in df["label"].values
