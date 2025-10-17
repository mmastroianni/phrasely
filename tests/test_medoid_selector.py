# tests/test_medoid_selector.py
import numpy as np
import pytest
from phrasely.medoids.medoid_selector import MedoidSelector


def test_cosine_small_cluster_exact():
    phrases = ["a", "b", "c"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.7071, 0.7071],
            [0.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 0])

    selector = MedoidSelector(metric="cosine", exact_threshold=10)
    medoids = selector.select(phrases, embeddings, labels)
    # middle vector should be closest to both extremes
    assert medoids == ["b"]


def test_euclidean_small_cluster_exact():
    phrases = ["x", "y", "z"]
    embeddings = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 0, 0])

    selector = MedoidSelector(metric="euclidean", exact_threshold=10)
    medoids = selector.select(phrases, embeddings, labels)
    # point at 1.0 is the geometric median
    assert medoids == ["y"]


def test_multiple_clusters():
    phrases = ["p1", "p2", "p3", "p4"]
    embeddings = np.array(
        [
            [0.0], [0.1],  # cluster 0
            [5.0], [5.1],  # cluster 1
        ]
    )
    labels = np.array([0, 0, 1, 1])

    selector = MedoidSelector(metric="euclidean", exact_threshold=10, return_indices=True)
    medoids, indices = selector.select(phrases, embeddings, labels)
    assert medoids == ["p1", "p3"]
    assert all(isinstance(i, int) for i in indices)


def test_large_cluster_fallback():
    n = 2000
    phrases = [f"p{i}" for i in range(n)]
    embeddings = np.random.randn(n, 5)
    labels = np.zeros(n, dtype=int)

    selector = MedoidSelector(metric="cosine", exact_threshold=100)
    medoids = selector.select(phrases, embeddings, labels)
    assert len(medoids) == 1
