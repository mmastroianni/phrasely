import numpy as np
import pandas as pd

from phrasely.pipeline import run_pipeline

# ------------------------------------------------------------
# Stub Components (extremely lightweight, no external deps)
# ------------------------------------------------------------


class StubLoader:
    """Returns a tiny, deterministic DataFrame."""

    def load(self):
        return pd.DataFrame({"phrase": ["alpha", "beta", "gamma"]})


class StubEmbedder:
    """Maps phrases to tiny deterministic embeddings."""

    def embed(self, phrases, dataset_name="unused"):
        # 3 phrases → 3x4 embedding matrix
        n = len(phrases)
        rng = np.random.default_rng(0)
        return rng.normal(size=(n, 4)).astype(np.float32)


class StubReducer:
    """A reducer matching the ReducerProtocol (2-D output)."""

    n_components = 2

    def reduce(self, X):
        # Just slice the first 2 dims
        return X[:, :2].copy()


class StubClusterer:
    """Assigns clusters deterministically."""

    def cluster(self, X):
        # alpha→0, beta→0, gamma→1
        return np.array([0, 0, 1], dtype=int)


class StubMedoids:
    """Select medoids deterministically."""

    def select(self, phrases, reduced, labels):
        # cluster 0 medoid = alpha, cluster 1 medoid = gamma
        return [0, 2], [phrases[0], phrases[2]]


# ------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------


def test_pipeline_smoke():
    """Pipeline should run end-to-end with stubs and return correct structure."""

    res = run_pipeline(
        loader=StubLoader(),
        embedder=StubEmbedder(),
        reducer=StubReducer(),
        clusterer=StubClusterer(),
        medoid_selector=StubMedoids(),
        use_gpu=False,
        stream=False,
    )

    # basic shape checks
    assert res.phrases == ["alpha", "beta", "gamma"]
    assert res.reduced.shape == (3, 2)
    assert (res.labels == np.array([0, 0, 1])).all()

    # medoid sanity
    assert res.medoids == ["alpha", "gamma"]
    assert res.medoid_indices == [0, 2]

    # dimensionality metadata
    assert res.orig_dim == 4
