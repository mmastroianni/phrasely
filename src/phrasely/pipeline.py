import gc
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch

from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.reduction.two_stage_reducer import TwoStageReducer
from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.pipeline_result import PipelineResult
from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
@contextmanager
def catch_time(label: str):
    logger.info(f"▶️  {label}...")
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.3f}s.")


# -------------------------------------------------------------
def _estimate_gpu_limits(vram_gb: float) -> int:
    """
    Estimate safe max rows for SVD + HDBSCAN on GPU.

    Empirically calibrated on:
      • 4 GB GPU → ~100k rows
      • 14.6 GB GPU → ~750k rows
    """
    if vram_gb <= 0:
        return 200_000  # safe CPU fallback

    limit = int(vram_gb * 52_000)
    return max(200_000, min(limit, 1_200_000))  # clamp


# -------------------------------------------------------------
def run_pipeline(
    loader_cls,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    *,
    reducer: str = "svd",                # "svd" or "two_stage"
    reducer_params: Optional[dict] = None,
    use_gpu: bool = True,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    stream: bool = False,
):
    """
    Full pipeline: load → embed → reduce → cluster → medoids.
    """

    loader_kwargs = loader_kwargs or {}
    reducer_params = reducer_params or {}

    logger.info("🚀 Starting Phrasely pipeline...")

    # GPU limits
    vram_gb = get_device_info().get("total", 0.0)
    max_gpu_rows = _estimate_gpu_limits(vram_gb)

    logger.info(
        f"Adaptive GPU limits — SVD ≤ {max_gpu_rows:,} rows, "
        f"HDBSCAN ≤ {max_gpu_rows:,} rows."
    )

    # ================
    # Stage 1: Load + Embed
    # ================
    with catch_time("Loading and embedding phrases"):
        loader = loader_cls(**loader_kwargs)

        phrases: List[str] = []
        emb_batches: List[np.ndarray] = []

        embedder = PhraseEmbedder()
        embedder.device = "cuda" if use_gpu else "cpu"

        if stream:
            count = 0
            for i, df in enumerate(loader.stream_load(), 1):
                batch_phrases = df["phrase"].tolist()
                batch_embeddings = embedder.embed(batch_phrases)

                # alignment fix
                if batch_embeddings.shape[0] != len(batch_phrases):
                    n = min(batch_embeddings.shape[0], len(batch_phrases))
                    batch_phrases = batch_phrases[:n]
                    batch_embeddings = batch_embeddings[:n]

                phrases.extend(batch_phrases)
                emb_batches.append(batch_embeddings)
                count += batch_embeddings.shape[0]

                if loader.max_phrases and count >= loader.max_phrases:
                    logger.info(f"Reached max_phrases={loader.max_phrases} — stopping.")
                    break

        else:
            df = loader.load()
            phrases = df["phrase"].tolist()
            embeddings = embedder.embed(phrases)
            emb_batches = [embeddings]

        embeddings = np.vstack(emb_batches)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory.")

    # ================
    # Stage 2: Reduce
    # ================
    with catch_time("Reducing dimensions"):

        if reducer == "svd":
            red = SVDReducer(
                n_components=reducer_params.get("n_components", 100),
                use_gpu=use_gpu,
            )
        elif reducer == "two_stage":
            red = TwoStageReducer(
                svd_components=reducer_params.get("svd_components", 100),
                umap_components=reducer_params.get("umap_components", 15),
                n_neighbors=reducer_params.get("n_neighbors", 15),
                min_dist=reducer_params.get("min_dist", 0.1),
                metric=reducer_params.get("metric", "cosine"),
                use_gpu=use_gpu,
            )
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

        reduced = red.reduce(embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ================
    # Stage 3: Cluster
    # ================
    if reduced.shape[0] > max_gpu_rows:
        logger.info("Too many rows for GPU clustering → using CPU.")
        use_gpu = False

    with catch_time("Clustering phrases"):
        clusterer = HDBSCANClusterer(
            use_gpu=use_gpu,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.cluster(reduced)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ================
    # Stage 4: Medoids
    # ================
    with catch_time("Selecting medoids"):
        selector = MedoidSelector(return_indices=True)
        medoid_indices, medoid_phrases = selector.select(phrases, reduced, labels)

    # Final
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(
        f"✅ Pipeline complete: {n_clusters} clusters, {len(medoid_phrases)} medoids."
    )

    return PipelineResult(
        phrases=phrases,
        reduced=reduced,
        labels=labels,
        medoids=medoid_phrases,
        medoid_indices=medoid_indices,
        embeddings=None,
        orig_dim=embeddings.shape[1],
    )
