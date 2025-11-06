from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.pipeline_result import PipelineResult
from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.reduction.two_stage_reducer import TwoStageReducer
from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


@contextmanager
def catch_time(label: str):
    """Context manager for timing."""
    logger.info("▶️  %s...", label)
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info("%s completed in %.3fs.", label, elapsed)


def _free_gpu() -> None:
    """Free PyTorch GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Freed GPU memory.")


def run_pipeline(
    *,
    loader_cls: Any,
    loader_kwargs: Dict[str, Any],
    n_components: int = 100,
    reducer: str = "svd",
    use_gpu: bool = True,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    stream: bool = False,
) -> PipelineResult:
    """
    Main end-to-end pipeline:
    load → embed → reduce → cluster → medoids.
    """
    logger.info("🚀 Starting Phrasely pipeline...")

    # ---------- GPU limits ----------
    vram_gb = get_device_info().get("total", 0.0)
    logger.info("Detected GPU VRAM: %.1f GB", vram_gb)

    max_rows_gpu = int(730_000 * max(1.0, vram_gb / 14.0))
    max_rows_gpu = min(max_rows_gpu, 1_500_000)

    logger.info(
        "Adaptive GPU limits — SVD ≤ %s rows, HDBSCAN ≤ %s rows.",
        f"{max_rows_gpu:,}",
        f"{max_rows_gpu:,}",
    )

    # ---------- Stage 1: Loading & embedding ----------
    with catch_time("Loading and embedding phrases"):
        loader = loader_cls(**loader_kwargs)

        if stream:
            phrases: List[str] = []
            emb_batches: List[np.ndarray] = []

            embedder = PhraseEmbedder()
            embedder.device = "cuda" if use_gpu else "cpu"

            for i, df in enumerate(loader.stream_load(), 1):
                batch_ph = df["phrase"].tolist()
                batch_emb = embedder.embed(batch_ph)

                if batch_emb.shape[0] != len(batch_ph):
                    n = min(batch_emb.shape[0], len(batch_ph))
                    batch_ph = batch_ph[:n]
                    batch_emb = batch_emb[:n]

                phrases.extend(batch_ph)
                emb_batches.append(batch_emb)

                logger.info(
                    "Streamed batch %d: +%s phrases",
                    i,
                    f"{len(batch_ph):,}",
                )

                if hasattr(loader, "max_phrases"):
                    mp = getattr(loader, "max_phrases")
                    if mp and len(phrases) >= mp:
                        logger.info("Reached max_phrases=%s — stopping.", mp)
                        break

            embeddings = np.vstack(emb_batches)
            del emb_batches, embedder

        else:
            df = loader.load()
            phrases = df["phrase"].tolist()

            embedder = PhraseEmbedder()
            embedder.device = "cuda" if use_gpu else "cpu"

            embeddings = embedder.embed(phrases)
            del embedder

        _free_gpu()

    if len(phrases) != embeddings.shape[0]:
        raise ValueError("Mismatch: phrases vs embeddings count.")

    # ---------- Stage 2: Reduction ----------
    with catch_time("Reducing dimensions"):
        reducer_obj: SVDReducer | TwoStageReducer

        if reducer == "two_stage":
            reducer_obj = TwoStageReducer(
                svd_components=n_components,
                umap_components=15,
                use_gpu=use_gpu,
            )
        else:
            reducer_obj = SVDReducer(
                n_components=n_components,
                use_gpu=use_gpu,
            )

        reduced = reducer_obj.reduce(embeddings)
        orig_dim = embeddings.shape[1]
        del embeddings
        _free_gpu()

    # ---------- Stage 3: Clustering ----------
    if reduced.shape[0] > max_rows_gpu:
        logger.info(
            "Too many rows (%s) for GPU → switching to CPU.",
            f"{reduced.shape[0]:,}",
        )
        use_gpu = False

    with catch_time("Clustering phrases"):
        clusterer = HDBSCANClusterer(
            use_gpu=use_gpu,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.cluster(reduced)
        del clusterer
        _free_gpu()

    if reduced.shape[0] != labels.shape[0]:
        raise ValueError("Mismatch: reduced vs labels length.")

    # ---------- Stage 4: Medoids ----------
    with catch_time("Selecting medoids"):
        selector = MedoidSelector(return_indices=True)
        medoid_indices, medoid_phrases = selector.select(
            phrases,
            reduced,
            labels,
        )

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(
        "✅ Pipeline complete: %s clusters, %s medoids.",
        n_clusters,
        len(medoid_phrases),
    )

    return PipelineResult(
        phrases=phrases,
        reduced=reduced,
        labels=labels,
        medoids=medoid_phrases,
        medoid_indices=medoid_indices,
        embeddings=None,
        orig_dim=orig_dim,
    )
