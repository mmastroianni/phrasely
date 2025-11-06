import gc
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Optional, Type

import numpy as np
import pandas as pd
import torch

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.pipeline_result import PipelineResult
from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 🎯 timing utility
# ---------------------------------------------------------------------
@contextmanager
def catch_time(label: str):
    logger.info(f"▶️  {label}...")
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.3f}s.")


# ---------------------------------------------------------------------
# 🚀 main pipeline
# ---------------------------------------------------------------------
def run_pipeline(
    loader_cls: Type,
    loader_kwargs: Optional[dict] = None,
    n_components: int = 100,
    use_gpu: bool = True,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    stream: bool = False,
):
    """
    Full Phrasely pipeline:
        data → embedding → SVD → HDBSCAN → medoids

    Works in both:
        • streaming mode (S3Loader)
        • offline mode (local Arrow/Parquet)
    """

    loader_kwargs = loader_kwargs or {}
    logger.info("🚀 Starting Phrasely pipeline...")

    # ------------------------------------------------------------------
    # GPU VRAM capacity-aware limits
    # ------------------------------------------------------------------
    vram_gb = get_device_info().get("total", 0)
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    _BASE_ROWS = 200_000
    scale_factor = max(1, 4.0 / max(1, vram_gb)) if vram_gb > 0 else 1.0

    max_rows_gpu = int(_BASE_ROWS / scale_factor)
    logger.info(
        f"Adaptive GPU limits — SVD: {max_rows_gpu:,} rows, "
        f"HDBSCAN: {max_rows_gpu:,} rows."
    )

    # ------------------------------------------------------------------
    # ✅ Stage 1: Loading + Embedding
    # ------------------------------------------------------------------
    with catch_time("Loading and embedding phrases"):

        loader = loader_cls(**loader_kwargs)

        # ==============================================================
        # STREAMING MODE
        # ==============================================================
        if stream:
            embedder = PhraseEmbedder(device="cuda" if use_gpu else "cpu")

            all_phrases: list[str] = []
            all_embeddings: list[np.ndarray] = []

            max_phrases = loader_kwargs.get("max_phrases", None)

            for i, df in enumerate(loader.stream_load(), 1):
                batch_phrases = df["phrase"].tolist()

                # ✅ IMPORTANT: disable caching in streaming mode
                batch_embeddings = embedder.embed(
                    batch_phrases,
                    dataset_name=None,  # ← no cache in streaming
                )

                # Safety alignment check
                if batch_embeddings.shape[0] != len(batch_phrases):
                    logger.warning(
                        f"⚠️ Embedding size mismatch in batch {i}: "
                        f"{len(batch_phrases)} phrases. "
                        f"Truncating to smallest length."
                    )
                    n = min(batch_embeddings.shape[0], len(batch_phrases))
                    batch_phrases = batch_phrases[:n]
                    batch_embeddings = batch_embeddings[:n]

                all_phrases.extend(batch_phrases)
                all_embeddings.append(batch_embeddings)

                logger.info(f"Streamed batch {i}: {len(batch_phrases):,} phrases")

                if max_phrases and len(all_phrases) >= max_phrases:
                    logger.info("Reached max_phrases limit; stopping stream.")
                    break

            embeddings = np.vstack(all_embeddings)
            phrases = all_phrases[: len(embeddings)]

        # ==============================================================
        # OFFLINE MODE
        # ==============================================================
        else:
            df = loader.load()
            phrases = df["phrase"].tolist()

            embedder = PhraseEmbedder(device="cuda" if use_gpu else "cpu")

            # ✅ allow caching in offline mode
            dataset_name = loader_kwargs.get("dataset_name", "default")

            embeddings = embedder.embed(
                phrases,
                dataset_name=dataset_name,
            )

        # ✅ free VRAM after embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after embedding.")

    # ------------------------------------------------------------------
    # ✅ Stage 2: Dimensionality reduction
    # ------------------------------------------------------------------
    with catch_time("Reducing dimensions"):

        reducer = SVDReducer(n_components=n_components, use_gpu=use_gpu)
        orig_dim = embeddings.shape[1]

        reduced = reducer.reduce(embeddings)
        del embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after SVD reduction.")

    # ------------------------------------------------------------------
    # ✅ Stage 3: Clustering
    # ------------------------------------------------------------------
    if reduced.shape[0] > max_rows_gpu:
        logger.info("Too many rows for GPU VRAM → forcing CPU clustering.")
        use_gpu = False

    with catch_time("Clustering phrases"):

        clusterer = HDBSCANClusterer(
            use_gpu=use_gpu,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        labels = clusterer.cluster(reduced)
        del clusterer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after clustering.")

    # Double-check alignment
    if not (len(phrases) == reduced.shape[0] == labels.shape[0]):
        raise ValueError(
            f"Length mismatch: phrases={len(phrases)}, "
            f"reduced={reduced.shape[0]}, labels={labels.shape[0]}"
        )

    # ------------------------------------------------------------------
    # ✅ Stage 4: Medoids
    # ------------------------------------------------------------------
    with catch_time("Selecting medoids"):

        selector = MedoidSelector(return_indices=True)  # ensure consistent output
        medoid_indices, medoid_phrases = selector.select(phrases, reduced, labels)

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
        embeddings=None,        # we free full embeddings
        orig_dim=orig_dim,
    )
