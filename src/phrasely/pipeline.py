import gc
import logging
from contextlib import contextmanager
from time import perf_counter

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


@contextmanager
def catch_time(label: str):
    """Context manager for timing and logging."""
    logger.info(f"▶️  {label}...")
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.3f}s.")


def run_pipeline(
    loader_cls,
    loader_kwargs=None,
    n_components=100,
    use_gpu=True,
    min_cluster_size=15,
    min_samples=None,
    stream: bool = False,
):
    """
    Run the full embedding → reduction → clustering → medoid pipeline.

    Parameters
    ----------
    loader_cls : class
        Data loader class (e.g., CC100OfflineLoader)
    loader_kwargs : dict, optional
        Arguments for the loader.
    stream : bool, default=False
        If True, uses loader.stream_load() to process data in batches.
    """
    loader_kwargs = loader_kwargs or {}
    logger.info("🚀 Starting Phrasely pipeline...")

    # --- GPU capacity check ---
    vram_gb = get_device_info().get("total", 0)
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    _BASE_ROWS = 200_000
    scale_factor = max(1, 4.0 / max(1, vram_gb)) if vram_gb > 0 else 1.0
    max_rows_gpu = int(_BASE_ROWS / scale_factor)
    logger.info(
        f"Adaptive GPU limits — SVD: {max_rows_gpu:,} rows, "
        + f"HDBSCAN: {max_rows_gpu:,} rows."
    )

    # --- Stage 1: Load + Embed ---
    with catch_time("Loading and embedding phrases"):
        loader = loader_cls(**loader_kwargs)

        if stream:
            all_phrases: list[str] = []
            all_embeddings: list[np.ndarray] = []
            embedder = PhraseEmbedder(device="cuda" if use_gpu else "cpu")

            for i, df in enumerate(loader.stream_load(), 1):
                batch_phrases = df["phrase"].tolist()
                batch_embeddings = embedder.embed(batch_phrases)

                # --- Alignment safeguard ---
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

                # --- Stop early if max_phrases reached ---
                if loader.max_phrases and len(all_phrases) >= loader.max_phrases:
                    logger.info("Reached max_phrases limit; stopping stream.")
                    break

            # --- Stack and trim ---
            total = loader.max_phrases or sum(x.shape[0] for x in all_embeddings)
            embeddings = np.vstack(all_embeddings)[:total]
            phrases = all_phrases[:total]
            logger.info(
                f"Streamed total: {len(phrases):,} phrases, "
                f"{embeddings.shape[0]:,} embeddings."
            )
            del embedder

        else:
            df = loader.load()
            phrases = (
                df["phrase"].tolist() if isinstance(df, pd.DataFrame) else list(df)
            )
            embedder = PhraseEmbedder(device="cuda" if use_gpu else "cpu")
            embeddings = embedder.embed(phrases)
            del embedder

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after embedding.")

    # --- Stage 2: Dimensionality Reduction ---
    with catch_time("Reducing dimensions"):
        reducer = SVDReducer(n_components=n_components, use_gpu=use_gpu)
        orig_dim = embeddings.shape[1] if embeddings is not None else None
        reduced = reducer.reduce(embeddings)
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after SVD reduction.")

    # --- Stage 3: Clustering ---
    if reduced.shape[0] > max_rows_gpu:
        logger.info("Too many rows for available GPU VRAM → forcing CPU clustering.")
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

    # --- Defensive shape check ---
    if not (len(phrases) == reduced.shape[0] == labels.shape[0]):
        raise ValueError(
            f"Length mismatch: phrases={len(phrases)}, "
            f"reduced={reduced.shape[0]}, labels={labels.shape[0]}"
        )

    # --- Stage 4: Medoid selection ---
    with catch_time("Selecting medoids"):
        selector = MedoidSelector(return_indices=True)
        medoid_indices, medoid_phrases = selector.select(phrases, reduced, labels)

    # --- Results ---
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
        embeddings=None,  # we don’t keep full embeddings
        orig_dim=orig_dim,  # 👈 NEW
    )
