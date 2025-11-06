import gc
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd
import torch

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.embeddings.phrase_embedder import EmbedderConfig, PhraseEmbedder
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.pipeline_result import PipelineResult
from phrasely.reduction.svd_reducer import SVDReducer

# If you want the two-stage reducer, keep/import it and enable via flag
# from phrasely.reduction.hybrid_reducer import TwoStageReducer
from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


@contextmanager
def catch_time(label: str):
    logger.info(f"▶️  {label}...")
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        logger.info(f"{label} completed in {elapsed:.3f}s.")


def _estimate_gpu_hdbscan_capacity(dim: int, vram_gb: float) -> int:
    """
    Rough, conservative capacity estimator (rows) for cuML HDBSCAN memory.
    Tuned so T4 (≈16 GB), dim=100 → ~740k rows allowance.

    Scales inversely with dimension; linear in VRAM.
    """
    if vram_gb <= 0:
        return 0
    base_rows_100d = 740_000.0 * (vram_gb / 16.0)  # anchor to T4
    return int(max(50_000, base_rows_100d * (100.0 / max(1, dim))))


def _free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_pipeline(
    loader_cls: Type,
    loader_kwargs: Optional[Dict] = None,
    *,
    # --- embedding ---
    embed_model: str = "intfloat/e5-small-v2",
    embed_max_len: int = 512,
    embed_batch_size: Optional[int] = None,  # auto if None
    # --- reduction ---
    reducer: str = "svd",  # "svd" or "two_stage"
    svd_components: int = 100,
    # umap_components: int = 15,  # if you enable a two-stage reducer
    # --- clustering ---
    use_gpu_clustering: bool = True,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    # --- loading/streaming ---
    stream: bool = True,
    max_phrases: Optional[int] = None,  # cap total rows when streaming
):
    """
    Full pipeline: Load → Embed → Reduce → Cluster → Medoids.

    - Streams from `loader_cls.stream_load()` if `stream=True`
    - Uses GPU for embeddings by default (fp16), with auto batch size
    - Uses cuML SVD (and optionally UMAP) when available
    - Chooses GPU HDBSCAN only if rows <= estimated capacity
    """
    loader_kwargs = loader_kwargs or {}
    logger.info("🚀 Starting Phrasely pipeline...")

    # --- hardware info ---
    dev = get_device_info()
    vram_gb = float(dev.get("total", 0.0))
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    # This governs HDBSCAN GPU offload; dim will be known after reduction
    # We will decide later once we know rows and dim.
    # ----------------------

    # =========================
    # Stage 1: Load + Embed
    # =========================
    with catch_time("Loading and embedding phrases"):
        loader = loader_cls(**loader_kwargs)

        embedder = PhraseEmbedder(
            EmbedderConfig(
                model_name=embed_model,
                max_length=embed_max_len,
                batch_size=embed_batch_size,  # auto if None
                device="cuda" if torch.cuda.is_available() else "cpu",
                normalize=True,
                use_torch_compile=True,
            )
        )

        if stream:
            all_phrases = []
            all_chunks = []

            total_seen = 0
            for i, df in enumerate(loader.stream_load(), 1):
                phrases = df["phrase"].tolist()

                # Optional global cap
                if max_phrases is not None:
                    remaining = max_phrases - total_seen
                    if remaining <= 0:
                        logger.info("Reached max_phrases limit; stopping stream.")
                        break
                    if len(phrases) > remaining:
                        phrases = phrases[:remaining]

                X = embedder.embed(phrases, dataset_name=None)  # no cache across shards
                # sanity (rare tokenizer edge-cases)
                n = min(len(phrases), X.shape[0])
                phrases = phrases[:n]
                X = X[:n]

                all_phrases.extend(phrases)
                all_chunks.append(X)
                total_seen += n

                logger.info(f"Streamed batch {i}: +{n:,} (total={total_seen:,})")

                if max_phrases is not None and total_seen >= max_phrases:
                    logger.info("Reached max_phrases limit; stopping stream.")
                    break

            embeddings = (
                np.vstack(all_chunks)
                if all_chunks
                else np.empty((0, 0), dtype=np.float32)
            )
            phrases = all_phrases
            del all_chunks

        else:
            df = loader.load()
            phrases = (
                df["phrase"].tolist() if isinstance(df, pd.DataFrame) else list(df)
            )
            if max_phrases is not None:
                phrases = phrases[:max_phrases]
            embeddings = embedder.embed(phrases, dataset_name=None)

        _free_cuda()
        logger.info("🧹 Freed GPU memory after embedding.")

    if embeddings.size == 0:
        raise ValueError("No embeddings produced; check loader or inputs.")

    # =========================
    # Stage 2: Reduction
    # =========================
    with catch_time("Reducing dimensions"):
        if reducer == "svd":
            reducer_obj = SVDReducer(n_components=svd_components, use_gpu=True)
            orig_dim = embeddings.shape[1]
            reduced = reducer_obj.reduce(embeddings)
            red_dim = reduced.shape[1]
            logger.info(f"SVD reduced {orig_dim} → {red_dim} dims.")
        elif reducer == "two_stage":
            # If you enable this, import TwoStageReducer above
            # reducer_obj = TwoStageReducer(
            #     svd_components=svd_components,
            #     umap_components=15,
            #     use_gpu=True,
            #     n_neighbors=15,
            #     min_dist=0.1,
            #     metric="cosine",
            # )
            # reduced = reducer_obj.reduce(embeddings)
            raise NotImplementedError(
                "Enable TwoStageReducer import & section if you want two-stage."
            )
        else:
            raise ValueError("reducer must be 'svd' or 'two_stage'")

        del embeddings
        _free_cuda()
        logger.info("🧹 Freed GPU memory after reduction.")

    # =========================
    # Stage 3: Clustering
    # =========================
    n_rows, red_dim = reduced.shape
    gpu_cap = _estimate_gpu_hdbscan_capacity(red_dim, vram_gb)
    allow_gpu = use_gpu_clustering and (n_rows <= gpu_cap)

    if not allow_gpu and use_gpu_clustering:
        logger.info(
            f"Too many rows for GPU HDBSCAN at {red_dim} dims → "
            f"forcing CPU clustering (rows={n_rows:,} > cap≈{gpu_cap:,})."
        )

    with catch_time("Clustering phrases"):
        clusterer = HDBSCANClusterer(
            use_gpu=allow_gpu,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.cluster(reduced)
        del clusterer
        _free_cuda()
        logger.info("🧹 Freed GPU memory after clustering.")

    if not (len(phrases) == reduced.shape[0] == labels.shape[0]):
        raise ValueError(
            f"Length mismatch: phrases={len(phrases)}, "
            f"reduced={reduced.shape[0]}, labels={labels.shape[0]}"
        )

    # =========================
    # Stage 4: Medoids
    # =========================
    with catch_time("Selecting medoids"):
        selector = MedoidSelector(return_indices=True)  # indices + phrases
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
        embeddings=None,
        orig_dim=red_dim,
    )
