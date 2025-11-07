# src/phrasely/pipeline.py

import gc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.pipeline_result import PipelineResult
from phrasely.reduction import ReducerProtocol, SVDReducer
from phrasely.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


def _ensure_float32(X: np.ndarray) -> np.ndarray:
    if X.dtype != np.float32:
        return X.astype(np.float32, copy=False)
    return X


def _free_gpu_mem() -> None:
    """Clear CUDA cache + Python garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _instantiate(obj_or_cls, kwargs: Dict[str, Any], default_cls):
    """
    Hybrid dependency injection:
    - if obj_or_cls is already an instance → use it
    - if obj_or_cls is a class → instantiate with kwargs
    - if None → instantiate default_cls with kwargs
    """
    if obj_or_cls is None:
        return default_cls(**kwargs)

    if isinstance(obj_or_cls, type):
        return obj_or_cls(**kwargs)

    # assume already instance
    return obj_or_cls


# ======================================================================
# Main Pipeline
# ======================================================================


def run_pipeline(
    *,
    loader: Optional[Any] = None,
    loader_cls: Optional[type] = None,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    embedder: Optional[Any] = None,
    embedder_cls: Optional[type] = None,
    embedder_kwargs: Optional[Dict[str, Any]] = None,
    reducer: Optional[ReducerProtocol] = None,
    reducer_cls: Optional[type] = None,
    reducer_kwargs: Optional[Dict[str, Any]] = None,
    clusterer: Optional[Any] = None,
    clusterer_cls: Optional[type] = None,
    clusterer_kwargs: Optional[Dict[str, Any]] = None,
    medoid_selector: Optional[Any] = None,
    medoid_selector_cls: Optional[type] = None,
    medoid_selector_kwargs: Optional[Dict[str, Any]] = None,
    # Optional execution parameters
    use_gpu: bool = True,
    stream: bool = False,
) -> PipelineResult:
    """
    Clean, dependency-injected pipeline for:
        Load → Embed → Reduce → Cluster → Medoids → Result

    All major components can be passed as instances or classes.
    Defaults to Phrasely's standard implementations.
    """

    # Normalization
    loader_kwargs = loader_kwargs or {}
    embedder_kwargs = embedder_kwargs or {}
    reducer_kwargs = reducer_kwargs or {}
    clusterer_kwargs = clusterer_kwargs or {}
    medoid_selector_kwargs = medoid_selector_kwargs or {}

    logger.info("🚀 Starting Phrasely pipeline…")

    # ------------------------------------------------------
    # GPU info (purely informational; components decide internally)
    # ------------------------------------------------------
    vinfo = get_device_info()
    vram_gb = float(vinfo.get("total", 0) or 0)
    logger.info(f"GPU visible: {torch.cuda.is_available()}, VRAM: {vram_gb:.1f} GB")

    # ------------------------------------------------------
    # Instantiate components (hybrid DI)
    # ------------------------------------------------------
    loader = _instantiate(loader or loader_cls, loader_kwargs, default_cls=None)
    if loader is None:
        raise ValueError("Loader is required: pass loader=instance or loader_cls=class.")

    embedder = _instantiate(embedder or embedder_cls, embedder_kwargs, PhraseEmbedder)
    reducer_obj = _instantiate(reducer or reducer_cls, reducer_kwargs, SVDReducer)
    clusterer = _instantiate(clusterer or clusterer_cls, clusterer_kwargs, HDBSCANClusterer)
    medoid_selector = _instantiate(
        medoid_selector or medoid_selector_cls,
        medoid_selector_kwargs,
        MedoidSelector,
    )

    # ------------------------------------------------------
    # Stage 1: Load + Embed
    # ------------------------------------------------------
    logger.info("▶️  Loading and embedding phrases…")

    phrases: List[str] = []
    emb_batches: List[np.ndarray] = []

    def _embed_batch(batch_phrases: List[str], ds_name: str) -> np.ndarray:
        emb = embedder.embed(batch_phrases, dataset_name=ds_name)
        if emb.shape[0] != len(batch_phrases):
            raise RuntimeError(
                f"Embedding mismatch: got {emb.shape[0]} embeddings "
                + f"for {len(batch_phrases)} phrases."
            )
        return emb

    # Stream mode
    if stream:
        if not hasattr(loader, "stream_load"):
            raise AttributeError("stream=True requires loader.stream_load().")

        for i, df in enumerate(loader.stream_load(), 1):
            batch_phrases = df["phrase"].astype(str).tolist()
            batch_emb = _embed_batch(batch_phrases, f"stream_{i:05d}")

            phrases.extend(batch_phrases)
            emb_batches.append(batch_emb)

            logger.info(
                "Streamed batch %d: +%d phrases (total=%d)", i, len(batch_phrases), len(phrases)
            )

    # Non-stream mode
    else:
        if not hasattr(loader, "load"):
            raise AttributeError("loader must implement .load() when stream=False.")

        df = loader.load()
        all_phrases = df["phrase"].astype(str).tolist()
        all_emb = _embed_batch(all_phrases, "full")

        phrases = all_phrases
        emb_batches = [all_emb]

    if not phrases:
        raise RuntimeError("Pipeline loader returned no phrases.")
    if not emb_batches:
        raise RuntimeError("Pipeline produced no embedding batches.")

    embeddings = np.vstack(emb_batches)
    embeddings = _ensure_float32(embeddings)
    phrases = phrases[: embeddings.shape[0]]

    logger.info("✅ Loaded %d phrases.", len(phrases))
    _free_gpu_mem()

    # ------------------------------------------------------
    # Stage 2: Dimensionality Reduction
    # ------------------------------------------------------
    logger.info("▶️  Reducing dimensions…")

    # reducer_obj satisfies ReducerProtocol
    reduced = reducer_obj.reduce(embeddings)
    reduced = _ensure_float32(reduced)

    orig_dim = embeddings.shape[1]
    reduced_dim = reduced.shape[1]

    logger.info("✅ Reduced %d → %d dims.", orig_dim, reduced_dim)

    del embeddings
    _free_gpu_mem()

    # ------------------------------------------------------
    # Stage 3: Clustering
    # ------------------------------------------------------
    logger.info("▶️  Clustering…")

    labels = clusterer.cluster(reduced)
    if labels.shape[0] != reduced.shape[0]:
        raise RuntimeError("Clusterer returned label count mismatch.")

    _free_gpu_mem()
    logger.info("✅ Clustering complete.")

    # ------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------
    if not (len(phrases) == reduced.shape[0] == labels.shape[0]):
        raise RuntimeError(
            f"Pipeline mismatch: phrases={len(phrases)}, "
            f"reduced={reduced.shape[0]}, labels={labels.shape[0]}"
        )

    # ------------------------------------------------------
    # Stage 4: Medoid selection
    # ------------------------------------------------------
    logger.info("▶️  Selecting medoids…")

    medoid_indices, medoid_phrases = medoid_selector.select(phrases, reduced, labels)

    logger.info(
        "✅ Medoid selection complete (%d clusters, %d medoids).",
        len(set(labels)) - (1 if -1 in labels else 0),
        len(medoid_phrases),
    )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------
    return PipelineResult(
        phrases=phrases,
        reduced=reduced,
        labels=labels,
        medoids=medoid_phrases,
        medoid_indices=medoid_indices,
        embeddings=None,
        orig_dim=orig_dim,
    )
