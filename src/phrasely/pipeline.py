import gc
import logging
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

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


# ---- Common reducer interface for mypy ---------------------------------------
class ReducerLike(Protocol):
    def reduce(self, X: np.ndarray) -> np.ndarray: ...


# ---- Helpers -----------------------------------------------------------------
def _ensure_float32(X: np.ndarray) -> np.ndarray:
    if X.dtype != np.float32:
        return X.astype(np.float32, copy=False)
    return X


def _estimated_gpu_hdbscan_limit(n_dims: int, vram_gb: float) -> int:
    """
    Heuristic row cap for GPU HDBSCAN based on reduced dimensionality and VRAM.
    Tuned so that: ~15 dims on ~16 GB → ~750k rows cap.
    Scales roughly linearly with VRAM and inversely with dims.
    """
    dims = max(8, int(n_dims))
    base_cap_at_16gb_15d = 750_000.0
    cap = base_cap_at_16gb_15d * (vram_gb / 16.0) * (15.0 / float(dims))
    # keep within sane bounds
    cap = max(200_000.0, min(cap, 2_000_000.0))
    return int(cap)


def _free_gpu_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---- Main pipeline ------------------------------------------------------------
def run_pipeline(
    *,
    loader_cls,
    loader_kwargs: Optional[Dict] = None,
    reducer: str = "svd",  # "svd" | "two_stage"
    reducer_params: Optional[Dict] = None,
    use_gpu: bool = True,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    stream: bool = False,
    force_gpu: bool = False,
) -> PipelineResult:
    """
    Full pipeline: Load → Embed → Reduce → Cluster → Medoids.

    Parameters
    ----------
    loader_cls : class
        A loader class with either `.load()` → DataFrame/Iterable or `.stream_load()` → generator.
    loader_kwargs : dict
        Arguments for the loader (e.g., S3 bucket/prefix, batch_size, max_files, etc.).
        If it includes `max_phrases`, the pipeline will stop embedding after that many rows.
    reducer : {"svd","two_stage"}
        Choose linear SVD or SVD→UMAP reduction.
    reducer_params : dict
        For "svd": {"n_components": int}
        For "two_stage": {"svd_components": int, "umap_components": int, "n_neighbors": int, "min_dist": float, "metric": str}
    use_gpu : bool
        Prefer GPU for reduction/clustering when possible.
    min_cluster_size : int
        HDBSCAN min cluster size.
    min_samples : Optional[int]
        HDBSCAN min samples.
    stream : bool
        If True, call loader.stream_load() and process in batches.
    force_gpu : bool
        If True, attempt GPU HDBSCAN regardless of adaptive limit (will still fall back on OOM).

    Returns
    -------
    PipelineResult
    """
    loader_kwargs = dict(loader_kwargs or {})
    reducer_params = dict(reducer_params or {})

    logger.info("🚀 Starting Phrasely pipeline...")

    # --- GPU info & adaptive limits
    vinfo = get_device_info()
    vram_gb = float(vinfo.get("total", 0.0))
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    # --- Stage 1: Load + Embed
    phrases: List[str] = []
    emb_batches: List[np.ndarray] = []

    max_phrases_kw = None
    if "max_phrases" in loader_kwargs:
        try:
            max_phrases_kw = int(loader_kwargs["max_phrases"])
        except Exception:
            max_phrases_kw = None  # ignore if malformed

    logger.info("▶️  Loading and embedding phrases...")

    # ✅ Initialize the embedder on the correct device at construction time
    embedder = PhraseEmbedder(
        device="cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )

    loader = loader_cls(**loader_kwargs)

    def _consume_batches(rows: Iterable[Tuple[List[str], np.ndarray]]) -> None:
        nonlocal phrases, emb_batches
        for i, (batch_phrases, batch_emb) in enumerate(rows, 1):
            # defensive alignment
            if batch_emb.shape[0] != len(batch_phrases):
                n = min(batch_emb.shape[0], len(batch_phrases))
                batch_phrases = batch_phrases[:n]
                batch_emb = batch_emb[:n]

            phrases.extend(batch_phrases)
            emb_batches.append(batch_emb)

            logger.info(
                f"Streamed batch {i}: +{len(batch_phrases):,} phrases (total {len(phrases):,})"
            )
            if max_phrases_kw and len(phrases) >= max_phrases_kw:
                logger.info(f"Reached max_phrases={max_phrases_kw} — stopping.")
                break

    if stream and hasattr(loader, "stream_load"):
        # Stream rows from loader, embed per-chunk with distinct cache keys
        count = 0
        for i, df in enumerate(loader.stream_load(), 1):
            batch_phrases = df["phrase"].tolist()
            count += len(batch_phrases)
            ds_name = f"stream_{i:05d}"  # unique cache key per batch
            batch_emb = embedder.embed(batch_phrases, dataset_name=ds_name)
            _consume_batches([(batch_phrases, batch_emb)])
            if max_phrases_kw and len(phrases) >= max_phrases_kw:
                break
    else:
        # Non-stream path (single big frame)
        if not hasattr(loader, "load"):
            raise AttributeError("Loader must implement .load() for non-stream mode.")
        df = loader.load()
        all_phrases = df["phrase"].tolist()
        if max_phrases_kw:
            all_phrases = all_phrases[:max_phrases_kw]
        all_emb = embedder.embed(all_phrases, dataset_name="full")
        _consume_batches([(all_phrases, all_emb)])

    # Finalize embeddings
    if len(emb_batches) == 0 or len(phrases) == 0:
        raise RuntimeError("No phrases/embeddings loaded — check loader and limits.")

    embeddings = np.vstack(emb_batches)
    if max_phrases_kw:
        embeddings = embeddings[:max_phrases_kw]
        phrases = phrases[:max_phrases_kw]

    # memory hygiene
    del emb_batches, embedder
    _free_gpu_mem()
    logger.info("🧹 Freed GPU memory.")
    logger.info("Loading and embedding phrases completed.")

    # --- Stage 2: Dimensionality Reduction
    logger.info("▶️  Reducing dimensions...")

    reducer_obj: ReducerLike
    if reducer == "two_stage":
        reducer_obj = TwoStageReducer(
            svd_components=int(reducer_params.get("svd_components", 100)),
            umap_components=int(reducer_params.get("umap_components", 15)),
            use_gpu=bool(use_gpu),
            n_neighbors=int(reducer_params.get("n_neighbors", 15)),
            min_dist=float(reducer_params.get("min_dist", 0.1)),
            metric=str(reducer_params.get("metric", "cosine")),
        )
        reduced = reducer_obj.reduce(embeddings)
        reduced_dims = reduced.shape[1]
    elif reducer == "svd":
        n_components = int(reducer_params.get("n_components", 100))
        reducer_obj = SVDReducer(n_components=n_components, use_gpu=use_gpu)
        reduced = reducer_obj.reduce(embeddings)
        reduced_dims = n_components
    else:
        raise ValueError("reducer must be 'svd' or 'two_stage'")

    orig_dim = embeddings.shape[1]
    del embeddings
    _free_gpu_mem()
    logger.info("Reducing dimensions completed.")

    # --- Stage 3: Clustering
    logger.info("▶️  Clustering phrases...")

    # Decide CPU/GPU HDBSCAN based on heuristic unless force_gpu=True
    try_gpu_hdbscan = bool(use_gpu and torch.cuda.is_available())
    if try_gpu_hdbscan and not force_gpu:
        cap = _estimated_gpu_hdbscan_limit(reduced_dims, vram_gb)
        if reduced.shape[0] > cap:
            logger.info("Reduced matrix too large for GPU HDBSCAN → using CPU backend.")
            try_gpu_hdbscan = False

    # Ensure dtype suitable for cuML
    reduced = _ensure_float32(reduced)

    clusterer = HDBSCANClusterer(
        use_gpu=try_gpu_hdbscan,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.cluster(reduced)

    _free_gpu_mem()
    logger.info("Clustering phrases completed.")

    # --- Sanity check
    if not (len(phrases) == reduced.shape[0] == labels.shape[0]):
        raise ValueError(
            f"Length mismatch: phrases={len(phrases)}, "
            f"reduced={reduced.shape[0]}, labels={labels.shape[0]}"
        )

    # --- Stage 4: Medoids
    logger.info("▶️  Selecting medoids...")
    selector = MedoidSelector(return_indices=True)
    medoid_indices, medoid_phrases = selector.select(phrases, reduced, labels)
    logger.info("Selecting medoids completed.")

    # --- Results
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
        orig_dim=orig_dim,
    )
