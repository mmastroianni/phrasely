import gc
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List, Optional, Protocol, Type

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


# ============================================================
# ✅ Reducer Protocol for mypy
# ============================================================
class Reducer(Protocol):
    def reduce(self, X: np.ndarray) -> np.ndarray: ...


# ============================================================
# Timing helper
# ============================================================
@contextmanager
def catch_time(label: str):
    logger.info(f"▶️  {label}...")
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.3f}s.")


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(
    loader_cls: Type,
    loader_kwargs: Optional[Dict[str, Any]] = None,
    reducer: str = "svd",
    reducer_params: Optional[Dict[str, Any]] = None,
    use_gpu: bool = True,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    stream: bool = False,
) -> PipelineResult:
    """
    Full Phrasely pipeline: Load → Embed → Reduce → Cluster → Medoids
    """

    loader_kwargs = loader_kwargs or {}
    reducer_params = reducer_params or {}

    logger.info("🚀 Starting Phrasely pipeline...")

    # --- GPU capacity check ---
    vram_gb = get_device_info().get("total", 0)
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    # Adaptive limits
    _BASE_ROWS = 200_000
    approx_max = int(_BASE_ROWS * max(1.0, (vram_gb / 4.0)))
    logger.info(
        f"Adaptive GPU limits — SVD ≤ {approx_max:,} rows, " +
        f"HDBSCAN ≤ {approx_max:,} rows."
    )

    # ============================================================
    # 1. Load + Embed
    # ============================================================
    from numpy.typing import NDArray

    with catch_time("Loading and embedding phrases"):
        loader = loader_cls(**loader_kwargs)

        phrases: List[str] = []
        emb_batches: List[NDArray[np.float32]] = []

        embedder = PhraseEmbedder(
            model_name="intfloat/e5-small-v2",
            device="cuda" if use_gpu else "cpu",
            batch_size=32,
            prefer_fp16=True,
        )

        for i, df in enumerate(loader.stream_load(), 1):
            batch_phrases = df["phrase"].tolist()
            batch_emb = embedder.embed(batch_phrases, dataset_name=f"stream_{i}")

            if len(batch_emb) != len(batch_phrases):
                n = min(len(batch_emb), len(batch_phrases))
                batch_emb = batch_emb[:n]
                batch_phrases = batch_phrases[:n]

            phrases.extend(batch_phrases)
            emb_batches.append(batch_emb)

            if loader.max_phrases and len(phrases) >= loader.max_phrases:
                logger.info(f"Reached max_phrases={loader.max_phrases} — stopping.")
                break

        embeddings = np.vstack(emb_batches)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory.")

    # ============================================================
    # 2. Reduction
    # ============================================================
    with catch_time("Reducing dimensions"):

        if reducer == "svd":
            r: Reducer = SVDReducer(
                n_components=reducer_params.get("n_components", 100),
                use_gpu=use_gpu,
            )
        elif reducer == "two_stage":
            r = TwoStageReducer(
                svd_components=reducer_params.get("svd_components", 100),
                umap_components=reducer_params.get("umap_components", 15),
                n_neighbors=reducer_params.get("n_neighbors", 15),
                min_dist=reducer_params.get("min_dist", 0.1),
                metric=reducer_params.get("metric", "cosine"),
                use_gpu=use_gpu,
            )
        else:
            raise ValueError(f"Unknown reducer: {reducer}")

        reduced = r.reduce(embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ============================================================
    # 3. Clustering
    # ============================================================
    with catch_time("Clustering phrases"):

        # GPU safeguard (approx threshold)
        if reduced.shape[0] > approx_max:
            logger.info("Reduced matrix too large for GPU HDBSCAN → using CPU backend.")
            use_gpu = False

        clusterer = HDBSCANClusterer(
            use_gpu=use_gpu,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        labels = clusterer.cluster(reduced)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # ============================================================
    # 4. Medoids
    # ============================================================
    with catch_time("Selecting medoids"):
        selector = MedoidSelector(return_indices=False)
        medoid_phrases = selector.select(phrases, reduced, labels)

    # ============================================================
    # 5. Return everything
    # ============================================================
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(
        f"✅ Pipeline complete: {n_clusters} clusters, {len(medoid_phrases)} medoids."
    )

    return PipelineResult(
        phrases=phrases,
        reduced=reduced,
        labels=labels,
        medoids=medoid_phrases,
        medoid_indices=None,
        embeddings=None,
        orig_dim=embeddings.shape[1],
    )
