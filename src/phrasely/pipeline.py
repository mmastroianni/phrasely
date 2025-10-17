import logging
import gc
import torch
from contextlib import contextmanager
from time import perf_counter
from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.utils.gpu_utils import get_device_info
from phrasely.pipeline_result import PipelineResult


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
):
    """Run the full embedding → reduction → clustering → medoid pipeline."""
    loader_kwargs = loader_kwargs or {}
    logger.info("🚀 Starting Phrasely pipeline...")

    # --- GPU capacity check ---
    vram_gb = get_device_info().get("total", 0)
    logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")

    _BASE_ROWS = 200_000
    scale_factor = max(1, 4.0 / max(1, vram_gb)) if vram_gb > 0 else 1.0
    max_rows_gpu = int(_BASE_ROWS / scale_factor)
    logger.info(
        f"Adaptive GPU limits — SVD: {max_rows_gpu:,} rows, HDBSCAN: {max_rows_gpu:,} rows."
    )

    # --- Stage 1: Load data ---
    with catch_time("Loading phrases"):
        loader = loader_cls(**loader_kwargs)
        phrases = loader.load()

    # --- Stage 2: Embed ---
    with catch_time("Embedding phrases"):
        embedder = PhraseEmbedder(device="cuda" if use_gpu else "cpu")
        embeddings = embedder.embed(phrases)
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after embedding.")

    # --- Stage 3: Dimensionality Reduction ---
    with catch_time("Reducing dimensions"):
        reducer = SVDReducer(n_components=n_components, use_gpu=use_gpu)
        reduced = reducer.reduce(embeddings)
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Freed GPU memory after SVD reduction.")

    # --- Stage 4: Clustering ---
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

    # --- Stage 5: Medoid selection ---
    with catch_time("Selecting medoids"):
        selector = MedoidSelector()
        medoids = selector.select(phrases, reduced, labels)

    # --- Results ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"✅ Pipeline complete: {n_clusters} clusters, {len(medoids)} medoids.")

    return PipelineResult(
        phrases=phrases,
        reduced=reduced,
        labels=labels,
        medoids=medoids,
        n_components=n_components,
    )
