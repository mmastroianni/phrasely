import warnings

from phrasely.clustering.hdbscan_clusterer import HDBSCANClusterer
from phrasely.data_loading.csv_loader import CSVLoader
from phrasely.embeddings.phrase_embedder import PhraseEmbedder
from phrasely.medoids.medoid_selector import MedoidSelector
from phrasely.reduction.svd_reducer import SVDReducer
from phrasely.utils.logger import setup_logger
from phrasely.utils.timing import catch_time

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logger(__name__)


def run_pipeline(csv_path: str):
    logger.info("Starting pipeline...")

    with catch_time("Loading phrases"):
        phrases = CSVLoader(csv_path).load()
    if not phrases:
        logger.warning(f"No phrases loaded from {csv_path}. Exiting.")
        return []

    with catch_time("Embedding phrases"):
        embeddings = PhraseEmbedder().embed(phrases)

    with catch_time("Reducing dimensions"):
        reduced = SVDReducer(n_components=50).reduce(embeddings)

    with catch_time("Clustering"):
        cluster_labels = HDBSCANClusterer().cluster(reduced)

    with catch_time("Selecting medoids"):
        medoids = MedoidSelector().select(phrases, reduced, cluster_labels)

    logger.info(f"Selected {len(medoids)} medoids.")
    if medoids:
        logger.info(f"Sample medoids: {medoids[:5]}")

    logger.info("Pipeline complete.")
    return medoids


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python -m src.phrasely.pipeline <path_to_csv>")
    else:
        run_pipeline(sys.argv[1])
