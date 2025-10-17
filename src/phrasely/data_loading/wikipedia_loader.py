import gzip
import io
import logging
from pathlib import Path

import pandas as pd
import requests

from phrasely.data_loading.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class WikipediaLoader(BaseLoader):
    """
    Downloads and loads Wikipedia article titles as short English phrases.

    Source: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz

    - Cached locally after first download
    - Filters titles that are too short, too long, or contain special chars
    - Supports random sampling for large-scale experiments
    """

    def __init__(self, cache_dir: str = "data_cache", sample_size: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.file_path = self.cache_dir / "wikipedia_titles.parquet"

    def _download_if_needed(self):
        if self.file_path.exists():
            logger.info(f"Using cached Wikipedia titles at {self.file_path}")
            return

        url = \
            ("https://dumps.wikimedia.org/enwiki/latest/"
             + "enwiki-latest-all-titles-in-ns0.gz")
        logger.info(f"Downloading Wikipedia titles from {url} ... (≈150 MB compressed)")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Decompress in memory as we stream
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            titles = [line.decode("utf-8").strip() for line in f if line.strip()]

        logger.info(f"Downloaded {len(titles):,} raw titles — filtering...")
        df = pd.DataFrame({"phrase": titles})

        # Basic cleaning
        df = df[
            df["phrase"].str.len().between(3, 80)  # drop very short/long
            & df["phrase"].str.match(
                r"^[A-Za-z0-9 ,()'\-\–\–:;!?.]+$"
            )  # filter symbols
        ]

        df.to_parquet(self.file_path)
        logger.info(f"Saved {len(df):,} clean titles to {self.file_path}")

    def load(self):
        self._download_if_needed()
        df = pd.read_parquet(self.file_path)
        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42)
        logger.info(f"Loaded {len(df):,} Wikipedia titles.")
        return df["phrase"].tolist()
