import logging
from pathlib import Path

import pandas as pd
import requests

from phrasely.data_loading.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PhraseDatasetLoader(BaseLoader):
    """Downloads or loads a large phrase dataset and supports sampling."""

    def __init__(self, cache_dir="data_cache", sample_size=None):
        self.cache_dir = Path(cache_dir)
        self.sample_size = sample_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.cache_dir / "phrases.parquet"

    def _download_if_needed(self):
        if self.file_path.exists():
            logger.info(f"Using cached phrase dataset at {self.file_path}")
            return

        url = "https://conceptnet.s3.amazonaws.com/downloads/2022/edges.csv.gz"
        logger.info(f"Downloading ConceptNet edges from {url} ...")
        tmp_file = self.cache_dir / "edges.csv.gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tmp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info("Processing ConceptNet edges into phrases...")
        df = pd.read_csv(
            tmp_file,
            sep="\t",
            header=None,
            usecols=[2, 3],
            names=["start", "end"],
        )
        phrases = pd.concat([df["start"], df["end"]]).dropna().unique()
        df_out = pd.DataFrame({"phrase": phrases})
        df_out.to_parquet(self.file_path)
        logger.info(f"Saved {len(df_out):,} phrases to {self.file_path}")

    def load(self):
        self._download_if_needed()
        df = pd.read_parquet(self.file_path)
        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
        logger.info(f"Loaded {len(df):,} phrases from cache.")
        return df["phrase"].tolist()
