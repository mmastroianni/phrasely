import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from phrasely.data_loading.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class WikitextLoader(BaseLoader):
    """
    Streams short English sentences from the Wikitext-103 dataset.

    - Works fully in streaming mode (no huge downloads)
    - Filters for short phrases (3–12 words)
    - Saves a local Parquet cache after the first run
    """

    def __init__(
        self,
        cache_dir: str = "data_cache",
        sample_size: int | None = None,
        max_phrases: int = 1_000_000,
        min_words: int = 3,
        max_words: int = 12,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.max_phrases = max_phrases
        self.min_words = min_words
        self.max_words = max_words
        self.file_path = self.cache_dir / "wikitext_phrases.parquet"

    def _download_if_needed(self):
        if self.file_path.exists():
            logger.info(f"Using cached Wikitext subset at {self.file_path}")
            return

        logger.info("Streaming Wikitext-103 dataset from Hugging Face...")
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        )

        phrases = []
        progress = tqdm(total=self.max_phrases, desc="Collecting phrases", ncols=90)

        for record in ds:
            text = record["text"].strip().replace("\n", " ")
            for sent in text.split(". "):
                words = sent.split()
                if self.min_words <= len(words) <= self.max_words:
                    phrases.append(sent)
                    progress.update(1)
                    if len(phrases) >= self.max_phrases:
                        break
            if len(phrases) >= self.max_phrases:
                break

        progress.close()
        logger.info(f"Collected {len(phrases):,} short sentences from Wikitext-103.")
        df = pd.DataFrame({"phrase": phrases})
        df.to_parquet(self.file_path)
        logger.info(f"Saved {len(df):,} filtered phrases to {self.file_path}")

    def load(self):
        self._download_if_needed()
        df = pd.read_parquet(self.file_path)
        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42)
        logger.info(f"Loaded {len(df):,} Wikitext phrases.")
        return df["phrase"].tolist()
