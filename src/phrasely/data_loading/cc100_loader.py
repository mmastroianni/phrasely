import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from phrasely.data_loading.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class CC100Loader(BaseLoader):
    """
    Streams short English sentences from the CC100 dataset, writing in chunks.

    - Uses Hugging Face Datasets to stream the 'cc100' English subset.
    - Filters for short phrases (3–12 words).
    - Writes every `chunk_size` phrases incrementally to Parquet.
    - Resumes from previous progress if file already exists.
    - Displays live progress with tqdm.
    """

    def __init__(
        self,
        cache_dir: str = "data_cache",
        sample_size: int | None = None,
        max_phrases: int = 1_000_000,
        min_words: int = 3,
        max_words: int = 12,
        chunk_size: int = 100_000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.max_phrases = max_phrases
        self.min_words = min_words
        self.max_words = max_words
        self.chunk_size = chunk_size
        self.file_path = self.cache_dir / "cc100_phrases_en.parquet"

    def _download_if_needed(self):
        # Check existing progress
        start_count = 0
        if self.file_path.exists():
            try:
                existing = pd.read_parquet(self.file_path, columns=["phrase"])
                start_count = len(existing)
                logger.info(f"Found existing file with {start_count:,} phrases.")
                if start_count >= self.max_phrases:
                    logger.info("Dataset already complete — skipping download.")
                    return
            except Exception as e:
                logger.warning(f"Could not read existing file ({e}); starting fresh.")
                self.file_path.unlink(missing_ok=True)

        logger.info("Streaming English CC100 dataset from Hugging Face...")
        disable_caching()
        ds = load_dataset("cc100", "en", split="train", streaming=True)

        phrases = []
        total_written = start_count
        pbar = tqdm(total=self.max_phrases, desc="Collecting phrases", ncols=100)
        pbar.update(start_count)

        for record in ds:
            text = record["text"].strip().replace("\n", " ")
            for sent in text.split(". "):
                words = sent.split()
                if self.min_words <= len(words) <= self.max_words:
                    if total_written + len(phrases) < self.max_phrases:
                        phrases.append(sent)
                        pbar.update(1)
                    else:
                        break

                    # Write chunk every N phrases
                    if len(phrases) >= self.chunk_size:
                        total_written += self._append_chunk(phrases)
                        phrases.clear()

                    if total_written >= self.max_phrases:
                        break
            if total_written >= self.max_phrases:
                break

        # Write any remaining phrases
        if phrases:
            total_written += self._append_chunk(phrases)
            phrases.clear()

        pbar.close()
        logger.info(f"Collected and saved {total_written:,} short sentences to {self.file_path}")

    def _append_chunk(self, phrases):
        """Append a chunk of phrases to the Parquet file."""
        df = pd.DataFrame({"phrase": phrases})
        mode = "append" if self.file_path.exists() else "overwrite"
        df.to_parquet(self.file_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Appended {len(df):,} phrases ({mode})")
        return len(df)

    def load(self):
        self._download_if_needed()
        df = pd.read_parquet(self.file_path)
        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42)
        logger.info(f"Loaded {len(df):,} CC100 phrases.")
        return df["phrase"].tolist()
