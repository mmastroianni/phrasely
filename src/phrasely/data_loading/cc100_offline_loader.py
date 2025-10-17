import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from phrasely.data_loading.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class CC100OfflineLoader(BaseLoader):
    """
    Loads existing CC100 Arrow shards from disk and filters them into phrases.

    - Reads pre-downloaded .arrow shards (no network).
    - Writes each chunk to its own Parquet part file to avoid overwriting.
    - Optionally merges all parts into one Parquet file at the end.
    """

    def __init__(
        self,
        arrow_dir: str,
        cache_dir: str = "data_cache",
        sample_size: int | None = None,
        max_phrases: int = 1_000_000,
        min_words: int = 3,
        max_words: int = 12,
        chunk_size: int = 100_000,
        merge_on_complete: bool = True,
    ):
        self.arrow_dir = Path(arrow_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.max_phrases = max_phrases
        self.min_words = min_words
        self.max_words = max_words
        self.chunk_size = chunk_size
        self.merge_on_complete = merge_on_complete

        self.parts_dir = self.cache_dir / "cc100_phrases_en_parts"
        self.parts_dir.mkdir(exist_ok=True)
        self.final_path = self.cache_dir / "cc100_phrases_en.parquet"

    def _download_if_needed(self):
        """Iterate through Arrow shards and build filtered phrase parts."""
        arrow_files = sorted(self.arrow_dir.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {self.arrow_dir}")

        start_count = len(list(self.parts_dir.glob("chunk_*.parquet")))
        logger.info(f"Resuming from {start_count} existing chunks in {self.parts_dir}")

        total_written = start_count * self.chunk_size
        pbar = tqdm(total=self.max_phrases, desc="Filtering CC100 shards", ncols=100)
        pbar.update(total_written)

        phrases = []

        for arrow_path in arrow_files:
            if total_written >= self.max_phrases:
                break
            logger.info(f"Processing shard: {arrow_path.name}")
            ds = Dataset.from_file(str(arrow_path))

            for record in ds:
                text = record["text"].strip().replace("\n", " ")
                for sent in text.split(". "):
                    words = sent.split()
                    if self.min_words <= len(words) <= self.max_words:
                        phrases.append(sent)
                        pbar.update(1)
                        if len(phrases) >= self.chunk_size:
                            total_written += self._write_chunk(phrases)
                            phrases.clear()
                        if total_written >= self.max_phrases:
                            break
                if total_written >= self.max_phrases:
                    break

        if phrases:
            total_written += self._write_chunk(phrases)
            phrases.clear()

        pbar.close()
        logger.info(f"Filtered and saved {total_written:,} phrases.")

        if self.merge_on_complete:
            self._merge_parts()

    def _write_chunk(self, phrases):
        """Write one chunk of phrases to a new part file."""
        df = pd.DataFrame({"phrase": phrases})
        chunk_id = len(list(self.parts_dir.glob("chunk_*.parquet")))
        out_path = self.parts_dir / f"chunk_{chunk_id:04d}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Wrote chunk {chunk_id:04d} with {len(df):,} phrases → {out_path.name}")
        return len(df)

    def _merge_parts(self):
        """Combine all part files into a single Parquet dataset."""
        logger.info(f"Merging parts from {self.parts_dir} ...")
        part_files = sorted(self.parts_dir.glob("chunk_*.parquet"))
        dfs = [pd.read_parquet(f) for f in part_files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_parquet(self.final_path, compression="snappy", index=False)
        logger.info(f"Merged {len(df_all):,} phrases into {self.final_path}")

    def load(self):
        self._download_if_needed()
        df = pd.read_parquet(self.final_path)
        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42)
        logger.info(f"Loaded {len(df):,} CC100 offline phrases.")

        if self.max_phrases and len(df) > self.max_phrases:
            df = df.sample(n=self.max_phrases, random_state=42)
        return df["phrase"].tolist()
