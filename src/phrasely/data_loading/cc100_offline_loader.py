import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CC100OfflineLoader:
    """
    Loads a pre-downloaded subset of the CC100 dataset from local
    Arrow or Parquet files.

    Supports both in-memory (`load()`) and streaming (`stream_load()`) modes.

    Parameters
    ----------
    arrow_dir : str or Path
        Directory containing local Arrow or Parquet shards (e.g., "data_cache/cc100").
    language : str, default="en"
        Optional language filter. If present in filenames (e.g. "cc100_en_*.parquet"),
        only matching files will be loaded. Use "" to load all languages.
    max_phrases : int, optional
        Maximum number of phrases to load (samples if larger).
    chunk_size : int, default=100_000
        Used for logging and partial loads when processing large directories.
    seed : int, default=42
        Random seed for reproducible sampling.
    max_files : int, default=5
        Maximum number of Arrow/Parquet shards to load. Helps avoid
        out-of-memory errors when large directories are present.
    batch_size : int, default=20_000
        Maximum number of rows per batch when streaming (stream_load mode).
    """

    def __init__(
        self,
        arrow_dir: str | Path,
        language: str = "en",
        max_phrases: Optional[int] = None,
        chunk_size: int = 100_000,
        seed: int = 42,
        max_files: Optional[int] = 5,
        batch_size: int = 20_000,
    ):
        self.arrow_dir = Path(arrow_dir)
        self.language = language
        self.max_phrases = max_phrases
        self.chunk_size = chunk_size
        self.seed = seed
        self.max_files = max_files
        self.batch_size = batch_size

    # ----------------------------------------------------------
    def _read_table(self, path: Path) -> pa.Table:
        """Read a single Arrow or Parquet file robustly."""
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pq.read_table(path)
        elif suffix == ".arrow":
            try:
                with pa.ipc.open_file(path) as reader:
                    return reader.read_all()
            except pa.lib.ArrowInvalid:
                # Fall back to stream format (used by Hugging Face datasets)
                with pa.ipc.open_stream(path) as reader:
                    return reader.read_all()
        else:
            raise ValueError(f"Unsupported file type: {path}")

    # ----------------------------------------------------------
    def _collect_files(self) -> list[Path]:
        """Return filtered and capped list of shards."""
        files = sorted(
            f
            for f in self.arrow_dir.glob("*.parquet")
            if (not self.language or self.language.lower() in f.name.lower())
        )
        files += sorted(
            f
            for f in self.arrow_dir.glob("*.arrow")
            if (not self.language or self.language.lower() in f.name.lower())
        )

        if not files:
            raise FileNotFoundError(
                f"No parquet/arrow files found in {self.arrow_dir}"
                + f"for language='{self.language}'"
            )

        if self.max_files is not None and len(files) > self.max_files:
            logger.warning(
                f"⚠️  Found {len(files)} shards; limiting to first {self.max_files} "
                "to avoid memory overflow."
            )
            files = files[: self.max_files]
        return files

    # ----------------------------------------------------------
    def _table_to_df(self, table: pa.Table) -> pd.DataFrame:
        """Convert Arrow table to standardized DataFrame with 'phrase' column."""
        df = table.to_pandas()
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})
        df = df.dropna(subset=["phrase"])
        return df

    # ----------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load and merge all shards into one DataFrame (in-memory)."""
        files = self._collect_files()
        logger.info(f"Resuming from {len(files)} existing chunks in {self.arrow_dir}")

        parts = []
        for path in files:
            table = self._read_table(path)
            df = self._table_to_df(table)
            parts.append(df)

            if len(df) > 0 and len(df) % self.chunk_size == 0:
                logger.info(f"Loaded {len(df):,} phrases from {path.name}")

        merged = pd.concat(parts, ignore_index=True)
        logger.info(f"Merged {len(merged):,} phrases from {len(parts)} parts.")

        if self.max_phrases is not None and len(merged) > self.max_phrases:
            merged = merged.sample(n=self.max_phrases, random_state=self.seed)
            logger.info(
                f"Sampled down to {len(merged):,} phrases "
                + f"(max_phrases={self.max_phrases})"
            )

        merged = merged.reset_index(drop=True)
        logger.info(f"Loaded {len(merged):,} CC100 offline phrases.")
        return merged

    # ----------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Stream shards in mini-batches to avoid large memory usage.
        Yields smaller DataFrames sequentially.
        """
        files = self._collect_files()
        logger.info(f"Streaming {len(files)} chunks from {self.arrow_dir}")

        for i, path in enumerate(files, 1):
            table = self._read_table(path)
            df = self._table_to_df(table)

            # Yield in manageable sub-batches
            for start in range(0, len(df), self.batch_size):
                sub_df = df.iloc[start : start + self.batch_size]
                logger.info(
                    f"Yielding {len(sub_df):,} rows from {path.name} "
                    f"({start // self.batch_size + 1}/"
                    f"{int(np.ceil(len(df) / self.batch_size))})"
                )
                yield sub_df
