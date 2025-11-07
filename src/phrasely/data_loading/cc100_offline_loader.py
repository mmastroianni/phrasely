import logging
from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CC100OfflineLoader:
    """
    Loads CC100 Arrow or Parquet shards from local disk, mirroring the behavior
    of CC100S3Loader for API consistency.

    Supports both:
    - load(): full in-memory load
    - stream_load(): batch-by-batch generator

    Parameters
    ----------
    arrow_dir : str or Path
        Directory containing local Arrow/Parquet shards.
    language : str, default="en"
        Case-insensitive substring filter on filenames.
    max_phrases : int, optional
        Global cap on number of phrases loaded.
    chunk_size : int, default=100_000
        Logging unit when loading large files in load().
    seed : int, default=42
        Sampling seed for max_phrases in load().
    max_files : int, optional
        Cap the number of shards considered.
    batch_size : int, default=20_000
        Batch size for streaming mode.
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
        self.language = language.lower() if language else ""
        self.max_phrases = max_phrases
        self.chunk_size = chunk_size
        self.seed = seed
        self.max_files = max_files
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    def _collect_files(self) -> List[Path]:
        """Return deterministic, filtered list of Arrow/Parquet shards."""
        arrow_files = sorted(
            f
            for f in self.arrow_dir.glob("*.arrow")
            if not self.language or self.language in f.name.lower()
        )
        parquet_files = sorted(
            f
            for f in self.arrow_dir.glob("*.parquet")
            if not self.language or self.language in f.name.lower()
        )

        files = parquet_files + arrow_files

        if not files:
            raise FileNotFoundError(
                f"No Arrow/Parquet files found in {self.arrow_dir} "
                f"for language='{self.language}'."
            )

        if self.max_files is not None and len(files) > self.max_files:
            logger.warning(f"Found {len(files)} files; limiting to first {self.max_files}.")
            files = files[: self.max_files]

        return files

    # ------------------------------------------------------------------
    def _read_table(self, path: Path) -> pa.Table:
        """Read a single Arrow/Parquet shard robustly."""
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            return pq.read_table(path)

        if suffix == ".arrow":
            try:
                with pa.ipc.open_file(path) as reader:
                    return reader.read_all()
            except pa.lib.ArrowInvalid:
                with pa.ipc.open_stream(path) as reader:
                    return reader.read_all()

        raise ValueError(f"Unsupported file type: {path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _table_to_df(table: pa.Table) -> pd.DataFrame:
        """Standardize an Arrow table to a DataFrame with a 'phrase' column."""
        df = table.to_pandas()

        # normalize naming
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})

        if "phrase" not in df.columns:
            raise ValueError("Shard missing required 'phrase' or 'text' column.")

        return df.dropna(subset=["phrase"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Load all shards into one DataFrame. Good for smaller local datasets.
        """
        files = self._collect_files()
        logger.info(f"Loading {len(files)} shards from {self.arrow_dir}")

        parts: List[pd.DataFrame] = []

        for path in files:
            table = self._read_table(path)
            df = self._table_to_df(table)

            parts.append(df)

            if len(df) >= self.chunk_size:
                logger.info(f"{path.name}: loaded {len(df):,} rows")

        merged = pd.concat(parts, ignore_index=True)
        logger.info(f"Loaded a total of {len(merged):,} phrases.")

        # Sample down if needed
        if self.max_phrases is not None and len(merged) > self.max_phrases:
            merged = merged.sample(self.max_phrases, random_state=self.seed)
            logger.info(f"Sampled down to {len(merged):,} phrases (max_phrases).")

        return merged.reset_index(drop=True)

    # ------------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Yield DataFrames in batches, respecting batch_size and max_phrases.
        Mirrors S3 loader's behavior.
        """
        files = self._collect_files()
        logger.info(f"Streaming {len(files)} shards from {self.arrow_dir}")

        yielded_total = 0

        for idx, path in enumerate(files, start=1):
            logger.info(f"[{idx}/{len(files)}] Reading {path.name}")

            table = self._read_table(path)
            df = self._table_to_df(table)

            rows = len(df)
            logger.info(f"{path.name}: {rows:,} rows")

            for start in range(0, rows, self.batch_size):
                batch = df.iloc[start : start + self.batch_size]
                batch_len = len(batch)

                # max_phrases handling
                if self.max_phrases is not None:
                    remaining = self.max_phrases - yielded_total
                    if remaining <= 0:
                        logger.info("Reached max_phrases limit.")
                        return
                    if batch_len > remaining:
                        batch = batch.iloc[:remaining]
                        logger.info("Truncated batch to respect max_phrases.")

                yielded_total += len(batch)
                yield batch

                if self.max_phrases is not None and yielded_total >= self.max_phrases:
                    logger.info("Reached max_phrases limit.")
                    return
