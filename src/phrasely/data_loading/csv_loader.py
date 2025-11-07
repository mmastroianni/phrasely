import logging
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Lightweight CSV loader that mirrors the behavior of CC100 loaders.

    Provides:
    - load()          full in-memory load
    - stream_load()   batch-by-batch generator

    Assumes:
    - first column contains text, or
    - column named 'phrase' exists

    Parameters
    ----------
    input_path : str or Path
        CSV file path.
    batch_size : int, default=20_000
        Rows per batch for streaming mode.
    max_phrases : int, optional
        Global phrase cap.
    """

    def __init__(
        self,
        input_path: str | Path,
        batch_size: int = 20_000,
        max_phrases: Optional[int] = None,
    ):
        self.input_path = Path(input_path)
        self.batch_size = batch_size
        self.max_phrases = max_phrases

    # ------------------------------------------------------------------
    def _read_csv(self) -> pd.DataFrame:
        """Read CSV and standardize to a DataFrame with a 'phrase' column."""
        try:
            df = pd.read_csv(self.input_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"[CSVLoader] File not found: {self.input_path}") from e

        # Identify phrase column
        if "phrase" in df.columns:
            phrases = df["phrase"]
        else:
            # fallback to first column
            first_col = df.columns[0]
            phrases = df[first_col].astype(str)

        out = pd.DataFrame({"phrase": phrases})
        out = out.dropna(subset=["phrase"]).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load all phrases into memory."""
        df = self._read_csv()
        total = len(df)
        logger.info(f"[CSVLoader] Loaded {total:,} rows from {self.input_path}")

        if self.max_phrases is not None and total > self.max_phrases:
            df = df.iloc[: self.max_phrases].reset_index(drop=True)
            logger.info(f"[CSVLoader] Truncated to {self.max_phrases:,} rows (max_phrases).")

        return df

    # ------------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """Stream CSV rows in batches."""
        df = self._read_csv()
        rows = len(df)
        yielded_total = 0

        for start in range(0, rows, self.batch_size):
            batch = df.iloc[start : start + self.batch_size]
            batch_len = len(batch)

            if self.max_phrases is not None:
                remaining = self.max_phrases - yielded_total
                if remaining <= 0:
                    logger.info("[CSVLoader] Reached max_phrases limit.")
                    return
                if batch_len > remaining:
                    batch = batch.iloc[:remaining]
                    logger.info("[CSVLoader] Truncated batch to max_phrases.")

            yielded_total += len(batch)
            yield batch

            if self.max_phrases is not None and yielded_total >= self.max_phrases:
                logger.info("[CSVLoader] Reached max_phrases limit.")
                return
