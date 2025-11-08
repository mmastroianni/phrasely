"""
CoyoOfflineLoader

Loads locally cached COYO caption shards from Parquet files.
Compatible with the standard Phrasely DataLoader interface.

Expected directory structure:
    data_cache/coyo_1m/
        shard_0000.parquet
        shard_0001.parquet
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CoyoOfflineLoader:
    """
    Loads COYO captions from local Parquet shards.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing cached COYO shards.
    max_phrases : int, optional
        Global cap on total phrases returned.
    batch_size : int, optional
        If provided, yields mini-batches of this size.
        If None, yields whole shards.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data_cache/coyo_1m",
        max_phrases: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_phrases = max_phrases
        self.batch_size = batch_size

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.files = sorted(self.cache_dir.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet shards in {self.cache_dir}")

        logger.info(
            "CoyoOfflineLoader initialized: %d shards found in %s",
            len(self.files),
            self.cache_dir,
        )

    # -----------------------------------------------------------
    def stream_load(self) -> Iterator[pd.DataFrame]:
        """
        Stream COYO phrases as DataFrames.
        Applies `max_phrases` and optional `batch_size` splitting.
        """
        total = 0

        for fpath in self.files:
            df = pd.read_parquet(fpath)

            # Normalize schema → must have column 'phrase'
            if "phrase" not in df.columns:
                if "text" in df.columns:
                    df = df.rename(columns={"text": "phrase"})
                else:
                    raise ValueError(
                        f"Shard {fpath} missing expected 'phrase' or 'text' columns."
                    )

            # Drop NAs to stay consistent with CC100 loaders
            df = df.dropna(subset=["phrase"])

            # --- Handle global max_phrases ---
            if self.max_phrases is not None:
                remaining = self.max_phrases - total
                if remaining <= 0:
                    return
                if len(df) > remaining:
                    df = df.iloc[:remaining]

            # --- Batch splitting (optional) ---
            if self.batch_size is None:
                total += len(df)
                logger.info(
                    "Yielding full shard: %s (%d rows, total=%d)",
                    fpath.name,
                    len(df),
                    total,
                )
                yield df
            else:
                n = len(df)
                for start in range(0, n, self.batch_size):
                    sub = df.iloc[start : start + self.batch_size]

                    # Enforce max_phrases inside batching
                    if self.max_phrases is not None:
                        remaining = self.max_phrases - total
                        if remaining <= 0:
                            return
                        if len(sub) > remaining:
                            sub = sub.iloc[:remaining]

                    total += len(sub)
                    logger.info(
                        "Yielding batch %d rows from %s (total=%d)",
                        len(sub),
                        fpath.name,
                        total,
                    )
                    yield sub

                    if self.max_phrases is not None and total >= self.max_phrases:
                        return


    def load(self) -> list[str]:
        """
        Convenience: return all phrases as a single list.
        """
        phrases: list[str] = []
        for df in self.stream_load():
            phrases.extend(df["phrase"].tolist())
        return phrases
