import logging
from typing import Optional, Iterator, List

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


class LAIONLoader:
    """
    Streaming loader for LAION caption datasets via HuggingFace Datasets.

    Features:
    ---------
    • Supports LAION-5B-en, LAION-400M, and metal/coco subsets.
    • Fully streaming, no local disk needed.
    • Yields mini-batches of captions as DataFrames.
    • Parameters mirror CC100S3Loader:
        - batch_size
        - max_phrases
        - max_shards (dataset slices)
    • Designed for Phrasely's pipeline structure.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset ID (e.g. "laion/laion5b-en")
    split : str
        Typically "train".
    caption_key : str
        Column name that contains text captions.
    batch_size : int
        Rows per yielded DataFrame.
    max_phrases : int, optional
        Global cap on rows.
    max_shards : int, optional
        HF streaming reads the whole dataset as a single shard,
        but this parameter lets you manually stop early.
    """

    def __init__(
        self,
        dataset_name: str = "laion/laion5b-en",
        split: str = "train",
        caption_key: str = "TEXT",
        batch_size: int = 20_000,
        max_phrases: Optional[int] = None,
        max_shards: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.caption_key = caption_key
        self.batch_size = batch_size
        self.max_phrases = max_phrases
        self.max_shards = max_shards

        logger.info(
            f"Initializing LAIONLoader(dataset='{dataset_name}', caption_key='{caption_key}', "
            f"batch_size={batch_size}, max_phrases={max_phrases})"
        )

        logger.info("Connecting to HuggingFace streaming dataset...")
        self.ds = load_dataset(dataset_name, split=split, streaming=True)
        logger.info("Connected to HuggingFace dataset.")

    # ------------------------------------------------------------
    def stream_load(self) -> Iterator[pd.DataFrame]:
        """
        Stream LAION captions in DataFrame batches.
        Matching the interface of CC100S3Loader.stream_load().
        """
        buffer: List[str] = []
        yielded_total = 0

        logger.info("Beginning LAION streaming iteration...")

        for i, row in enumerate(self.ds):
            if self.caption_key not in row:
                continue

            buffer.append(row[self.caption_key])

            # reached batch
            if len(buffer) >= self.batch_size:
                df = pd.DataFrame({"phrase": buffer})
                yielded_total += len(df)
                yield df
                buffer = []

                # global phrase cap
                if self.max_phrases and yielded_total >= self.max_phrases:
                    logger.info("Reached max_phrases limit.")
                    return

            # manual shard limit simulation
            if self.max_shards and i >= self.max_shards * self.batch_size:
                logger.info("Reached max_shards simulated limit.")
                break

        # final partial batch
        if buffer:
            df = pd.DataFrame({"phrase": buffer})
            yield df

        logger.info("Finished LAION streaming.")

    # ------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Convenience: load ALL rows (up to max_phrases) into one DataFrame.
        WARNING: not recommended for large datasets.
        """
        logger.info("Concatenating all streamed batches into a single DataFrame...")

        dfs = []
        total = 0

        for df in self.stream_load():
            dfs.append(df)
            total += len(df)
            if self.max_phrases and total >= self.max_phrases:
                break

        logger.info(f"Final DataFrame size: {total:,} rows.")
        return pd.concat(dfs, ignore_index=True)
