import logging
from typing import Optional, Iterator, List

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


class COYOLoader:
    """
    Stream captions from the COYO-700M dataset via HuggingFace streaming API.

    Parameters
    ----------
    batch_size : int, default=50_000
        Number of phrases per yielded DataFrame batch.

    max_phrases : int, optional
        Cap total number of phrases yielded.

    language_filter : str, optional
        If provided (e.g. "en"), keeps only captions where dataset['language']
        matches this value. COYO has mixed language but is mostly English.
    """

    def __init__(
        self,
        batch_size: int = 50_000,
        max_phrases: Optional[int] = None,
        language_filter: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.max_phrases = max_phrases
        self.language_filter = language_filter.lower() if language_filter else None

        logger.info(
            f"COYOLoader(batch_size={batch_size}, max_phrases={max_phrases}, "
            f"language_filter={self.language_filter})"
        )

        logger.info("Connecting to COYO-700M via HuggingFace streaming...")
        self.ds = load_dataset(
            "kakaobrain/coyo-700m",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        logger.info("Connected to COYO-700M.")

    # ------------------------------------------------------------------
    def stream_load(self) -> Iterator[pd.DataFrame]:
        """
        Stream captions in DataFrame batches.

        Yields
        ------
        pd.DataFrame with a single column "phrase".
        """
        batch: List[str] = []
        yielded_total = 0

        for example in self.ds:
            text = example.get("text", None)
            if not text:
                continue

            if self.language_filter:
                lang = example.get("language", "")
                if lang.lower() != self.language_filter:
                    continue

            batch.append(text)

            # Check if batch is ready
            if len(batch) >= self.batch_size:
                df = pd.DataFrame({"phrase": batch})
                yield df
                yielded_total += len(batch)
                batch = []

                logger.info(f"Yielded {yielded_total:,} phrases so far...")

                if self.max_phrases and yielded_total >= self.max_phrases:
                    logger.info("Reached max_phrases limit.")
                    return

        # final partial batch
        if batch:
            df = pd.DataFrame({"phrase": batch})
            yield df
