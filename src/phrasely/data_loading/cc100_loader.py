import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


class CC100Loader:
    """
    Loads a manageable subset of the CC100 dataset from Hugging Face.

    Parameters
    ----------
    language : str, default="en"
        Language code, e.g. 'en', 'fr', 'de'. Defaults to English.
    max_phrases : int, optional
        Maximum number of phrases to load (samples if dataset is larger).
    cache_dir : str, optional
        Directory for Hugging Face cache. If None, uses default ~/.cache/huggingface.
    seed : int, default=42
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        language: str = "en",
        max_phrases: Optional[int] = 100_000,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.language = language
        self.max_phrases = max_phrases
        self.cache_dir = cache_dir
        self.seed = seed

    # ----------------------------------------------------------

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading CC100 ({self.language}) subset from Hugging Face...")

        dataset = load_dataset(
            "cc100",
            self.language,
            split="train",
            cache_dir=self.cache_dir,
        )

        # Convert to DataFrame
        df = pd.DataFrame(dataset)[["text"]].rename(columns={"text": "phrase"})
        logger.info(f"Loaded {len(df):,} rows for language='{self.language}'")

        # Optional down-sampling
        if self.max_phrases is not None and len(df) > self.max_phrases:
            df = df.sample(n=self.max_phrases, random_state=self.seed)
            logger.info(f"Sampled {len(df):,} rows (max_phrases={self.max_phrases})")

        # Clean up and return
        df = df.dropna(subset=["phrase"])
        df = df[df["phrase"].str.len() > 0].reset_index(drop=True)

        logger.info(f"Returning {len(df):,} cleaned phrases.")
        return df
