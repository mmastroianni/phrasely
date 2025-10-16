from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class PhraseEmbedder:
    """Generates embeddings for a list of short phrases."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, phrases: List[str]) -> np.ndarray:
        """Return embeddings as a NumPy array."""
        embeddings = self.model.encode(phrases, batch_size=32, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
