import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PhraseEmbedder:
    """
    Generates and caches embeddings for input phrases using a SentenceTransformer model.

    Features:
        • GPU or CPU inference depending on availability.
        • Automatic fp16 conversion for lower VRAM usage.
        • Transparent caching via dataset_name.
        • Batch processing with progress bar.

    Example:
        embedder = PhraseEmbedder()
        embeddings = embedder.embed(phrases, dataset_name="msmarco")
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 8,
        device: str | None = None,
    ):
        # Auto-detect GPU if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        logger.info(
            f"PhraseEmbedder using model={model_name}, device={device}, "
            f"batch_size={batch_size}"
        )

        # Load model
        self.model = SentenceTransformer(model_name, device=device)

        # Convert to fp16 if on GPU to save VRAM
        if device == "cuda":
            try:
                self.model = self.model.half()
                logger.info("Converted model to fp16 for reduced VRAM usage.")
            except Exception as e:
                logger.warning(f"Could not convert model to fp16: {e}")

    # ------------------------------------------------------------------
    def embed(self, phrases: list[str], dataset_name: str = "default") -> np.ndarray:
        """
        Generate or load cached embeddings for a given dataset.

        Args:
            phrases: list of input phrases.
            dataset_name: unique identifier for the dataset (used for caching).

        Returns:
            np.ndarray of shape (n_phrases, embedding_dim)
        """
        cache_dir = Path("data_cache")
        cache_dir.mkdir(exist_ok=True)

        safe_model = self.model_name.replace("/", "-")
        cache_file = cache_dir / f"embeddings_{dataset_name}_{safe_model}.npy"

        # Try loading cached embeddings
        if cache_file.exists():
            logger.info(
                f"🔁 Loading cached embeddings for '{dataset_name}' from {cache_file}"
            )
            return np.load(cache_file)

        # Compute new embeddings
        logger.info(
            f"⚙️ Computing embeddings for '{dataset_name}' using {self.model_name}"
        )
        embeddings = self.model.encode(
            phrases,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=self.device,
        )

        # --- Normalize type for mypy and downstream code ---
        if isinstance(embeddings, list):
            # Handle list of tensors or arrays
            if len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor):
                embeddings = torch.stack(embeddings)
            embeddings = np.array(embeddings)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Save cache
        np.save(cache_file, embeddings)
        logger.info(f"✅ Saved embeddings to {cache_file}")
        return embeddings
