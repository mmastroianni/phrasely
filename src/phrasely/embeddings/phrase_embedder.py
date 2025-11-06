import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PhraseEmbedder:
    """
    Generates embeddings for input phrases using a SentenceTransformer model.

    Supports:
        • GPU or CPU inference depending on availability.
        • Optional fp16 on GPU to reduce VRAM usage.
        • Optional caching for offline/batch mode.
        • Safe, no-cache streaming for S3 loaders.

    Usage:
        embedder = PhraseEmbedder()
        X = embedder.embed(phrases, dataset_name="myset")

        # stream-safe:
        X = embedder.embed(phrases, dataset_name=None)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 8,
        device: Optional[str] = None,
        fp16: bool = True,
    ):
        # Auto-detect GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16

        logger.info(
            f"PhraseEmbedder using model={model_name}, device={device}, "
            f"batch_size={batch_size}"
        )

        # Load model
        self.model = SentenceTransformer(model_name, device=device)

        # FP16 conversion (GPU only)
        if device == "cuda" and fp16:
            try:
                self.model = self.model.half()
                logger.info("Model converted to fp16 (GPU).")
            except Exception as e:
                logger.warning(f"Could not convert model to fp16: {e}")

    # --------------------------------------------------------------
    def _compute_embeddings(self, phrases: List[str]) -> np.ndarray:
        """Internal helper: compute embeddings with correct dtype handling."""
        out = self.model.encode(
            phrases,
            batch_size=self.batch_size,
            convert_to_numpy=False,  # we normalize manually
            show_progress_bar=True,
            device=self.device,
        )

        # Convert to numpy
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        elif isinstance(out, list):
            if len(out) and isinstance(out[0], torch.Tensor):
                out = torch.stack(out).detach().cpu().numpy()
            else:
                out = np.array(out)

        return np.asarray(out, dtype=np.float32)

    # --------------------------------------------------------------
    def embed(
        self,
        phrases: List[str],
        dataset_name: Optional[str] = "default",
        no_cache: bool = False,
    ) -> np.ndarray:
        """
        Compute embeddings, optionally using a persistent cache.

        Parameters
        ----------
        phrases : list[str]
        dataset_name : str or None
            If None → no caching (streaming mode).
        no_cache : bool
            If True → force fresh compute even if cache exists.
        """

        if dataset_name is None or no_cache:
            logger.info("Embedding batch without cache (streaming mode).")
            return self._compute_embeddings(phrases)

        # ---------------------------------------------------------
        # Cached offline mode
        # ---------------------------------------------------------
        cache_dir = Path("data_cache")
        cache_dir.mkdir(exist_ok=True)

        safe_model = self.model_name.replace("/", "-")
        cache_file = cache_dir / f"embeddings_{dataset_name}_{safe_model}.npy"

        # Try loading cached embeddings
        if not no_cache and cache_file.exists():
            logger.info(f"🔁 Loading cached embeddings: {cache_file}")
            return np.load(cache_file)

        # Compute new embeddings
        logger.info(f"⚙️ Computing embeddings for '{dataset_name}'...")
        emb = self._compute_embeddings(phrases)

        # Save cache
        np.save(cache_file, emb)
        logger.info(f"✅ Saved embeddings → {cache_file}")

        return emb
