from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PhraseEmbedder:
    """
    Generate embeddings for phrases using SentenceTransformer.

    • GPU or CPU inference
    • Optional fp16 mode on GPU
    • Built-in caching to disk
    • Batch inference
    """

    def __init__(
        self,
        model_name: str = "epam/sbert-e5-small-v2",  # ✅ default model (your choice)
        batch_size: int = 32,
        device: str | None = None,
        prefer_fp16: bool = True,
        cache_dir: str | Path = "data_cache",
    ):
        # ---------- device detection ----------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.prefer_fp16 = prefer_fp16
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # ---------- model load ----------
        logger.info(
            "PhraseEmbedder: model=%s device=%s batch=%d fp16=%s",
            model_name,
            device,
            batch_size,
            prefer_fp16,
        )

        self.model = SentenceTransformer(model_name, device=device)

        # ---------- fp16 conversion ----------
        if device == "cuda" and prefer_fp16:
            try:
                self.model = self.model.half()
                logger.info("Converted SentenceTransformer model to fp16.")
            except Exception as e:
                logger.warning("Could not cast model to fp16: %s", e)

    # ------------------------------------------------------------------

    def _cache_path(self, dataset_name: str) -> Path:
        safe_model = self.model_name.replace("/", "-")
        return self.cache_dir / f"embeddings_{dataset_name}_{safe_model}.npy"

    # ------------------------------------------------------------------

    def embed(self, phrases: List[str], dataset_name: str = "default") -> np.ndarray:
        """
        Main embedding entry point.

        Returns:
            np.ndarray of shape (N, D)
        """
        cache_file = self._cache_path(dataset_name)

        # ---------- load cache if available ----------
        if cache_file.exists():
            logger.info("🔁 Loading cached embeddings from %s", cache_file)
            emb = np.load(cache_file)
            return emb.astype(np.float32, copy=False)

        if not phrases:
            raise ValueError("embed() received empty phrase list")

        # ---------- compute embeddings ----------
        logger.info(
            "⚙️ Computing embeddings for %s phrases using model=%s",
            len(phrases),
            self.model_name,
        )

        # SentenceTransformer handles batching internally
        embeddings = self.model.encode(
            phrases,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=self.device,
        )

        # ---------- normalize output ----------
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # cuML & UMAP prefer float32
        embeddings = embeddings.astype(np.float32, copy=False)

        # ---------- save cache ----------
        np.save(cache_file, embeddings)
        logger.info("✅ Saved embeddings to %s", cache_file)

        return embeddings
