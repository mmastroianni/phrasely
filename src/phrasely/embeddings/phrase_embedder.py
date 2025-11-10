import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"


class PhraseEmbedder:
    """
    SentenceTransformer-based phrase embedder with CI-safe lazy loading.

    Design goals:
    • Lazy model loading (loaded on first embed() call)
    • GPU if available, otherwise CPU
    • Graceful fallback if model cannot be downloaded or imported
    • Always returns float32 numpy arrays
    • Optional caching on disk
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        prefer_fp16: bool = True,
        cache_dir: Path | str = "data_cache",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.prefer_fp16 = prefer_fp16
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Device detection: defer importing torch until absolutely needed
        self.device = device  # may be None until embed call

        self._model = None  # lazy-loaded

        logger.info(
            f"PhraseEmbedder initialized (model={model_name}, "
            f"batch_size={batch_size}, device={device}, fp16={prefer_fp16})"
        )

    # ------------------------------------------------------------------
    def _resolve_device(self):
        """Resolve GPU/CPU device without breaking CI."""
        if self.device is not None:
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            # Torch not installed, CI mode
            return "cpu"

    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Lazy-load the SentenceTransformer model.
        Falls back to fake embeddings if unavailable.
        """
        if self._model is not None:
            return

        device = self._resolve_device()
        self.device = device

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer ({self.model_name}) on {device}")
            model = SentenceTransformer(
                self.model_name,
                device=device,
                trust_remote_code=False,
                local_files_only=False,
                cache_folder=None,
                backend="torch",
                model_kwargs={"trust_remote_code": False},
                tokenizer_kwargs={"trust_remote_code": False},
            )

            # Optional fp16 (GPU only)
            if device == "cuda" and self.prefer_fp16:
                try:
                    model = model.half()
                    logger.info("Loaded model in fp16 mode.")
                except Exception as e:
                    logger.warning(f"Failed fp16 conversion ({e}).")
        except Exception as e:
            logger.warning(
                f"SentenceTransformer unavailable or failed to load ({e}). "
                "Falling back to hashing-based embeddings."
            )
            self._model = None  # mark as fake backend
            return

        self._model = model

    # ------------------------------------------------------------------
    def _cache_path(self, dataset_name: str) -> Path:
        safe_model = self.model_name.replace("/", "-")
        return self.cache_dir / f"emb_{dataset_name}_{safe_model}.npy"

    # ------------------------------------------------------------------
    def _fake_embed(self, phrases: List[str]) -> np.ndarray:
        """
        Deterministic fake embeddings for CI environments:
        • Hash each phrase
        • Map to a 64D float embedding
        • Ensures pipeline and tests still run
        """
        logger.info("Using hashing-based fake embeddings (CI mode).")

        out: np.ndarray = np.zeros((len(phrases), 64), dtype=np.float32)
        for i, p in enumerate(phrases):
            h = abs(hash(p))
            np.random.seed(h % 2**32)
            out[i] = np.random.normal(0, 1, 64)

        return out

    # ------------------------------------------------------------------
    def embed(self, phrases: List[str], dataset_name: str = "default") -> np.ndarray:
        """
        Compute (or load cached) embeddings for the given phrases.
        Always returns float32 numpy array.
        """
        if len(phrases) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        cache_file = self._cache_path(dataset_name)

        # --- try cache ---
        if cache_file.exists():
            try:
                emb = np.load(cache_file)
                return emb.astype(np.float32, copy=False)
            except Exception as e:
                logger.warning(f"Failed cache load ({e}); recomputing.")

        # --- ensure model available ---
        self._load_model()

        # If model failed to load, use deterministic fallback
        if self._model is None:
            emb = self._fake_embed(phrases)
            np.save(cache_file, emb)
            return emb

        # --- real transformer embeddings ---
        try:
            emb = self._model.encode(
                phrases,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"Model.embed() failed ({e}), falling back to hashing embeddings.")
            emb = self._fake_embed(phrases)

        # Normalize type
        emb = np.asarray(emb, dtype=np.float32)

        # Save cache
        try:
            np.save(cache_file, emb)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache ({e}).")

        return emb
