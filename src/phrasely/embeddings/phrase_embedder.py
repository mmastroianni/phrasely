import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PhraseEmbedder:
    """
    GPU/CPU-aware phrase embedder with:
      • automatic model/batch selection,
      • on-disk caching,
      • chunked streaming for large datasets.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        cache_dir: str = "data_cache",
        chunk_size: int = 10_000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        # --- Detect VRAM ---
        self.vram_gb = 0
        if self.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                self.vram_gb = props.total_memory / 1024**3
            except Exception:
                pass

        # --- Model selection ---
        if model_name:
            self.model_name = model_name
        elif self.vram_gb >= 6:
            self.model_name = "all-mpnet-base-v2"
        else:
            self.model_name = "paraphrase-MiniLM-L6-v2"

        # --- Batch size heuristic ---
        if batch_size:
            self.batch_size = batch_size
        elif self.vram_gb >= 10:
            self.batch_size = 64
        elif self.vram_gb >= 6:
            self.batch_size = 32
        else:
            self.batch_size = 8

        logger.info(
            f"PhraseEmbedder using model={self.model_name}, device={self.device}, "
            f"VRAM≈{self.vram_gb:.1f} GB, batch_size={self.batch_size}"
        )

        # --- Load model ---
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if self.device == "cuda" and self.vram_gb < 6:
            try:
                self.model = self.model.half()
                logger.info("Converted model to fp16 for reduced VRAM usage.")
            except Exception:
                logger.warning("Could not convert model to fp16; continuing in fp32.")

    # ---------------------------------------------------------------------

    def embed(self, phrases: list[str]) -> np.ndarray:
        """Return embeddings with chunked caching to disk."""
        cache_path = self._cache_path(phrases)
        tmp_path = cache_path.with_suffix(".partial.npy")

        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path, allow_pickle=False)

        logger.info("Computing embeddings in streaming mode...")
        total = len(phrases)
        f_out = open(tmp_path, "ab")

        for start in tqdm(
            range(0, total, self.chunk_size), desc="Embedding chunks", ncols=90
        ):
            end = min(start + self.chunk_size, total)
            chunk_phrases = phrases[start:end]
            try:
                emb = self.model.encode(
                    chunk_phrases,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning("CUDA OOM → retrying this chunk on CPU")
                    cpu_model = SentenceTransformer(self.model_name, device="cpu")
                    emb = cpu_model.encode(
                        chunk_phrases, batch_size=32, show_progress_bar=False
                    )
                else:
                    raise
            np.save(f_out, np.asarray(emb, dtype=np.float32))

        f_out.close()

        # Merge all chunk arrays into a single file
        logger.info("Finalizing embeddings cache...")
        all_embs = self._load_all_chunks(tmp_path)
        np.save(cache_path, all_embs)
        Path(tmp_path).unlink(missing_ok=True)
        logger.info(f"Saved full embeddings cache to {cache_path}")
        return all_embs

    # ---------------------------------------------------------------------

    def _load_all_chunks(self, tmp_path: Path) -> np.ndarray:
        """Load sequential .npy chunks from append file."""
        data = []
        with open(tmp_path, "rb") as f:
            while True:
                try:
                    data.append(np.load(f))
                except ValueError:
                    break
                except EOFError:
                    break
        return np.concatenate(data, axis=0)

    def _cache_path(self, phrases: list[str]) -> Path:
        sample = "\n".join(phrases[:1000])
        data_hash = hashlib.md5(sample.encode("utf-8")).hexdigest()
        name = f"embeddings_{self.model_name.replace('/', '-')}_{data_hash}.npy"
        return self.cache_dir / name
