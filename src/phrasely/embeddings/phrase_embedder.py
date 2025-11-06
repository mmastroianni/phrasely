import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def _get_vram_gb() -> float:
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024**3)
        except Exception:
            return 0.0
    return 0.0


def _estimate_batch_size(max_length: int, vram_gb: float) -> int:
    """
    Very safe heuristic for sentence encoders (fp16). T4 (16 GB) → 64 by default.
    You can override via constructor.
    """
    if not torch.cuda.is_available():
        return 16
    if vram_gb >= 22:
        return 96
    if vram_gb >= 14:
        return 64
    if vram_gb >= 8:
        return 32
    return 16


@dataclass
class EmbedderConfig:
    model_name: str = "intfloat/e5-small-v2"  # ✅ default per your request
    max_length: int = 512  # typical context for e5 models
    batch_size: Optional[int] = None  # auto if None
    device: Optional[str] = None  # "cuda" or "cpu" or None→auto
    normalize: bool = True  # L2 normalize output
    use_torch_compile: bool = True  # try torch.compile for speed
    cache_dir: str = "data_cache"  # on-disk embedding cache
    cache_key: Optional[str] = None  # override dataset cache key
    dtype: torch.dtype = torch.float16  # fp16 on GPU, float32 on CPU


class PhraseEmbedder:
    """
    Fast GPU embedder built on HF Transformers (not sentence-transformers wrapper),
    with fp16 on GPU, torch.compile (when available), and simple on-disk caching.

    Usage:
        embedder = PhraseEmbedder(EmbedderConfig())
        X = embedder.embed(list_of_phrases, dataset_name="cc100_1M_en")
    """

    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.cfg = config or EmbedderConfig()

        # --- device / dtype ---
        if self.cfg.device is None:
            self.cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.cfg.device == "cpu":
            # fp16 on CPU is not supported → float32
            self.cfg.dtype = torch.float32

        # --- tokenizer & model ---
        logger.info(
            f"Loading encoder: {self.cfg.model_name} "
            f"(device={self.cfg.device}, dtype={self.cfg.dtype})"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=True
        )
        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        self.model.to(self.cfg.device)

        # fp16 if CUDA
        if self.cfg.device == "cuda":
            try:
                self.model = self.model.half()
                logger.info("Encoder converted to fp16.")
            except Exception as e:
                logger.warning(f"Could not set fp16: {e}")

        # torch.compile (PyTorch 2.x)
        if self.cfg.use_torch_compile:
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                logger.info("Enabled torch.compile for the encoder.")
            except Exception as e:
                logger.info(f"torch.compile unavailable/failed ({e}); continuing.")

        # --- batch size heuristic ---
        if self.cfg.batch_size is None:
            vram_gb = _get_vram_gb()
            self.cfg.batch_size = _estimate_batch_size(self.cfg.max_length, vram_gb)
            logger.info(
                f"Auto batch_size={self.cfg.batch_size} (VRAM≈{vram_gb:.1f} GB)"
            )

        # --- cache dir ---
        self.cache_dir = Path(self.cfg.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------- public API -------------------------

    def embed(
        self, phrases: List[str], dataset_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Returns (N, D) numpy float32 embeddings. Caches by (dataset_name, model).
        If dataset_name is None → no caching.
        """
        if not isinstance(phrases, list):
            phrases = list(phrases)

        cache_file = None
        if dataset_name is not None:
            safe_model = self.cfg.model_name.replace("/", "-")
            cache_key = self.cfg.cache_key or dataset_name
            cache_file = self.cache_dir / f"emb_{safe_model}_{cache_key}.npy"
            if cache_file.exists():
                logger.info(f"🔁 Loading cached embeddings from {cache_file}")
                arr = np.load(cache_file, mmap_mode="r")
                # ensure shape matches
                if arr.shape[0] == len(phrases):
                    return np.asarray(arr, dtype=np.float32)
                else:
                    logger.warning(
                        f"Cache length mismatch: cache has {arr.shape[0]}, "
                        f"phrases={len(phrases)}. Recomputing."
                    )

        X = self._encode_in_batches(phrases)

        if self.cfg.normalize:
            X = self._l2_normalize(X)

        if cache_file is not None:
            np.save(cache_file, X.astype(np.float32))
            logger.info(f"✅ Saved embeddings to {cache_file}")

        return X.astype(np.float32)

    # ------------------------- internals -------------------------

    @torch.inference_mode()
    def _encode_in_batches(self, texts: List[str]) -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        bs = int(self.cfg.batch_size or 32)
        max_len = int(self.cfg.max_length)

        # Some E5/GTE models expect "query: " / "passage: " prefixes for certain tasks.
        # We'll keep raw texts for now; easy to add a prefix switch later.
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            toks = self.tokenizer(
                chunk,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            toks = {k: v.to(self.cfg.device) for k, v in toks.items()}

            with (
                torch.autocast(device_type="cuda", dtype=self.cfg.dtype)
                if self.cfg.device == "cuda"
                else _nullcontext()
            ):
                out = self.model(**toks)
                # mean-pool over tokens using attention mask
                attn = toks["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                last = out.last_hidden_state  # (B, L, H)
                masked = last * attn
                summed = masked.sum(dim=1)  # (B, H)
                counts = attn.sum(dim=1).clamp(min=1)  # (B, 1)
                vecs = summed / counts
                all_vecs.append(vecs.detach().cpu().float().numpy())

            if (i // bs) % 50 == 0:
                logger.info(f"Embedding progress: {min(i+bs, len(texts))}/{len(texts)}")

        return np.concatenate(all_vecs, axis=0)

    @staticmethod
    def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return X / norms


# small helper to avoid importing contextlib at top-level if not needed
class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
