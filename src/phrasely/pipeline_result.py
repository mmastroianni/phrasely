from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Structured output of the full Phrasely clustering pipeline.

    Contains:
      • phrases – original text
      • reduced – reduced embeddings (SVD or UMAP)
      • labels – cluster labels
      • medoids – medoid phrases
      • medoid_indices – index positions of medoids
      • embeddings – optional original embeddings (usually None)
      • orig_dim – original embedding dimensionality
    """

    phrases: List[str]
    reduced: np.ndarray
    labels: np.ndarray
    medoids: List[str]
    medoid_indices: Optional[List[int]] = None
    embeddings: Optional[np.ndarray] = None
    orig_dim: Optional[int] = None

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a JSON-serializable summary of the pipeline output."""
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        emb_dim = (
            int(self.embeddings.shape[1])
            if isinstance(self.embeddings, np.ndarray)
            else int(self.orig_dim or 0)
        )
        red_dim = int(self.reduced.shape[1]) if isinstance(self.reduced, np.ndarray) else 0

        return {
            "n_phrases": len(self.phrases),
            "n_clusters": n_clusters,
            "n_medoids": len(self.medoids),
            "embedding_dim": emb_dim,
            "reduced_dim": red_dim,
        }

    # ----------------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        Save the PipelineResult to:
          • <path>.npz      – compressed array data
          • <path>_meta.json – summary & metadata
        """
        path = Path(path)
        base = path.with_suffix("")  # strip extension

        logger.info(f"Saving PipelineResult to {base}.npz and {base}_meta.json")

        np.savez_compressed(
            f"{base}.npz",
            phrases=np.array(self.phrases, dtype=object),
            reduced=self.reduced,
            labels=self.labels,
            medoids=np.array(self.medoids, dtype=object),
            medoid_indices=np.array(self.medoid_indices or [], dtype=int),
        )

        meta = self.summary()
        meta["orig_dim"] = self.orig_dim

        with open(f"{base}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ----------------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------------
    @staticmethod
    def load(path: str | Path) -> "PipelineResult":
        """
        Load a PipelineResult previously saved via .save().
        Automatically loads:
          • <path>.npz
          • <path>_meta.json (if available)
        """
        path = Path(path)
        base = path.with_suffix("")

        logger.info(f"Loading PipelineResult from {base}.npz")

        with np.load(f"{base}.npz", allow_pickle=True) as data:
            phrases = data["phrases"].tolist()
            reduced = data["reduced"]
            labels = data["labels"]
            medoids = data["medoids"].tolist()
            medoid_indices = (
                data["medoid_indices"].tolist()
                if "median_indices" in data or "medoid_indices" in data
                else None
            )

        # Load metadata JSON if present
        orig_dim = None
        meta_path = f"{base}_meta.json"
        if Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                orig_dim = meta.get("orig_dim")

        return PipelineResult(
            phrases=phrases,
            reduced=reduced,
            labels=labels,
            medoids=medoids,
            medoid_indices=medoid_indices,
            embeddings=None,
            orig_dim=orig_dim,
        )

    # ----------------------------------------------------------------------
    # Pretty repr
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PipelineResult(phrases={s['n_phrases']:,}, "
            f"clusters={s['n_clusters']:,}, "
            f"medoids={s['n_medoids']:,}, "
            f"embedding_dim={s['embedding_dim']}, "
            f"reduced_dim={s['reduced_dim']})"
        )
