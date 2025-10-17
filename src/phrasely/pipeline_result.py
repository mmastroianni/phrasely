from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    phrases: List[str]
    embeddings: np.ndarray
    reduced: np.ndarray
    labels: np.ndarray
    medoids: List[str]

    # -------------------- Summary --------------------
    def summary(self):
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return {
            "n_phrases": len(self.phrases),
            "n_clusters": n_clusters,
            "n_medoids": len(self.medoids),
            "embedding_dim": (
                self.embeddings.shape[1] if self.embeddings.size else 0
            ),
            "reduced_dim": (
                self.reduced.shape[1] if self.reduced.size else 0
            ),
        }

    # -------------------- Save --------------------
    def save(self, path: str | Path):
        """Save pipeline result to compressed .npz and metadata JSON."""
        path = Path(path)
        base = path.with_suffix("")  # strip any extension

        logger.info(f"Saving PipelineResult to {base}.npz and {base}_meta.json")

        np.savez_compressed(
            f"{base}.npz",
            phrases=np.array(self.phrases, dtype=object),
            embeddings=self.embeddings,
            reduced=self.reduced,
            labels=self.labels,
            medoids=np.array(self.medoids, dtype=object),
        )

        with open(f"{base}_meta.json", "w") as f:
            json.dump(self.summary(), f, indent=2)

    # -------------------- Load --------------------
    @staticmethod
    def load(path: str | Path) -> "PipelineResult":
        """Load a saved PipelineResult from disk."""
        path = Path(path)
        base = path.with_suffix("")  # strip extension
        logger.info(f"Loading PipelineResult from {base}.npz")

        data = np.load(f"{base}.npz", allow_pickle=True)
        phrases = data["phrases"].tolist()
        medoids = data["medoids"].tolist()
        embeddings = data["embeddings"]
        reduced = data["reduced"]
        labels = data["labels"]

        return PipelineResult(
            phrases=phrases,
            embeddings=embeddings,
            reduced=reduced,
            labels=labels,
            medoids=medoids,
        )
