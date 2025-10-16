import logging

import numpy as np

logger = logging.getLogger(__name__)


class MedoidSelector:
    """Selects representative medoids for each cluster."""

    def select(self, phrases, embeddings, labels):
        unique_labels = np.unique(labels)
        medoids = []

        for lbl in unique_labels:
            if lbl == -1:
                continue  # skip noise cluster
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue
            # pick median index or a central point in cluster
            medoid_idx = idx[len(idx) // 2]
            medoids.append(phrases[medoid_idx])

        logger.info(
            f"Selected {len(medoids)} medoids across {len(unique_labels)}"
            + " clusters."
        )
        return medoids
