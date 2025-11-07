import logging
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


class MedoidSelector:
    """
    Robust medoid selector supporting both exact and approximate computation.

    Behavior:
    ---------
    • For clusters with <= exact_threshold points:
        Computes *exact* medoid via chunked pairwise distances.
        O(n²) work but chunked to avoid RAM spikes.

    • For clusters > exact_threshold:
        Uses *centroid-approximation medoid*
        (nearest point to centroid in cosine or euclidean space).

    GPU Support:
    ------------
    • If prefer_gpu=True and CuPy available, distance math uses GPU.
    • Otherwise CPU NumPy.

    Parameters
    ----------
    metric : {"cosine", "euclidean"}
        Distance metric.
    exact_threshold : int
        Max cluster size for exact medoid computation.
    chunk_size : int
        Partition size for chunked pairwise ops.
    prefer_gpu : bool
        If True and CuPy available, use GPU for pairwise ops.
    return_indices : bool
        If True → return (indices, phrases)
        Else → return phrases only.
    """

    def __init__(
        self,
        metric: str = "cosine",
        exact_threshold: int = 1500,
        chunk_size: int = 2048,
        prefer_gpu: bool = True,
        return_indices: bool = False,
    ):
        metric = metric.lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        self.metric = metric
        self.exact_threshold = int(exact_threshold)
        self.chunk_size = int(chunk_size)
        self.prefer_gpu = bool(prefer_gpu)
        self.return_indices = bool(return_indices)

    # ------------------------------------------------------------------
    def select(
        self,
        phrases: Sequence[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[List[str], List[int]] | List[str]:
        """
        Select medoids for each cluster (excluding noise).
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (N, D)")

        if embeddings.shape[0] != len(phrases) or labels.shape[0] != len(phrases):
            raise ValueError("Length mismatch in medoid selection inputs")

        unique_labels = sorted(int(x) for x in np.unique(labels) if x != -1)

        medoid_indices: List[int] = []
        medoid_phrases: List[str] = []

        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue

            cluster_emb = embeddings[idx]
            local_idx = self._cluster_medoid_index(cluster_emb)
            global_idx = int(idx[local_idx])

            medoid_indices.append(global_idx)
            medoid_phrases.append(phrases[global_idx])

        logger.info(
            "Selected %d medoids across %d clusters.",
            len(medoid_phrases),
            len(unique_labels),
        )

        if self.return_indices:
            return medoid_phrases, medoid_indices
        return medoid_phrases

    # ------------------------------------------------------------------
    def _cluster_medoid_index(self, X: np.ndarray) -> int:
        """
        Compute medoid index for a single cluster's embedding matrix (n, d).
        """
        n = X.shape[0]

        # GPU decision
        use_gpu = bool(self.prefer_gpu and _HAS_CUPY and n >= 512)
        xp = cp if use_gpu else np
        Xp = xp.asarray(X)

        # Cosine normalization
        if self.metric == "cosine":
            Xp = self._normalize_rows(Xp, xp)

        # --------------------------------------------------------------
        #   Small cluster → exact medoid via chunked pairwise distances
        # --------------------------------------------------------------
        if n <= self.exact_threshold:
            return self._exact_medoid(Xp, xp)

        # --------------------------------------------------------------
        #   Large cluster → approximate medoid (centroid strategy)
        # --------------------------------------------------------------
        centroid = xp.mean(Xp, axis=0, keepdims=True)

        if self.metric == "cosine":
            centroid = self._normalize_rows(centroid, xp)
            sims = (Xp @ centroid.T).ravel()
            idx = int(xp.argmax(sims))
        else:
            diffs = Xp - centroid
            d2 = xp.sum(diffs * diffs, axis=1)
            idx = int(xp.argmin(d2))

        return idx

    # ------------------------------------------------------------------
    def _exact_medoid(self, Xp, xp) -> int:
        """
        Compute exact medoid:
            argmin_i Σ_j distance(X[i], X[j])
        using chunked blocks to preserve memory.

        Supports cosine and euclidean.
        """
        n = Xp.shape[0]
        chunk = self.chunk_size

        best_i = -1
        best_val = float("inf")

        if self.metric == "cosine":
            # Pre-normalized → cosine distance = 1 - dot
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                dots = Xp[start:end] @ Xp.T  # shape (block, n)
                # cosine distance sum = Σ (1 - dot)
                row_sums = n - xp.sum(dots, axis=1)  # smaller = better

                local_idx = int(xp.argmin(row_sums))
                val = float(row_sums[local_idx])

                candidate = start + local_idx
                if val < best_val:
                    best_val = val
                    best_i = candidate

            return best_i

        else:
            # Euclidean: d(i,j)^2 = ||Xi||^2 + ||Xj||^2 - 2 <Xi, Xj>
            norms = xp.sum(Xp * Xp, axis=1)
            total_norms = xp.sum(norms)

            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = Xp[start:end]
                block_norms = xp.sum(block * block, axis=1)

                dots = block @ Xp.T
                # sum of squared distances for each row
                row_sums_sq = n * block_norms + total_norms - 2.0 * xp.sum(dots, axis=1)

                local_idx = int(xp.argmin(row_sums_sq))
                val = float(row_sums_sq[local_idx])

                candidate = start + local_idx
                if val < best_val:
                    best_val = val
                    best_i = candidate

            return best_i

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_rows(X, xp):
        norms = xp.linalg.norm(X, axis=1, keepdims=True)
        norms = xp.where(norms == 0, 1.0, norms)
        return X / norms
