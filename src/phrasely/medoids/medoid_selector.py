import logging
from typing import List, Sequence
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False


class MedoidSelector:
    """
    Selects representative medoids for each cluster.

    For clusters with size <= exact_threshold, computes the *true medoid*:
      argmin_i sum_j distance(x_i, x_j).

    For larger clusters, approximates by choosing the point nearest
    to the cluster centroid
    (cosine: nearest to normalized mean vector; euclidean: nearest to mean).

    Parameters
    ----------
    metric : {"cosine", "euclidean"}
        Distance metric for medoid selection. Default "cosine".
    exact_threshold : int
        Max cluster size to compute exact medoid (O(n^2) time, chunked memory).
        Default 1500.
    chunk_size : int
        Row-block size for chunked pairwise computations. Default 2048.
    prefer_gpu : bool
        If True and CuPy is available, use GPU arrays/ops for heavy math.
    return_indices : bool
        If True, also return medoid indices into the original `phrases`/`embeddings`.
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

    # ----------------------- public API -----------------------

    def select(
        self,
        phrases: Sequence[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ):
        """Select representative medoids for each cluster."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array (N, D)")
        if embeddings.shape[0] != len(phrases) or labels.shape[0] != len(phrases):
            raise ValueError("length mismatch among phrases/embeddings/labels")

        unique = sorted(int(x) for x in np.unique(labels) if x != -1)
        medoid_phrases: List[str] = []
        medoid_indices: List[int] = []

        for lbl in unique:
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue
            cluster_emb = embeddings[idx]
            medoid_local = self._cluster_medoid_index(cluster_emb)
            medoid_global = int(idx[medoid_local])
            medoid_indices.append(medoid_global)
            medoid_phrases.append(phrases[medoid_global])

        logger.info(
            f"Selected {len(medoid_phrases)} medoids across {len(unique)} clusters."
        )

        # ✅ Return only phrases unless explicitly asked for indices
        if self.return_indices:
            return medoid_phrases, medoid_indices
        else:
            return medoid_phrases

    # ----------------------- internals -----------------------

    def _cluster_medoid_index(self, X: np.ndarray) -> int:
        n = X.shape[0]
        use_gpu = _CUPY_AVAILABLE and self.prefer_gpu and (n >= 512)

        xp = cp if (use_gpu and _CUPY_AVAILABLE) else np
        Xp = xp.asarray(X)

        if self.metric == "cosine":
            Xp = self._normalize_rows(Xp, xp)

        if n <= self.exact_threshold:
            if self.metric == "cosine":
                best_i, best_val = -1, xp.inf
                for start in range(0, n, self.chunk_size):
                    stop = min(start + self.chunk_size, n)
                    dots = Xp[start:stop] @ Xp.T
                    row_sums = n - xp.sum(dots, axis=1)
                    mins_idx = xp.argmin(row_sums)
                    mins_val = row_sums[mins_idx]
                    candidate_i = int(start + int(mins_idx))
                    if mins_val < best_val:
                        best_val = float(mins_val)
                        best_i = candidate_i
                return int(best_i)
            else:
                norms = xp.sum(Xp * Xp, axis=1)
                total_norm_sum = xp.sum(norms)
                best_i, best_val = -1, xp.inf
                for start in range(0, n, self.chunk_size):
                    stop = min(start + self.chunk_size, n)
                    block = Xp[start:stop]
                    block_norms = xp.sum(block * block, axis=1)
                    dots = block @ Xp.T
                    row_sums_sq = (
                        n * block_norms + total_norm_sum - 2.0 * xp.sum(dots, axis=1)
                    )
                    mins_idx = xp.argmin(row_sums_sq)
                    mins_val = row_sums_sq[mins_idx]
                    candidate_i = int(start + int(mins_idx))
                    if mins_val < best_val:
                        best_val = float(mins_val)
                        best_i = candidate_i
                return int(best_i)

        # Approximate for large clusters: nearest to centroid
        centroid = xp.mean(Xp, axis=0, keepdims=True)
        if self.metric == "cosine":
            centroid = self._normalize_rows(centroid, xp)
            sims = (Xp @ centroid.T).ravel()
            idx = int(xp.argmax(sims))
        else:
            diffs = Xp - centroid
            d2 = xp.sum(diffs * diffs, axis=1)
            idx = int(xp.argmin(d2))

        return int(idx)

    @staticmethod
    def _normalize_rows(X, xp_module):
        norms = xp_module.linalg.norm(X, axis=1, keepdims=True)
        norms = xp_module.where(norms == 0, 1.0, norms)
        return X / norms
