import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_clusters_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    texts=None,
    alpha: float = None,
    point_size: int = None,
    clip_outliers: bool = True,
    annotate: bool = True,
    savepath: str | None = None,
    dbcv_score: float | None = None,
    phrases: list[str] | None = None,
):
    """
    Plot 2D cluster visualization from reduced embeddings.

    Parameters
    ----------
    coords : np.ndarray
        2D array of shape (N, 2) with visualization coordinates.
    labels : np.ndarray
        Cluster labels (same length as coords).
    texts : list[str], optional
        Optional medoid phrases or representative labels.
    alpha : float, optional
        Point transparency; defaults scale by dataset size.
    point_size : int, optional
        Marker size; defaults scale by dataset size.
    clip_outliers : bool, default=True
        Clip extreme coordinates (>5 std) for cleaner visuals.
    annotate : bool, default=True
        If True, annotate medoid phrases.
    savepath : str, optional
        If provided, saves plot to file.
    dbcv_score : float, optional
        Optional DBCV score to show in title.
    phrases : list[str], optional
        Full phrase list (used to locate medoid coordinates).
    """
    if coords.shape[1] != 2:
        raise ValueError("coords must be 2D (N, 2) array")

    n_points = len(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # auto alpha and point size scaling
    if alpha is None:
        alpha = 0.6 if n_points < 5000 else 0.3 if n_points < 50000 else 0.15
    if point_size is None:
        point_size = 30 if n_points < 5000 else 10 if n_points < 50000 else 5

    # clip extreme outliers to improve readability
    if clip_outliers:
        mean, std = coords.mean(0), coords.std(0)
        mask = np.all(np.abs(coords - mean) < 5 * std, axis=1)
        coords, labels = coords[mask], labels[mask]
        logger.info(f"Clipped outliers: kept {mask.sum()} / {len(mask)} points")

    plt.figure(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for color, lbl in zip(colors, unique_labels):
        mask = labels == lbl
        label_name = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            label=label_name,
        )

    # optional DBCV overlay
    title = f"Cluster visualization ({n_clusters} clusters)"
    if dbcv_score is not None:
        title += f"  [DBCV={dbcv_score:.3f}]"
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(markerscale=1.5, fontsize=8, framealpha=0.6)

    # annotate medoids if provided
    if texts is not None and annotate:
        phrase_to_idx = {p: i for i, p in enumerate(phrases or [])}
        from matplotlib.cm import get_cmap

        cmap = get_cmap("tab10")

        medoid_limit = min(len(texts), 40)
        for i, text in enumerate(texts[:medoid_limit]):
            idx = phrase_to_idx.get(text)
            if idx is None or idx >= len(coords):
                continue
            color = cmap((i % 10) / 10)
            plt.text(
                coords[idx, 0],
                coords[idx, 1],
                text,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="lightgray", alpha=0.6, edgecolor="none", pad=1),
            )

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        logger.info(f"Saved cluster plot to {savepath}")
    else:
        plt.show()

    plt.close()
