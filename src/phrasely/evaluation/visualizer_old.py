import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phrasely.pipeline_result import PipelineResult

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """
    Lightweight plotting utilities for inspecting clustering results.

    Provides:
      • Cluster size histogram
      • Silhouette/DBCV bar chart
      • Optional medoid phrase preview
    """

    def __init__(self, figsize=(8, 5), style: str = "default"):
        plt.style.use(style)
        self.figsize = figsize

    # -------------------------------------------------------------

    def plot_cluster_sizes(self, result: PipelineResult, bins: int = 30):
        """Show histogram of cluster sizes."""
        labels = np.asarray(result.labels)
        unique, counts = np.unique(labels, return_counts=True)
        df = pd.DataFrame({"label": unique, "size": counts}).sort_values(
            "size", ascending=False
        )

        plt.figure(figsize=self.figsize)
        plt.hist(df["size"], bins=bins, color="steelblue", edgecolor="white")
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster Size (# phrases)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------

    def plot_metrics_bar(self, metrics: dict):
        """Visualize silhouette and DBCV scores side by side."""
        vals = []
        names = []
        for key in ["silhouette", "dbcv"]:
            val = metrics.get(key)
            if val is not None and not np.isnan(val):
                vals.append(val)
                names.append(key.upper())

        if not vals:
            logger.warning("No valid metrics to plot.")
            return

        plt.figure(figsize=(5, 4))
        plt.bar(names, vals, color=["#4CAF50", "#2196F3"])
        plt.ylim(0, 1)
        plt.title("Clustering Quality Metrics")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------

    def show_top_medoids(self, result: PipelineResult, n: int = 10):
        """Display the first few medoid phrases for quick inspection."""
        print(f"Top {min(n, len(result.medoids))} Medoids:")
        for i, phrase in enumerate(result.medoids[:n], start=1):
            print(f"{i:2d}. {phrase}")
