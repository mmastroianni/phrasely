"""
Evaluation module for Phrasely.

Provides cluster quality metrics such as:
- DBCV (density-based validation)
- Cohesion (intra-cluster compactness)
- Silhouette score
- Unified evaluator interface
"""

from .cohesion import compute_cohesion
from .dbcv_score import compute_dbcv
from .evaluator import ClusterEvaluator
from .silhouette_score import compute_silhouette

__all__ = [
    "compute_dbcv",
    "compute_cohesion",
    "compute_silhouette",
    "ClusterEvaluator",
]
