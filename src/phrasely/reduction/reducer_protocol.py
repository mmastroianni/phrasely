from typing import Protocol
import numpy as np


class ReducerProtocol(Protocol):
    """
    A simple protocol that unifies SVDReducer, TwoStageReducer,
    and any future reducers.

    A reducer must:
        • expose n_components (int)
        • implement reduce(X: np.ndarray) -> np.ndarray

    This allows pipeline.py to type-check correctly, even
    when switching between different reducer implementations.
    """

    n_components: int

    def reduce(self, X: np.ndarray) -> np.ndarray:
        ...
