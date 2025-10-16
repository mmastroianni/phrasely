from abc import ABC, abstractmethod
from typing import List


class BaseLoader(ABC):
    """Abstract base class for phrase data loaders."""

    @abstractmethod
    def load(self) -> List[str]:
        """Load a list of phrases from a source."""
        pass
