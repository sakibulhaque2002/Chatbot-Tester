# tests/adapters/base_adapter.py
from abc import ABC, abstractmethod

class SimilarityAdapter(ABC):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    @abstractmethod
    def compute(self, actual: str, expected: str) -> float:
        """Compute similarity score between actual and expected."""
        pass

    def check_pass(self, actual: str, expected: str) -> bool:
        """Return True if score >= threshold."""
        return self.compute(actual, expected) >= self.threshold
