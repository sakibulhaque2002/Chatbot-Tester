# tests/adapters/cosine_adapter.py
import numpy as np
from numpy.linalg import norm
from utils.embed import get_embeddings
from .base_adapter import SimilarityAdapter

class CosineAdapter(SimilarityAdapter):
    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def compute(self, actual: str, expected: str) -> float:
        actual_emb, expected_emb = get_embeddings([actual, expected])
        sim = float(np.dot(actual_emb, expected_emb) / (norm(actual_emb) * norm(expected_emb)))
        return sim
