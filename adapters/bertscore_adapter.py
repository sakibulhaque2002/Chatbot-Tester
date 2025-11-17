# tests/adapters/bertscore_adapter.py
from bert_score import BERTScorer
from .base_adapter import SimilarityAdapter

class BERTScoreAdapter(SimilarityAdapter):
    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)
        # Load model once per session
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def compute(self, actual: str, expected: str) -> float:
        _, _, F1 = self.scorer.score([actual], [expected])
        return F1[0].item()
