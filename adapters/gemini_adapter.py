# tests/adapters/gemini_adapter.py
from .base_adapter import SimilarityAdapter
import os
import requests
import re
from config import OPENROUTER_API_KEY


class GeminiAdapter(SimilarityAdapter):
    def __init__(self, api_key: str = OPENROUTER_API_KEY, threshold: float = 0.7, model: str = "google/gemini-2.5-flash"):
        super().__init__(threshold)
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def compute(self, actual: str, expected: str) -> float:
        prompt = f"""
        Compare the following two sentences and return a numeric score between 0 and 1 
        indicating how semantically similar they are. Use the scale below:

        Score	Description
        1.0 Perfect match; identical meaning.
        0.9 Nearly identical; almost exactly the same meaning.
        0.8 Very high similarity; only small differences.
        0.7 High similarity; most of the meaning is the same with some differences.
        0.6 Fair similarity; many shared ideas, but notable differences.
        0.5 Moderate similarity; about half of the meaning overlaps.
        0.4 Low similarity; some shared concepts, but mostly different.
        0.3 Weak similarity; some minor overlap in meaning.
        0.2 Very weak similarity; barely related.
        0.1 Extremely low similarity; almost unrelated.
        0.0 No semantic similarity; completely unrelated.


        Actual: "{actual}"
        Expected: "{expected}"

        Return only the number.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }

        response = requests.post(self.endpoint, headers=headers, json=data, timeout=10)
        result = response.json()

        score_text = result["choices"][0]["message"]["content"].strip()

        # Extract numeric score from model response
        match = re.search(r"0?\.\d+|1\.0|1", score_text)
        if match:
            return float(match.group())
        else:
            return 0.0
