"""
extractive_summarizer.py
────────────────────────
Offline, frequency-based extractive summarizer.
No model download required.

Scores every sentence by the total frequency of its
meaningful words, then returns the top-N sentences in
their original order.
"""

import re
from collections import Counter
from typing import List


STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "that", "this", "was", "are", "be", "been",
    "by", "from", "as", "its", "into", "than", "then", "so", "if", "about",
    "has", "have", "had", "not", "which", "who", "they", "their", "he", "she",
    "we", "you", "i", "my", "our", "your", "his", "her", "also", "more", "can",
}


class ExtractiveSummarizer:
    """
    Picks the most informative existing sentences from an article.
    Works entirely offline — no model or internet needed.
    """

    def __init__(self, num_sentences: int = 3):
        """
        Args:
            num_sentences: How many sentences to extract.
        """
        self.num_sentences = num_sentences

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _word_freq(self, text: str) -> Counter:
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(w for w in words if w not in STOPWORDS and len(w) > 2)

    def _score(self, sentence: str, freq: Counter) -> float:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        return sum(freq[t] for t in tokens if t not in STOPWORDS)

    def summarize(self, text: str) -> str:
        """
        Extract the most important sentences from the text.

        Args:
            text: Raw article text.

        Returns:
            Key sentences joined as a paragraph.

        Raises:
            ValueError if text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        sentences = self._split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text.strip()

        freq = self._word_freq(text)
        ranked = sorted(sentences, key=lambda s: self._score(s, freq), reverse=True)
        top = set(ranked[: self.num_sentences])

        # Return sentences in their original order
        return " ".join(s for s in sentences if s in top)
