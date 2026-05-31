"""
src/summarizer.py
─────────────────
Extractive summarizer using word-frequency scoring.
Works entirely offline — no model download or internet needed.

Algorithm
─────────
1. Split article into sentences.
2. Build a normalised word-frequency table (stop-words excluded).
3. Score every sentence by the sum of its word frequencies.
4. Return the top-N sentences in their original reading order.
"""

import re
from collections import Counter
from typing import List


STOPWORDS: set = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or","but",
    "with","that","this","was","are","be","been","by","from","as","its","into",
    "than","then","so","if","about","has","have","had","not","which","who",
    "they","their","he","she","we","you","i","my","our","your","his","her",
    "also","more","can","will","would","could","should","may","might","shall",
    "do","did","does","just","very","much","many","some","such","when","where",
    "after","before","while","since","because","though","although",
}


class ExtractiveSummarizer:
    """
    Picks the most informative sentences from an article.
    Pure Python — no dependencies beyond the standard library.
    """

    def __init__(self, num_sentences: int = 3):
        if num_sentences < 1:
            raise ValueError("num_sentences must be at least 1.")
        self.num_sentences = num_sentences

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 20]

    def _word_frequencies(self, text: str) -> Counter:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        freq  = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
        if freq:
            max_freq = max(freq.values())
            for word in freq:
                freq[word] /= max_freq
        return freq

    def _score_sentence(self, sentence: str, freq: Counter) -> float:
        tokens = re.findall(r'\b[a-z]+\b', sentence.lower())
        return sum(freq[t] for t in tokens if t not in STOPWORDS)

    def summarize(self, text: str) -> str:
        """
        Extract the most important sentences from the article.

        Args:
            text: Cleaned article text.

        Returns:
            Top sentences joined as a paragraph.

        Raises:
            ValueError if text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        sentences = self._split_sentences(text)

        if len(sentences) <= self.num_sentences:
            return text.strip()

        freq   = self._word_frequencies(text)
        scores = [(s, self._score_sentence(s, freq)) for s in sentences]

        top = {s for s, _ in sorted(scores, key=lambda x: x[1], reverse=True)[: self.num_sentences]}
        return " ".join(s for s in sentences if s in top)
