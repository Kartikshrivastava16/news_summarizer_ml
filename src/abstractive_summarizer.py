"""
src/abstractive_summarizer.py
─────────────────────────────
Abstractive summarizer using HuggingFace Transformers (BART / T5).

Unlike extractive summarization (which picks existing sentences),
abstractive summarization GENERATES new text — rephrasing and
condensing the article just like a human would.

Model used by default: facebook/bart-large-cnn
  • State-of-the-art news summarization model
  • ~1.6 GB download on first use (cached afterwards)
  • No internet needed after the first download

Alternative lightweight model (faster, smaller):
  • sshleifer/distilbart-cnn-12-6  (~900 MB)

Usage
─────
    from src.abstractive_summarizer import AbstractiveSummarizer

    model = AbstractiveSummarizer()           # downloads on first call
    summary = model.summarize(article_text)
    print(summary)
"""

from __future__ import annotations

import warnings
from typing import Optional


# Default model — best quality for news summarization
DEFAULT_MODEL = "facebook/bart-large-cnn"

# Lightweight alternative (set env var NEWSML_MODEL to override)
LIGHTWEIGHT_MODEL = "sshleifer/distilbart-cnn-12-6"


class AbstractiveSummarizer:
    """
    Generates a brand-new condensed version of a news article
    using a pre-trained sequence-to-sequence transformer model.

    The model is loaded lazily on the first call to summarize()
    so importing this class has zero overhead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_output_tokens: int = 130,
        min_output_tokens: int = 30,
        device: Optional[int] = None,
    ):
        """
        Args:
            model_name:         HuggingFace model hub name.
            max_output_tokens:  Maximum tokens in the generated summary.
            min_output_tokens:  Minimum tokens in the generated summary.
            device:             GPU device index (0 for first GPU).
                                Pass -1 or None to force CPU.
        """
        self.model_name        = model_name
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        self.device            = device

        # Pipeline is loaded on first summarize() call (lazy loading)
        self._pipeline = None

    # ── Private helpers ────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        """
        Load the HuggingFace summarization pipeline.
        Downloads the model weights on the first call (~1.6 GB for BART).
        Subsequent runs use the local cache — no internet needed.
        """
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "HuggingFace Transformers is not installed.\n"
                "Run:  pip install transformers torch sentencepiece"
            ) from exc

        print(f"\n  [AbstractiveSummarizer] Loading model: {self.model_name}")
        print("  (First run downloads ~1.6 GB — cached afterwards)\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline = pipeline(
                task="summarization",
                model=self.model_name,
                device=self.device,         # None → auto (CPU / GPU)
                truncation=True,
            )

        print(f"  [AbstractiveSummarizer] Model ready.\n")

    def _chunk_text(self, text: str, max_chars: int = 3000) -> list[str]:
        """
        Split very long articles into chunks so they fit within
        the model's token limit (1024 tokens for BART).
        Splits on sentence boundaries where possible.
        """
        if len(text) <= max_chars:
            return [text]

        chunks, current = [], ""
        for sentence in text.replace("\n", " ").split(". "):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 2 > max_chars:
                if current:
                    chunks.append(current.strip())
                current = sentence + ". "
            else:
                current += sentence + ". "
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text[:max_chars]]

    # ── Public API ─────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """
        Generate an abstractive summary of the article.

        Args:
            text: Cleaned article text (use utils.clean_text first).

        Returns:
            A newly generated summary string.

        Raises:
            ValueError:  if text is empty or too short.
            ImportError: if transformers / torch are not installed.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        if len(text.strip()) < 80:
            raise ValueError("Article too short for abstractive summarization (min 80 chars).")

        # Lazy-load the model on first call
        if self._pipeline is None:
            self._load_pipeline()

        chunks   = self._chunk_text(text)
        summaries = []

        for chunk in chunks:
            # Clamp token limits to the chunk length to avoid warnings
            max_tok = min(self.max_output_tokens, max(30, len(chunk.split()) // 2))
            min_tok = min(self.min_output_tokens, max_tok - 5)

            result = self._pipeline(
                chunk,
                max_length=max_tok,
                min_length=max(5, min_tok),
                do_sample=False,        # deterministic / reproducible
                truncation=True,
            )
            summaries.append(result[0]["summary_text"].strip())

        return " ".join(summaries)

    def is_available(self) -> bool:
        """
        Return True if the transformers package is importable.
        Use this to gracefully fall back to extractive mode in the UI.
        """
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        loaded = "loaded" if self._pipeline else "not loaded"
        return f"AbstractiveSummarizer(model='{self.model_name}', pipeline={loaded})"
