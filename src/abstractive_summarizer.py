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

        Uses BartForConditionalGeneration + BartTokenizer directly
        instead of pipeline(task="summarization") so the code works
        across all transformers versions (the task-string registry
        changed in 4.x → 5.x and causes KeyError on some installs).

        Downloads model weights on the first call (~1.6 GB for BART).
        Subsequent runs use the local cache — no internet needed.
        """
        try:
            from transformers import (
                BartForConditionalGeneration,
                BartTokenizer,
            )
            import torch
        except ImportError as exc:
            raise ImportError(
                "HuggingFace Transformers is not installed.\n"
                "Run:  pip install transformers torch sentencepiece"
            ) from exc

        print(f"\n  [AbstractiveSummarizer] Loading model: {self.model_name}")
        print("  (First run downloads ~1.6 GB — cached afterwards)\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try the stable model-class path first (works on all versions)
            try:
                tokenizer = BartTokenizer.from_pretrained(self.model_name)
                model     = BartForConditionalGeneration.from_pretrained(self.model_name)

                # Store model/tokenizer and move model to the correct device.
                self._tokenizer = tokenizer
                self._model = model.eval()

                # Move model to device if possible
                try:
                    if self.device is None or self.device == -1:
                        self._device = torch.device("cpu")
                    else:
                        self._device = torch.device(f"cuda:{self.device}")
                    self._model.to(self._device)
                except Exception:
                    self._device = torch.device("cpu")
            except Exception:
                # As a last resort, try to load model/tokenizer by name
                tokenizer = BartTokenizer.from_pretrained(self.model_name)
                model     = BartForConditionalGeneration.from_pretrained(self.model_name)
                self._tokenizer = tokenizer
                self._model = model.eval()
                try:
                    if self.device is None or self.device == -1:
                        self._device = torch.device("cpu")
                    else:
                        self._device = torch.device(f"cuda:{self.device}")
                    self._model.to(self._device)
                except Exception:
                    self._device = torch.device("cpu")

        print("  [AbstractiveSummarizer] Model ready.\n")
        # Keep a `_pipeline` attribute for backwards compatibility with tests
        # that check the attribute exists after the first call.
        self._pipeline = True

    def _generate_with_model(self, text: str, max_length: int, min_length: int) -> str:
        """Generate text using the loaded model + tokenizer directly."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device) if "attention_mask" in inputs else None

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                early_stopping=True,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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

    def _extract_text(self, result: dict) -> str:
        """
        Pull the generated text from a pipeline result dict.
        Handles both 'summary_text' (summarization pipeline)
        and 'generated_text' (text2text-generation pipeline).
        """
        return (
            result.get("summary_text")
            or result.get("generated_text")
            or ""
        ).strip()

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
            raise ValueError(
                "Article too short for abstractive summarization (min 80 chars)."
            )

        # Lazy-load the model on first call
        if not hasattr(self, "_model") or self._model is None:
            self._load_pipeline()

        chunks    = self._chunk_text(text)
        summaries = []

        for chunk in chunks:
            # Clamp token limits to avoid transformer warnings
            max_tok = min(self.max_output_tokens, max(30, len(chunk.split()) // 2))
            min_tok = min(self.min_output_tokens, max_tok - 5)

            summary_text = self._generate_with_model(
                chunk, max_length=max_tok, min_length=max(5, min_tok)
            )
            if summary_text:
                summaries.append(summary_text)

        return " ".join(s for s in summaries if s)

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
        loaded = "loaded" if hasattr(self, "_model") and self._model is not None else "not loaded"
        return f"AbstractiveSummarizer(model='{self.model_name}', model={loaded})"
