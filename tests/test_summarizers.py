"""
tests/test_summarizers.py
─────────────────────────
Unit tests for ExtractiveSummarizer and utils.
Abstractive tests are skipped by default (avoids the ~1.6 GB model download).

Run:
    python -m pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractive_summarizer import ExtractiveSummarizer
from src.utils import clean_text, save_summary

SAMPLE = (
    "Scientists discovered a new species of frog in the Amazon rainforest. "
    "The frog, named Ranitomeya fantastica, is brightly colored in red and black. "
    "Researchers believe its vivid colors warn predators of its toxicity. "
    "The discovery was published in the journal Herpetologica. "
    "This brings the total number of known poison dart frog species to 175. "
    "Conservation experts say protecting its habitat is critical for survival."
)


class TestExtractiveSummarizer:

    def test_returns_string(self):
        result = ExtractiveSummarizer(num_sentences=2).summarize(SAMPLE)
        assert isinstance(result, str) and len(result) > 0

    def test_summary_shorter_than_original(self):
        result = ExtractiveSummarizer(num_sentences=2).summarize(SAMPLE)
        assert len(result) < len(SAMPLE)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ExtractiveSummarizer().summarize("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            ExtractiveSummarizer().summarize("   ")

    def test_short_text_returned_unchanged(self):
        short = "This is a single short sentence."
        result = ExtractiveSummarizer(num_sentences=3).summarize(short)
        assert result == short


class TestUtils:

    def test_clean_strips_html(self):
        assert clean_text("<p>Hello <b>world</b>!</p>") == "Hello world!"

    def test_clean_collapses_whitespace(self):
        assert clean_text("Hello   \n\n  world") == "Hello world"

    def test_clean_empty_string(self):
        assert clean_text("") == ""

    def test_save_creates_file(self, tmp_path):
        path = save_summary("Test", "Abstract.", "Extract.", str(tmp_path))
        assert os.path.isfile(path)

    def test_save_file_contains_summaries(self, tmp_path):
        path = save_summary("Test", "Abstract result.", "Extract result.", str(tmp_path))
        content = open(path).read()
        assert "Abstract result." in content
        assert "Extract result." in content
