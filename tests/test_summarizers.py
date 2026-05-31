"""
tests/test_summarizers.py
─────────────────────────
Unit tests for ExtractiveSummarizer, AbstractiveSummarizer, and utils.

Run all tests:
    python -m pytest tests/ -v

Run only extractive tests (no model download needed):
    python -m pytest tests/ -v -k "not abstractive"

Run only abstractive tests (requires transformers + ~1.6 GB download):
    python -m pytest tests/ -v -k "abstractive"
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractive_summarizer  import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import clean_text, save_summary

# ── Shared sample article ──────────────────────────────────────────────
SAMPLE = (
    "Scientists discovered a new species of frog in the Amazon rainforest. "
    "The frog, named Ranitomeya fantastica, is brightly colored in red and black. "
    "Researchers believe its vivid colors warn predators of its toxicity. "
    "The discovery was published in the journal Herpetologica. "
    "This brings the total number of known poison dart frog species to 175. "
    "Conservation experts say protecting its habitat is critical for survival."
)


# ══════════════════════════════════════════════════════════════════════
# Extractive Summarizer Tests
# ══════════════════════════════════════════════════════════════════════

class TestExtractiveSummarizer:

    def test_returns_string(self):
        result = ExtractiveSummarizer(num_sentences=2).summarize(SAMPLE)
        assert isinstance(result, str) and len(result) > 0

    def test_summary_shorter_than_original(self):
        result = ExtractiveSummarizer(num_sentences=2).summarize(SAMPLE)
        assert len(result) < len(SAMPLE)

    def test_correct_sentence_count(self):
        """Summary should not contain more sentences than requested."""
        result = ExtractiveSummarizer(num_sentences=2).summarize(SAMPLE)
        # Count rough sentence endings
        count = result.count('.') + result.count('!') + result.count('?')
        assert count <= 3  # allow slight variance for abbreviations

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ExtractiveSummarizer().summarize("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            ExtractiveSummarizer().summarize("   ")

    def test_short_text_returned_as_is(self):
        short = "This is a single short sentence."
        result = ExtractiveSummarizer(num_sentences=3).summarize(short)
        assert result == short

    def test_invalid_num_sentences_raises(self):
        with pytest.raises(ValueError):
            ExtractiveSummarizer(num_sentences=0)

    def test_sentences_in_original_order(self):
        """Extracted sentences must appear in original article order."""
        result = ExtractiveSummarizer(num_sentences=3).summarize(SAMPLE)
        positions = [SAMPLE.find(s) for s in result.split('. ') if s in SAMPLE]
        assert positions == sorted(positions)


# ══════════════════════════════════════════════════════════════════════
# Abstractive Summarizer Tests
# ══════════════════════════════════════════════════════════════════════

class TestAbstractiveSummarizer:

    def test_is_available_returns_bool(self):
        """is_available() must always return a bool, regardless of install."""
        model = AbstractiveSummarizer()
        result = model.is_available()
        assert isinstance(result, bool)

    def test_repr_contains_model_name(self):
        model = AbstractiveSummarizer()
        assert "facebook/bart-large-cnn" in repr(model)

    def test_empty_raises(self):
        model = AbstractiveSummarizer()
        with pytest.raises(ValueError):
            model.summarize("")

    def test_too_short_raises(self):
        model = AbstractiveSummarizer()
        with pytest.raises(ValueError):
            model.summarize("Too short.")

    @pytest.mark.skipif(
        not AbstractiveSummarizer().is_available(),
        reason="transformers not installed — skipping model inference tests"
    )
    def test_summarize_returns_string(self):
        model = AbstractiveSummarizer()
        result = model.summarize(SAMPLE)
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.skipif(
        not AbstractiveSummarizer().is_available(),
        reason="transformers not installed — skipping model inference tests"
    )
    def test_summary_shorter_than_original(self):
        model = AbstractiveSummarizer()
        result = model.summarize(SAMPLE)
        assert len(result) < len(SAMPLE)

    @pytest.mark.skipif(
        not AbstractiveSummarizer().is_available(),
        reason="transformers not installed — skipping model inference tests"
    )
    def test_pipeline_cached_after_first_call(self):
        """Second call must reuse the loaded pipeline (no reload)."""
        model = AbstractiveSummarizer()
        model.summarize(SAMPLE)
        assert model._pipeline is not None
        model.summarize(SAMPLE)  # should NOT re-print loading messages

    @pytest.mark.skipif(
        not AbstractiveSummarizer().is_available(),
        reason="transformers not installed — skipping model inference tests"
    )
    def test_custom_model_name(self):
        """Lightweight distilbart model should also work."""
        model = AbstractiveSummarizer(model_name="sshleifer/distilbart-cnn-12-6")
        result = model.summarize(SAMPLE)
        assert isinstance(result, str) and len(result) > 0


# ══════════════════════════════════════════════════════════════════════
# Utils Tests
# ══════════════════════════════════════════════════════════════════════

class TestUtils:

    def test_clean_strips_html(self):
        assert clean_text("<p>Hello <b>world</b>!</p>") == "Hello world!"

    def test_clean_collapses_whitespace(self):
        assert clean_text("Hello   \n\n  world") == "Hello world"

    def test_clean_empty_string(self):
        assert clean_text("") == ""

    def test_clean_preserves_normal_text(self):
        text = "NASA launched a new rocket yesterday."
        assert clean_text(text) == text

    def test_save_creates_file(self, tmp_path):
        """save_summary with only extractive text should create a file."""
        path = save_summary("Test Article", "Extractive summary.", str(tmp_path))
        assert os.path.isfile(path)

    def test_save_extractive_in_file(self, tmp_path):
        """Extractive summary must appear in the saved file."""
        path = save_summary("Test", "Extract result.", str(tmp_path))
        content = open(path, encoding="utf-8").read()
        assert "Extract result." in content

    def test_save_abstractive_in_file(self, tmp_path):
        """When abstractive text is provided it must appear in the saved file."""
        path = save_summary("Test", "Extract result.", str(tmp_path), abstractive="Abstract result.")
        content = open(path, encoding="utf-8").read()
        assert "Extract result."  in content
        assert "Abstract result." in content

    def test_save_no_abstractive_section_when_empty(self, tmp_path):
        """When abstractive is not provided, the section should be absent."""
        path = save_summary("Test", "Extract.", str(tmp_path))
        content = open(path, encoding="utf-8").read()
        assert "ABSTRACTIVE" not in content

    def test_save_filename_contains_title(self, tmp_path):
        path = save_summary("My Article", "Summary.", str(tmp_path))
        assert "My_Article" in os.path.basename(path)

    def test_save_creates_output_dir_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "new_subdir")
        path = save_summary("Test", "Summary.", new_dir)
        assert os.path.isdir(new_dir)
        assert os.path.isfile(path)
