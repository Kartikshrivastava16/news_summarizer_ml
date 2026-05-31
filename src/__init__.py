"""
src/__init__.py
───────────────
Makes src/ a proper Python package and exports the public API.
"""

from .summarizer              import ExtractiveSummarizer
from .abstractive_summarizer  import AbstractiveSummarizer
from .utils                   import clean_text, load_article_from_file, save_summary

__all__ = [
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
    "clean_text",
    "load_article_from_file",
    "save_summary",
]
