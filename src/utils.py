"""
src/utils.py
────────────
Shared helpers: text cleaning, file I/O, output saving.
"""

import re
import os
from datetime import datetime


def clean_text(text: str) -> str:
    """Strip HTML tags, normalise whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_article_from_file(filepath: str) -> str:
    """Read a plain-text article from disk."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_summary(
    article_title: str,
    extractive: str,
    output_dir: str = "outputs",
    abstractive: str = "",
) -> str:
    """
    Write the summary/summaries to a timestamped .txt file in output_dir.

    Args:
        article_title:  Title of the article (used in filename + header).
        extractive:     Extractive summary text.
        output_dir:     Directory to write the file into (created if missing).
        abstractive:    Abstractive summary text (optional).

    Returns:
        Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[^\w\-]', '_', article_title)[:30]
    filepath   = os.path.join(output_dir, f"summary_{safe_title}_{timestamp}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"ARTICLE   : {article_title}\n")
        f.write(f"GENERATED : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("EXTRACTIVE SUMMARY  (frequency scoring · offline):\n")
        f.write("-" * 60 + "\n")
        f.write((extractive or "(none)") + "\n\n")

        if abstractive:
            f.write("ABSTRACTIVE SUMMARY  (HuggingFace BART · generated text):\n")
            f.write("-" * 60 + "\n")
            f.write(abstractive + "\n")

    return filepath
