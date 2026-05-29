"""
utils.py
────────
Shared helpers: text cleaning, file loading, and saving summaries.
"""

import re
import os
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Normalise raw article text:
      - Strip HTML tags
      - Collapse newlines / tabs into single spaces
      - Collapse multiple spaces into one

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = re.sub(r'<[^>]+>', '', text)           # remove HTML tags
    text = re.sub(r'[\r\n\t]+', ' ', text)        # newlines → space
    text = re.sub(r' {2,}', ' ', text)            # collapse spaces
    return text.strip()


def load_article_from_file(filepath: str) -> str:
    """
    Read a .txt article from disk.

    Args:
        filepath: Absolute or relative path to the text file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError if the path does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_summary(
    article_title: str,
    abstractive: str,
    extractive: str,
    output_dir: str = "outputs",
) -> str:
    """
    Write both summaries to a timestamped .txt file.

    Args:
        article_title : Display name for the article.
        abstractive   : Abstractive summary text.
        extractive    : Extractive summary text.
        output_dir    : Folder to write the file into.

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
        f.write("ABSTRACTIVE SUMMARY (BART model):\n")
        f.write(abstractive + "\n\n")
        f.write("EXTRACTIVE SUMMARY (frequency-based):\n")
        f.write(extractive + "\n")

    return filepath
