"""
main.py
───────
Entry point for the News Article Summarizer.

Usage:
    python main.py                          # summarise all sample articles
    python main.py --file path/article.txt  # summarise a specific file
    python main.py --interactive            # paste your own article
"""

import os
import sys
import argparse
import textwrap

# ── make sure the project root is on the path ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ───────────────────────────────────────────────────────────────────────

from src.abstractive_summarizer import AbstractiveSummarizer
from src.extractive_summarizer import ExtractiveSummarizer
from src.utils import clean_text, load_article_from_file, save_summary

SAMPLE_DIR = os.path.join(ROOT, "data", "sample_articles")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
W = 70  # terminal line width


# ── pretty printers ─────────────────────────────────────────────────────

def header():
    print("\n" + "═" * W)
    print("   📰  NEWS ARTICLE SUMMARIZER")
    print("   NLP · Extractive  +  Abstractive  (BART)")
    print("═" * W + "\n")


def print_result(title: str, abstractive: str, extractive: str):
    print(f"\n{'─' * W}")
    print(f"  ARTICLE : {title}")
    print(f"{'─' * W}")

    print("\n  🤖  Abstractive Summary  (BART — Hugging Face Transformers)")
    print("  " + "·" * (W - 2))
    for line in textwrap.wrap(abstractive, width=W - 4):
        print("    " + line)

    print("\n  ✂️   Extractive Summary  (frequency-based · offline)")
    print("  " + "·" * (W - 2))
    for line in textwrap.wrap(extractive, width=W - 4):
        print("    " + line)
    print()


# ── run modes ────────────────────────────────────────────────────────────

def run_demo(abs_m: AbstractiveSummarizer, ext_m: ExtractiveSummarizer):
    """Summarise every .txt file in data/sample_articles/."""
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt"))
    if not files:
        print("  [!] No .txt files found in data/sample_articles/")
        return

    for fname in files:
        title = fname.replace("_", " ").replace(".txt", "").title()
        text  = clean_text(load_article_from_file(os.path.join(SAMPLE_DIR, fname)))

        abstract  = abs_m.summarize(text)
        extracted = ext_m.summarize(text)

        print_result(title, abstract, extracted)
        saved = save_summary(title, abstract, extracted, OUTPUT_DIR)
        print(f"  💾  Saved → {saved}\n")


def run_file(path: str, abs_m: AbstractiveSummarizer, ext_m: ExtractiveSummarizer):
    """Summarise a single user-specified .txt file."""
    title     = os.path.basename(path).replace("_", " ").replace(".txt", "").title()
    text      = clean_text(load_article_from_file(path))
    abstract  = abs_m.summarize(text)
    extracted = ext_m.summarize(text)

    print_result(title, abstract, extracted)
    saved = save_summary(title, abstract, extracted, OUTPUT_DIR)
    print(f"  💾  Saved → {saved}\n")


def run_interactive(abs_m: AbstractiveSummarizer, ext_m: ExtractiveSummarizer):
    """Accept article text typed / pasted in the terminal."""
    print("\n  📝  INTERACTIVE MODE")
    print("  Paste your article below.")
    print("  Type  END  on a new blank line when finished.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    raw = " ".join(lines)
    if not raw.strip():
        print("  [!] No text entered. Exiting.")
        return

    text      = clean_text(raw)
    abstract  = abs_m.summarize(text)
    extracted = ext_m.summarize(text)

    print_result("User Article", abstract, extracted)
    saved = save_summary("User Article", abstract, extracted, OUTPUT_DIR)
    print(f"  💾  Saved → {saved}\n")


# ── entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="News Article Summarizer")
    parser.add_argument("--interactive", action="store_true",
                        help="Paste your own article in the terminal.")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a .txt file to summarise.")
    args = parser.parse_args()

    header()

    abs_m = AbstractiveSummarizer()
    ext_m = ExtractiveSummarizer(num_sentences=3)

    if args.file:
        run_file(args.file, abs_m, ext_m)
    elif args.interactive:
        run_interactive(abs_m, ext_m)
    else:
        run_demo(abs_m, ext_m)

    print("═" * W)
    print("  ✅  Done! Summaries saved to  outputs/")
    print("═" * W + "\n")


if __name__ == "__main__":
    main()
