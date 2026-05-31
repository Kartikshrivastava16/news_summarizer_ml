"""
main.py
───────
Command-line entry point for the News Article Summarizer.

Usage examples
──────────────
  python main.py                              # Summarise all sample articles (extractive)
  python main.py --mode both                  # Extractive + Abstractive on all samples
  python main.py --file path/article.txt      # Summarise a specific .txt file
  python main.py --file path/article.txt --mode abstractive
  python main.py --interactive                # Paste your own article in the terminal
  python main.py --web                        # Launch the local web UI (localhost:5000)
  python main.py --sentences 5               # Extract 5 sentences instead of 3
"""

import os
import sys
import argparse
import textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.extractive_summarizer  import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import clean_text, load_article_from_file, save_summary

SAMPLE_DIR = os.path.join(ROOT, "data", "sample_articles")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
W          = 72


# ── Display helpers ────────────────────────────────────────────────────

def header():
    bar = "═" * W
    print(f"\n{bar}")
    print("   📰  NEWS ARTICLE SUMMARIZER  —  ML / NLP Project")
    print("   Extractive · Abstractive (HuggingFace Transformers)")
    print(f"{bar}\n")


def print_result(title: str, extractive: str, abstractive: str = ""):
    bar  = "─" * W
    dots = "·" * (W - 2)
    print(f"\n{bar}")
    print(f"  ARTICLE : {title}")
    print(bar)

    if extractive:
        print("\n  ✂️   Extractive Summary  (frequency scoring · offline)")
        print(f"  {dots}")
        for line in textwrap.wrap(extractive, width=W - 4):
            print("    " + line)

    if abstractive:
        print("\n  🤖  Abstractive Summary  (HuggingFace BART · generated)")
        print(f"  {dots}")
        for line in textwrap.wrap(abstractive, width=W - 4):
            print("    " + line)

    print()


# ── Run modes ──────────────────────────────────────────────────────────

def run_demo(ext_m: ExtractiveSummarizer, abs_m: AbstractiveSummarizer, mode: str):
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt"))
    if not files:
        print("  [!] No .txt files found in data/sample_articles/")
        return

    for fname in files:
        title = fname.replace("_", " ").replace(".txt", "").title()
        raw   = load_article_from_file(os.path.join(SAMPLE_DIR, fname))
        text  = clean_text(raw)

        extractive  = ext_m.summarize(text) if mode in ("extractive", "both") else ""
        abstractive = abs_m.summarize(text) if mode in ("abstractive", "both") else ""

        print_result(title, extractive, abstractive)
        saved = save_summary(title, extractive or "(not requested)", OUTPUT_DIR, abstractive=abstractive)
        print(f"  💾  Saved → {saved}\n")


def run_file(path: str, ext_m: ExtractiveSummarizer, abs_m: AbstractiveSummarizer, mode: str):
    title = os.path.basename(path).replace("_", " ").replace(".txt", "").title()
    text  = clean_text(load_article_from_file(path))

    extractive  = ext_m.summarize(text) if mode in ("extractive", "both") else ""
    abstractive = abs_m.summarize(text) if mode in ("abstractive", "both") else ""

    print_result(title, extractive, abstractive)
    saved = save_summary(title, extractive or "(not requested)", OUTPUT_DIR, abstractive=abstractive)
    print(f"  💾  Saved → {saved}\n")


def run_interactive(ext_m: ExtractiveSummarizer, abs_m: AbstractiveSummarizer, mode: str):
    print("\n  📝  INTERACTIVE MODE")
    print("  Paste your article below. Type  END  on its own line when finished.\n")
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

    text = clean_text(raw)
    extractive  = ext_m.summarize(text) if mode in ("extractive", "both") else ""
    abstractive = abs_m.summarize(text) if mode in ("abstractive", "both") else ""

    print_result("User Article", extractive, abstractive)
    saved = save_summary("User Article", extractive or "(not requested)", OUTPUT_DIR, abstractive=abstractive)
    print(f"  💾  Saved → {saved}\n")


def run_web():
    try:
        from app import create_app
        flask_app = create_app()
        print("\n  🌐  Web UI starting at  http://localhost:5000")
        print("  Press  Ctrl+C  to stop.\n")
        flask_app.run(debug=False, host="0.0.0.0", port=5000)
    except ImportError:
        print("  [!] Flask not installed.  Run:  pip install flask")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="News Article Summarizer — ML/NLP Project")
    parser.add_argument("--interactive", action="store_true", help="Paste your own article in the terminal")
    parser.add_argument("--file",        type=str, default=None, help="Path to a .txt article file")
    parser.add_argument("--web",         action="store_true",    help="Launch the Flask web UI")
    parser.add_argument("--sentences",   type=int, default=3,    help="Number of sentences to extract (default: 3)")
    parser.add_argument(
        "--mode",
        type=str,
        default="extractive",
        choices=["extractive", "abstractive", "both"],
        help="Summarization mode (default: extractive)",
    )
    args = parser.parse_args()

    if args.web:
        run_web()
        return

    header()

    ext_m = ExtractiveSummarizer(num_sentences=args.sentences)
    abs_m = AbstractiveSummarizer()

    # Warn if abstractive mode requested but transformers not installed
    if args.mode in ("abstractive", "both") and not abs_m.is_available():
        print("  ⚠️  WARNING: transformers is not installed.")
        print("  Run:  pip install transformers torch sentencepiece")
        print("  Falling back to extractive-only mode.\n")
        args.mode = "extractive"

    if args.file:
        run_file(args.file, ext_m, abs_m, args.mode)
    elif args.interactive:
        run_interactive(ext_m, abs_m, args.mode)
    else:
        run_demo(ext_m, abs_m, args.mode)

    bar = "═" * W
    print(bar)
    print("  ✅  Done!  Summaries saved to  outputs/")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
