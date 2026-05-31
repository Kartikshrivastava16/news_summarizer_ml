"""
config.py
─────────
Central configuration for the News Article Summarizer.
Edit values here to change app behaviour without touching other files.
"""

# ── Summarizer defaults ────────────────────────────────────────────────
DEFAULT_SENTENCES   = 3      # how many sentences to extract by default
MIN_SENTENCES       = 1
MAX_SENTENCES       = 10

MIN_ARTICLE_CHARS   = 80     # reject articles shorter than this
MAX_ARTICLE_CHARS   = 6000   # textarea cap

# ── Flask server ───────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

# ── Output directory ───────────────────────────────────────────────────
OUTPUT_DIR      = "outputs"
SAMPLE_DIR      = "data/sample_articles"
