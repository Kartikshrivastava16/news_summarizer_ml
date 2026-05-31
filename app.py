"""
app.py
──────
Flask web application for the News Article Summarizer.

Start with:
    python app.py

Then open:  http://localhost:5000

Routes
──────
GET  /              → serves the web UI
POST /summarize     → runs extractive + (optionally) abstractive summarization
GET  /model-status  → returns whether the abstractive model is available
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify
from src.extractive_summarizer  import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import clean_text, save_summary

OUTPUT_DIR = os.path.join(ROOT, "outputs")


def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    # Extractive model — zero cost, always available
    ext_model = ExtractiveSummarizer(num_sentences=3)

    # Abstractive model — lazy loaded on first request
    abs_model = AbstractiveSummarizer()

    # ── Routes ─────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/model-status")
    def model_status():
        """Tell the frontend whether HuggingFace Transformers is installed."""
        return jsonify({
            "abstractive_available": abs_model.is_available(),
            "model_name": abs_model.model_name,
        })

    @app.route("/summarize", methods=["POST"])
    def summarize():
        data   = request.get_json(force=True)
        text   = data.get("text", "").strip()
        title  = data.get("title", "Untitled Article").strip() or "Untitled Article"
        n_sent = int(data.get("num_sentences", 3))
        mode   = data.get("mode", "extractive")   # "extractive" | "abstractive" | "both"

        # ── Validation ──────────────────────────────────────────────────
        if not text:
            return jsonify({"error": "Article text is required."}), 400
        if len(text) < 80:
            return jsonify({"error": "Article too short. Please provide more text."}), 400

        cleaned = clean_text(text)

        extractive_result  = ""
        abstractive_result = ""

        # ── Extractive ──────────────────────────────────────────────────
        if mode in ("extractive", "both"):
            try:
                ext_model.num_sentences = n_sent
                extractive_result = ext_model.summarize(cleaned)
            except Exception as exc:
                return jsonify({"error": f"Extractive error: {exc}"}), 500

        # ── Abstractive ─────────────────────────────────────────────────
        if mode in ("abstractive", "both"):
            if not abs_model.is_available():
                return jsonify({
                    "error": (
                        "HuggingFace Transformers is not installed. "
                        "Run: pip install transformers torch sentencepiece"
                    )
                }), 503
            try:
                abstractive_result = abs_model.summarize(cleaned)
            except Exception as exc:
                return jsonify({"error": f"Abstractive error: {exc}"}), 500

        # ── Save & respond ──────────────────────────────────────────────
        save_summary(
            title,
            extractive_result or "(not requested)",
            OUTPUT_DIR,
            abstractive=abstractive_result,
        )

        return jsonify({
            "extractive":  extractive_result,
            "abstractive": abstractive_result,
            "mode":        mode,
        })

    return app


if __name__ == "__main__":
    flask_app = create_app()
    print("\n  🌐  Web UI running at  http://localhost:5000")
    flask_app.run(debug=False, host="0.0.0.0", port=5000)
