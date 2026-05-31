# 📰 News Article Summarizer — ML/NLP Project

**Kartik Shrivastava** · Hex Softwares Internship · Project 1

Automatically summarizes news articles into concise key points using two NLP techniques — offline frequency-based extractive summarization and HuggingFace BART-powered abstractive summarization. Accessible via a Flask web UI or the command line.

---

## 🗂️ Project Structure

```
news_summarizer_ml/
│
├── main.py                              ← CLI entry point (all modes)
├── app.py                               ← Flask web server
├── config.py                            ← Central configuration
├── requirements.txt                     ← All Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── extractive_summarizer.py         ← Frequency-based NLP (offline, no download)
│   ├── abstractive_summarizer.py        ← HuggingFace BART (seq2seq generation)
│   ├── summarizer.py                    ← Alias for ExtractiveSummarizer
│   └── utils.py                         ← Text cleaning, file I/O, save_summary
│
├── templates/
│   ├── index.html                       ← Main web UI (black & white newspaper theme)
│   └── redirect.html                    ← Server-detection page (checks localhost:5000)
│
├── static/
│   ├── css/style.css                    ← Full stylesheet (design tokens, responsive)
│   └── js/app.js                        ← Client logic + offline JS extractive fallback
│
├── notebooks/
│   └── summarizer_demo.ipynb            ← Jupyter walkthrough (9 sections)
│
├── data/
│   └── sample_articles/
│       ├── ai_breakthrough.txt
│       ├── climate_report.txt
│       └── space_exploration.txt
│
├── outputs/                             ← Auto-saved summaries (timestamped .txt files)
│
└── tests/
    ├── __init__.py
    └── test_summarizers.py              ← 26 unit tests (pytest)
```

---

## 🧠 How It Works

### Extractive Summarization — offline, instant

Pure Python word-frequency scoring. No model download, no internet, no GPU needed.

1. **Clean** — strip HTML tags, collapse whitespace via `utils.clean_text()`
2. **Split** — tokenise into sentences on `.!?` boundaries (min 20 chars each)
3. **Score** — build normalised word-frequency table (stop-words excluded), score each sentence by summing its token frequencies
4. **Extract** — pick top-N sentences, re-join in original reading order

### Abstractive Summarization — HuggingFace Transformers

Sequence-to-sequence generation using `facebook/bart-large-cnn`. Generates brand-new text, not just picked sentences.

1. **Clean & Chunk** — strip HTML, split articles over 3000 chars on sentence boundaries
2. **Encode** — `BartTokenizer` converts text to token IDs (truncated to 1024 tokens)
3. **Generate** — `BartForConditionalGeneration` runs beam search to produce a condensed sequence
4. **Decode** — token IDs decoded back to human-readable summary text

> First run downloads ~1.6 GB of model weights into HuggingFace's local cache. All subsequent runs are fully offline.

The pipeline loader uses `BartForConditionalGeneration` + `BartTokenizer` directly (not `pipeline(task="summarization")`) so it works correctly across all `transformers` versions including v5+.

---

## ⚙️ Setup

```bash
# 1. Clone / open the project folder
cd news_summarizer_ml

# 2. Install all dependencies
pip install -r requirements.txt
```

**CPU-only PyTorch** (default, smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU — CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Extractive mode only needs `flask` and `pytest`. The heavy ML packages (`transformers`, `torch`, `sentencepiece`) are only required for abstractive mode. The app gracefully falls back to extractive-only if they are not installed.

---

## ▶️ Running the Project

### Web UI (recommended)

```bash
python app.py
# Open http://localhost:5000 in your browser
```

Or via `main.py`:
```bash
python main.py --web
```

The web UI supports all three modes (Extractive / Abstractive / Both), an adjustable sentence slider (1–10), live character/word counter, copy-to-clipboard, and auto-saves every result to `outputs/`.

### CLI — Demo mode (all 3 sample articles)

```bash
python main.py                            # extractive only (default)
python main.py --mode both                # extractive + abstractive
python main.py --mode abstractive         # abstractive only
python main.py --sentences 5             # extract 5 sentences
```

### CLI — Single file

```bash
python main.py --file data/sample_articles/climate_report.txt
python main.py --file data/sample_articles/ai_breakthrough.txt --mode both
```

### CLI — Interactive (paste your own article)

```bash
python main.py --interactive
python main.py --interactive --mode both
# Paste text, then type END on its own line to finish
```

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook summarizer_demo.ipynb
```

The notebook has 9 sections: setup, imports, article loading, extractive demo (with sentence-count variation), abstractive demo (gracefully skipped if `transformers` not installed), side-by-side comparison, utility demos, error handling, and a "try your own article" cell.

### Tests

```bash
# All 26 tests
python -m pytest tests/ -v

# Offline only — no model download
python -m pytest tests/ -v -k "not abstractive"

# Abstractive tests only — requires transformers + ~1.6 GB download on first run
python -m pytest tests/ -v -k "abstractive"

# Quick pass/fail summary
python -m pytest -q
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `flask` | ≥ 3.0.0 | Web UI server |
| `transformers` | ≥ 4.40.0 | HuggingFace BART model |
| `torch` | ≥ 2.2.0 | PyTorch backend for BART |
| `sentencepiece` | ≥ 0.2.0 | Tokenizer for T5 / other models |
| `accelerate` | ≥ 0.29.0 | Faster model loading |
| `pytest` | ≥ 8.0.0 | Unit tests |

---

## 💾 Output Files

Every summarization run saves a timestamped `.txt` file to `outputs/`. Example:

```
ARTICLE   : Ai Breakthrough
GENERATED : 2026-05-31 17:50:47
============================================================

EXTRACTIVE SUMMARY  (frequency scoring · offline):
------------------------------------------------------------
The AI model uses a transformer-based architecture trained on
millions of research papers...

ABSTRACTIVE SUMMARY  (HuggingFace BART · generated text):
------------------------------------------------------------
Scientists unveiled an AI system capable of diagnosing rare
diseases with 94% accuracy using only patient notes...
```

Filename format: `summary_<Title>_<YYYYMMDD>_<HHMMSS>.txt`

The abstractive section is omitted when only extractive mode is used.

---

## 🔧 Configuration

Edit `config.py` to change defaults without touching any other file:

```python
DEFAULT_SENTENCES = 3      # sentences to extract by default
MIN_SENTENCES     = 1      # slider minimum
MAX_SENTENCES     = 10     # slider maximum

MIN_ARTICLE_CHARS = 80     # reject articles shorter than this
MAX_ARTICLE_CHARS = 6000   # textarea character cap in the web UI

FLASK_HOST  = "0.0.0.0"   # bind address
FLASK_PORT  = 5000         # web server port
FLASK_DEBUG = False        # set True for development hot-reload

OUTPUT_DIR  = "outputs"             # where .txt summaries are saved
SAMPLE_DIR  = "data/sample_articles"
```

---

## 🌐 API Reference (Flask)

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI (`templates/index.html`) |
| `GET` | `/model-status` | Returns `{ abstractive_available, model_name }` |
| `POST` | `/summarize` | Runs summarization, returns JSON results |

`POST /summarize` — request body (JSON):

```json
{
  "text":          "Full article text here...",
  "title":         "Article Title",
  "num_sentences": 3,
  "mode":          "extractive"
}
```

`mode` options: `"extractive"` · `"abstractive"` · `"both"`

Response:

```json
{
  "extractive":  "Top sentences from the article...",
  "abstractive": "BART-generated summary...",
  "mode":        "both"
}
```

---

## 📋 Project Requirements (Hex Softwares)

| Requirement | Status |
|---|---|
| ML model to automatically summarize news articles into concise key points | ✅ |
| NLP — Extractive summarization | ✅ |
| NLP — Abstractive summarization with HuggingFace Transformers | ✅ |
| Saves time for readers, provides quick insights | ✅ |
| Enhances information accessibility | ✅ |

---

*Hex Softwares Internship · ML/NLP Project 01*
