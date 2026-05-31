# 📰 News Article Summarizer — ML/NLP Project

**Kartik Shrivastava** · Hex Softwares Internship · Project 1

---

## 🗂️ Project Structure

```
news_summarizer_ml/
│
├── main.py                              ← CLI entry point (all modes)
├── app.py                               ← Flask web server
├── config.py                            ← Central configuration
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── extractive_summarizer.py         ← Frequency-based NLP (offline)
│   ├── abstractive_summarizer.py        ← HuggingFace BART/T5 (generative)
│   ├── summarizer.py                    ← Alias ExtractiveSummarizer
│   └── utils.py                         ← Text cleaning, file I/O, saving
│
├── templates/
│   └── index.html                       ← Web UI (black & white newspaper theme)
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── notebooks/
│   └── summarizer_demo.ipynb            ← Jupyter notebook walkthrough
│
├── data/
│   └── sample_articles/
│       ├── ai_breakthrough.txt
│       ├── climate_report.txt
│       └── space_exploration.txt
│
├── outputs/                             ← Auto-saved summaries (timestamped .txt)
│
└── tests/
    └── test_summarizers.py
```

---

## 🧠 How It Works

### Extractive Summarization (offline, no download)
Word-frequency scoring — pure Python, zero dependencies beyond stdlib:

1. **Clean** — strip HTML tags, collapse whitespace
2. **Split** — tokenise text into sentences on `.!?` boundaries
3. **Score** — build normalised word-frequency table (stop-words excluded), score each sentence
4. **Extract** — pick top-N sentences, re-join in original reading order

### Abstractive Summarization (HuggingFace Transformers)
Sequence-to-sequence generation using `facebook/bart-large-cnn`:

1. **Clean & Chunk** — strip HTML, split articles > 3000 chars into sentence-boundary chunks
2. **Encode** — BART tokenizer converts text to token IDs (truncated to 1024)
3. **Generate** — seq2seq model generates brand-new condensed sentences (beam search)
4. **Decode** — token IDs decoded back to human-readable summary

> First run downloads ~1.6 GB of model weights (cached afterwards — no internet needed after that).

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (default):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU (CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## ▶️ Run

### Web UI
```bash
python app.py
# Open: http://localhost:5000
```

### CLI — Demo (all sample articles)
```bash
python main.py                          # extractive only (default)
python main.py --mode both              # extractive + abstractive
python main.py --mode abstractive       # abstractive only
python main.py --sentences 5            # extract 5 sentences
```

### CLI — Specific File
```bash
python main.py --file data/sample_articles/climate_report.txt
python main.py --file data/sample_articles/ai_breakthrough.txt --mode both
```

### CLI — Interactive (paste in terminal)
```bash
python main.py --interactive
python main.py --interactive --mode both
```

### Tests
```bash
# All tests (extractive + utils; skips abstractive if transformers not installed)
python -m pytest tests/ -v

# Offline only (no model download)
python -m pytest tests/ -v -k "not abstractive"

# Abstractive tests (requires transformers + ~1.6 GB download)
python -m pytest tests/ -v -k "abstractive"
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web UI server |
| `transformers` | HuggingFace BART model |
| `torch` | PyTorch backend for transformers |
| `sentencepiece` | Tokenizer for T5 / other models |
| `accelerate` | Faster model loading (optional) |
| `pytest` | Unit tests |

> **Extractive mode** requires only `flask` and `pytest` — no heavy ML libraries needed.

---

## 💾 Output Files

Every summarization run saves a timestamped `.txt` file to `outputs/`:

```
ARTICLE   : Ai Breakthrough
GENERATED : 2024-01-15 14:32:07
============================================================

EXTRACTIVE SUMMARY  (frequency scoring · offline):
------------------------------------------------------------
The AI model uses a transformer-based architecture...

ABSTRACTIVE SUMMARY  (HuggingFace BART · generated text):
------------------------------------------------------------
Scientists unveiled an AI system that diagnoses rare diseases...
```

---

## 🔧 Configuration

Edit `config.py` to change defaults without touching other files:

```python
DEFAULT_SENTENCES = 3      # sentences to extract
MIN_ARTICLE_CHARS = 80     # minimum article length
FLASK_PORT        = 5000   # web server port
```
