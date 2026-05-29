# 📰 News Article Summarizer

**Kartik Shrivastava**  
ML Project — NLP · Extractive + Abstractive Summarization

---

## 📁 Project Structure

```
news_summarizer_ml/
│
├── main.py                         ← Entry point — run this
│
├── src/                            ← Core Python package
│   ├── __init__.py
│   ├── abstractive_summarizer.py   ← BART model (Hugging Face)
│   ├── extractive_summarizer.py    ← Offline frequency-based method
│   └── utils.py                    ← Text cleaning, file I/O, saving
│
├── data/
│   └── sample_articles/            ← Ready-to-use .txt news articles
│       ├── ai_breakthrough.txt
│       ├── climate_report.txt
│       └── space_exploration.txt
│
├── outputs/                        ← Summaries auto-saved here
│
├── tests/
│   └── test_summarizers.py         ← pytest unit tests
│
├── notebooks/                      ← Jupyter notebooks (optional)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Techniques

| Method | Approach | Library |
|---|---|---|
| **Abstractive** | BART generates new sentences | Hugging Face Transformers |
| **Extractive** | Frequency scoring picks top sentences | Pure Python (offline) |

```
News Article
    │
    ├──▶ Extractive  →  score sentences by word freq  →  pick top-N
    │
    └──▶ Abstractive →  BART encoder-decoder          →  generate summary
```

---

## ⚙️ Setup

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

> The BART model (~1.6 GB) downloads automatically on the first run and is cached locally.

### 2 · Run

**Demo** — summarises all 3 sample articles:
```bash
python main.py
```

**Specific file:**
```bash
python main.py --file data/sample_articles/climate_report.txt
```

**Interactive** — paste any article in the terminal:
```bash
python main.py --interactive
```
Type your article, then type `END` on a new line to submit.

**Unit tests:**
```bash
python -m pytest tests/ -v
```

---

## 📤 Output

Each run saves a `.txt` file in `outputs/`:

```
outputs/
└── summary_Climate_Report_20250528_143012.txt
```

Contents:
```
ARTICLE   : Climate Report
GENERATED : 2025-05-28 14:30:12
============================================================

ABSTRACTIVE SUMMARY (BART model):
...

EXTRACTIVE SUMMARY (frequency-based):
...
```

---

## 🔑 Libraries

| Library | Purpose |
|---|---|
| `transformers` | BART model loading & inference |
| `torch` | PyTorch backend |
| `sentencepiece` | BART tokenizer dependency |
| `pytest` | Unit testing |
