# 📰 News Article Summarizer — ML/NLP Project

**Kartik Shrivastava** · Hex Softwares Internship · Project 1

---

## 🗂️ Project Structure

```
news_summarizer_ml/
│
├── main.py                          ← CLI entry point
├── app.py                           ← Flask web server
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── extractive_summarizer.py     ← Frequency-based NLP (offline)
│   └── utils.py                     ← Text cleaning, file I/O, saving
│
├── templates/
│   └── index.html                   ← Web UI
│
├── data/
│   └── sample_articles/
│       ├── ai_breakthrough.txt
│       ├── climate_report.txt
│       └── space_exploration.txt
│
├── outputs/                         ← Auto-saved summaries
│
└── tests/
    └── test_summarizers.py
```

---

## 🧠 How it works

Word-frequency extractive summarization — fully offline, no model download:

1. **Clean** — strip HTML, collapse whitespace
2. **Split** — tokenise into sentences
3. **Score** — build normalised word-frequency table, score each sentence
4. **Extract** — pick top-N sentences, return in original reading order

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Only `flask` and `pytest` are needed — no heavy ML libraries.

---

## ▶️ Run

**Web UI:**
```bash
python app.py
# Open: http://localhost:5000
```

**CLI demo** (all sample articles):
```bash
python main.py
```

**Specific file:**
```bash
python main.py --file data/sample_articles/climate_report.txt
```

**Interactive** (paste in terminal):
```bash
python main.py --interactive
```

**Tests:**
```bash
python -m pytest tests/ -v
```
