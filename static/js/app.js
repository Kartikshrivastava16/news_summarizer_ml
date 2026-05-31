/**
 * NewsML — Client-Side Application Logic
 * static/js/app.js
 *
 * Responsibilities:
 *  - Word-frequency extractive summarization (offline JS fallback)
 *  - Flask /summarize API integration
 *  - UI state: mode toggle, slider, char/word counter, progress bar
 *  - Result rendering, copy-to-clipboard, toast notifications
 *  - Form reset and error handling
 */

"use strict";

/* ══════════════════════════════════════════════════════════════════════
   Constants
   ══════════════════════════════════════════════════════════════════════ */

const STOPWORDS = new Set([
  "the","a","an","is","it","in","on","at","to","for","of","and","or","but",
  "with","that","this","was","are","be","been","by","from","as","its","into",
  "than","then","so","if","about","has","have","had","not","which","who",
  "they","their","he","she","we","you","i","my","our","your","his","her",
  "also","more","can","will","would","could","should","may","might","shall",
  "do","did","does","just","very","much","many","some","such","when","where",
  "after","before","while","since","because","though","although"
]);

const MODE_HINTS = {
  extractive:  "Offline · frequency scoring · instant",
  abstractive: "HuggingFace BART · requires transformers install · ~1.6 GB first run",
  both:        "Runs extractive first (instant), then BART (requires transformers)"
};

const MAX_CHARS   = 6000;
const MIN_CHARS   = 80;

/* ══════════════════════════════════════════════════════════════════════
   App State
   ══════════════════════════════════════════════════════════════════════ */

let currentMode          = "extractive";
let isFlask              = false;
let abstractiveAvailable = false;

/* ══════════════════════════════════════════════════════════════════════
   Flask Detection
   ══════════════════════════════════════════════════════════════════════ */

/**
 * Ping the Flask /model-status route.
 * Sets isFlask and abstractiveAvailable accordingly.
 * Updates the header status pill.
 */
async function detectFlask() {
  try {
    const res = await fetch("/model-status", {
      signal: AbortSignal.timeout(1500)
    });
    if (res.ok) {
      isFlask = true;
      const data = await res.json();
      abstractiveAvailable = data.abstractive_available;

      const pill = document.getElementById("abs-status-pill");
      if (pill) {
        pill.textContent = abstractiveAvailable
          ? "BART · Ready"
          : "BART · Install needed";
      }
    }
  } catch {
    isFlask = false;
  }
}

/* ══════════════════════════════════════════════════════════════════════
   Offline JS Extractive Summarizer
   (mirrors the Python ExtractiveSummarizer algorithm)
   ══════════════════════════════════════════════════════════════════════ */

/**
 * Build a normalised word-frequency map from text, stop-words excluded.
 * @param {string} text
 * @returns {Object.<string, number>}
 */
function buildFrequencyMap(text) {
  const words = (text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [])
    .filter(w => !STOPWORDS.has(w));

  const freq = {};
  let maxF = 1;
  for (const w of words) {
    freq[w] = (freq[w] || 0) + 1;
    if (freq[w] > maxF) maxF = freq[w];
  }
  for (const k in freq) freq[k] /= maxF;
  return freq;
}

/**
 * Score a single sentence against the frequency map.
 * @param {string} sentence
 * @param {Object} freq
 * @returns {number}
 */
function scoreSentence(sentence, freq) {
  const tokens = (sentence.toLowerCase().match(/\b[a-z]{3,}\b/g) || []);
  return tokens.reduce((sum, t) => sum + (!STOPWORDS.has(t) ? (freq[t] || 0) : 0), 0);
}

/**
 * Offline extractive summarizer — frequency-based sentence scoring.
 * @param {string} text  Cleaned article text.
 * @param {number} n     Number of sentences to extract.
 * @returns {string}     Top-N sentences in original reading order.
 */
function jsSummarize(text, n) {
  // Split into sentences on . ! ? boundaries
  const sentences = (text.match(/[^.!?]+[.!?]+/g) || [])
    .map(s => s.trim())
    .filter(s => s.length > 25);

  if (!sentences.length) return text.trim();
  if (sentences.length <= n) return sentences.join(" ");

  const freq   = buildFrequencyMap(text);
  const scored = sentences.map((s, i) => ({ s, i, score: scoreSentence(s, freq) }));

  const topIdx = new Set(
    [...scored]
      .sort((a, b) => b.score - a.score)
      .slice(0, n)
      .map(x => x.i)
  );

  // Return in original reading order
  return scored.filter(x => topIdx.has(x.i)).map(x => x.s).join(" ");
}

/* ══════════════════════════════════════════════════════════════════════
   UI Helpers
   ══════════════════════════════════════════════════════════════════════ */

/** Count words in a string. */
function countWords(text) {
  return (text.trim().match(/\S+/g) || []).length;
}

/** Update the character and word counters below the textarea. */
function updateCharCount() {
  const text = document.getElementById("article-text").value;
  const cc   = document.getElementById("char-count");
  cc.textContent = `${text.length} / ${MAX_CHARS}`;
  cc.classList.toggle("warn", text.length > MAX_CHARS - 500);
  document.getElementById("word-count").textContent = `${countWords(text)} words`;
}

/** Update the displayed value next to the sentence slider. */
function updateSlider(value) {
  document.getElementById("slider-val").textContent = value;
}

/** Set the progress bar width (0–100). */
function setProgress(pct) {
  const bar = document.getElementById("progress-bar");
  if (bar) bar.style.width = pct + "%";
}

/** Show an error banner with a message. */
function showError(message) {
  const banner = document.getElementById("error-banner");
  document.getElementById("error-msg").textContent = message;
  banner.style.display = "flex";
}

/** Hide the error banner. */
function hideError() {
  document.getElementById("error-banner").style.display = "none";
}

/**
 * Show a toast notification.
 * @param {string} message
 * @param {number} [duration=3200]  ms before auto-hide.
 */
function showToast(message, duration = 3200) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = "✓  " + message;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), duration);
}

/** Format a byte count as a human-readable string. */
function formatBytes(bytes) {
  if (bytes < 1024)    return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

/** Debounce wrapper. */
function debounce(fn, delay = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/* ══════════════════════════════════════════════════════════════════════
   Mode Toggle
   ══════════════════════════════════════════════════════════════════════ */

/**
 * Switch the active summarization mode.
 * @param {"extractive"|"abstractive"|"both"} mode
 */
function setMode(mode) {
  currentMode = mode;

  // Update button active state
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  // Update hint text
  const hint = document.getElementById("mode-hint");
  if (hint) hint.textContent = MODE_HINTS[mode] || "";

  // Show sentence slider only for extractive / both
  const slider = document.getElementById("slider-section");
  if (slider) slider.style.display = (mode === "abstractive") ? "none" : "";
}

/* ══════════════════════════════════════════════════════════════════════
   Result Panel Rendering
   ══════════════════════════════════════════════════════════════════════ */

/**
 * Populate and show a result panel.
 * @param {string} panelId   Element id of the outer panel.
 * @param {string} textId    Element id of the result text container.
 * @param {string} metaId    Element id of the meta line.
 * @param {string} statsId   Element id of the stats row.
 * @param {string} text      Summary text to display.
 * @param {string} label     Label for the meta line.
 */
function showResultPanel(panelId, textId, metaId, statsId, text, label) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  document.getElementById(textId).textContent = text || "(no output)";
  document.getElementById(metaId).textContent =
    `${new Date().toLocaleTimeString()}  ·  ${label}`;
  document.getElementById(statsId).innerHTML =
    `<span>${countWords(text)} words</span><span>${text.length} chars</span>`;

  panel.style.display = "block";
}

/* ══════════════════════════════════════════════════════════════════════
   Copy to Clipboard
   ══════════════════════════════════════════════════════════════════════ */

/**
 * Copy the text content of an element to the clipboard.
 * @param {string} textId   Id of element whose text to copy.
 * @param {HTMLElement} btn Button element (its label is temporarily changed).
 */
function copyResult(textId, btn) {
  const text = document.getElementById(textId)?.textContent || "";
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = original; }, 2000);
  }).catch(() => {
    showToast("Copy failed — please select and copy manually.");
  });
}

/* ══════════════════════════════════════════════════════════════════════
   Form Reset
   ══════════════════════════════════════════════════════════════════════ */

function resetForm() {
  const titleEl = document.getElementById("article-title");
  const textEl  = document.getElementById("article-text");
  const slider  = document.getElementById("num-slider");
  const sliderV = document.getElementById("slider-val");
  const cc      = document.getElementById("char-count");
  const wc      = document.getElementById("word-count");
  const output  = document.getElementById("output-section");

  if (titleEl) titleEl.value = "";
  if (textEl)  textEl.value  = "";
  if (slider)  slider.value  = "3";
  if (sliderV) sliderV.textContent = "3";
  if (cc) { cc.textContent = `0 / ${MAX_CHARS}`; cc.classList.remove("warn"); }
  if (wc)  wc.textContent  = "0 words";
  if (output)  output.style.display = "none";

  ["ext-panel", "abs-panel", "abs-unavailable"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  hideError();
}

/* ══════════════════════════════════════════════════════════════════════
   Main Submit Handler
   ══════════════════════════════════════════════════════════════════════ */

async function handleSubmit() {
  hideError();

  const text  = (document.getElementById("article-text")?.value || "").trim();
  const title = (document.getElementById("article-title")?.value || "").trim() || "Untitled Article";
  const nSent = parseInt(document.getElementById("num-slider")?.value || "3", 10);
  const mode  = currentMode;

  // ── Validation ────────────────────────────────────────────────────
  if (!text) {
    showError("Please paste your article text.");
    return;
  }
  if (text.length < MIN_CHARS) {
    showError(`Article too short — provide at least ${MIN_CHARS} characters.`);
    return;
  }

  // ── Lock UI ───────────────────────────────────────────────────────
  const btn = document.getElementById("submit-btn");
  const lbl = document.getElementById("btn-label");
  if (btn) btn.disabled = true;
  if (lbl) lbl.innerHTML = 'Processing <span class="spinner"></span>';
  setProgress(20);

  // Hide stale results
  ["ext-panel", "abs-panel", "abs-unavailable"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  try {
    if (isFlask) {
      /* ── Flask API Mode ─────────────────────────────────────────── */
      setProgress(50);

      const res = await fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, title, num_sentences: nSent, mode })
      });

      setProgress(90);
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Server error. Please try again.");
        return;
      }

      if (data.extractive && (mode === "extractive" || mode === "both")) {
        showResultPanel(
          "ext-panel", "out-extractive", "ext-meta", "ext-stats",
          data.extractive,
          `"${title}" · ${nSent} sentence${nSent !== 1 ? "s" : ""}`
        );
      }

      if (mode === "abstractive" || mode === "both") {
        if (data.abstractive) {
          showResultPanel(
            "abs-panel", "out-abstractive", "abs-meta", "abs-stats",
            data.abstractive,
            `"${title}" · BART`
          );
        } else {
          const warn = document.getElementById("abs-unavailable");
          if (warn) warn.style.display = "block";
        }
      }

      showToast("Saved to outputs/");

    } else {
      /* ── Offline / Browser-Only Mode ────────────────────────────── */
      setProgress(70);

      if (mode === "abstractive") {
        // Can't run BART in the browser without Flask
        const warn = document.getElementById("abs-unavailable");
        if (warn) warn.style.display = "block";

      } else {
        const result = jsSummarize(text, nSent);
        setProgress(95);
        showResultPanel(
          "ext-panel", "out-extractive", "ext-meta", "ext-stats",
          result,
          `"${title}" · ${nSent} sentence${nSent !== 1 ? "s" : ""} · offline`
        );

        if (mode === "both") {
          const warn = document.getElementById("abs-unavailable");
          if (warn) warn.style.display = "block";
        }
      }

      showToast("Summary ready (offline mode)");
    }

    // ── Show output section & scroll to it ───────────────────────
    const outputSection = document.getElementById("output-section");
    if (outputSection) {
      outputSection.style.display = "block";
      setTimeout(() => {
        outputSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    }

  } catch (err) {
    showError("Unexpected error: " + err.message);
  } finally {
    setProgress(100);
    setTimeout(() => setProgress(0), 400);
    if (btn) btn.disabled = false;
    if (lbl) lbl.textContent = "Summarize";
  }
}

/* ══════════════════════════════════════════════════════════════════════
   Boot
   ══════════════════════════════════════════════════════════════════════ */

// Detect Flask on page load
detectFlask();
