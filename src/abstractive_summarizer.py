"""
abstractive_summarizer.py
─────────────────────────
Generates human-like summaries of news articles using
facebook/bart-large-cnn via Hugging Face Transformers.

Uses AutoTokenizer + AutoModelForSeq2SeqLM directly so it works
with both Transformers v4 and v5 (v5 removed the "summarization"
pipeline task string).
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class AbstractiveSummarizer:
    """
    Abstractive summarizer powered by BART
    (Bidirectional and Auto-Regressive Transformer).

    Generates new sentences that capture the core meaning of
    the input article — does not just copy existing sentences.
    """

    MODEL_NAME    = "facebook/bart-large-cnn"
    MAX_INPUT_LEN = 1024   # BART's hard token limit
    MAX_OUT_LEN   = 150
    MIN_OUT_LEN   = 40

    def __init__(self):
        print(f"  [Model] Loading: {self.MODEL_NAME}")
        print("  [Model] First run downloads ~1.6 GB. Please wait...\n")

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        self._model.eval()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        print(f"  [Model] Ready  (device: {self._device})\n")

    def summarize(
        self,
        text: str,
        max_length: int = MAX_OUT_LEN,
        min_length: int = MIN_OUT_LEN,
    ) -> str:
        """
        Produce an abstractive summary of the given text.

        Args:
            text       : Article text to summarise.
            max_length : Maximum tokens in the output.
            min_length : Minimum tokens in the output.

        Returns:
            Summary as a plain string.

        Raises:
            ValueError if text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        # Tokenise — truncate to model's hard limit
        inputs = self._tokenizer(
            text.strip(),
            return_tensors="pt",
            max_length=self.MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

        return self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
