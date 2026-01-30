"""
Batch sentiment inference using a pre-trained Hugging Face model.
Tracks metrics: processed count, failed count, batches run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Pipeline, pipeline

from config import BATCH_SIZE, MAX_LENGTH, MODEL_NAME


@dataclass
class InferenceMetrics:
    """Counters exposed for logging and /metrics."""

    total_processed: int = 0
    total_failed: int = 0
    batches_run: int = 0

    def to_prometheus(self) -> str:
        lines = [
            "# HELP inference_processed_total Number of rows successfully inferred",
            "# TYPE inference_processed_total counter",
            f"inference_processed_total {self.total_processed}",
            "# HELP inference_failed_total Number of rows that failed",
            "# TYPE inference_failed_total counter",
            f"inference_failed_total {self.total_failed}",
            "# HELP inference_batches_total Number of batches executed",
            "# TYPE inference_batches_total counter",
            f"inference_batches_total {self.batches_run}",
        ]
        return "\n".join(lines)


def _sanitize_text(raw: Any) -> str:
    """Coerce cell to string and strip; empty -> placeholder to avoid model errors."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    return s if s else ""


def _truncate_and_clean(text: str, max_chars: int = 512) -> str:
    """Truncate and collapse whitespace for model input."""
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars].strip() or ""


def load_sentiment_pipeline(
    model_name: str = MODEL_NAME,
    device: int | str | None = None,
    max_length: int = MAX_LENGTH,
) -> Pipeline:
    """Load Hugging Face sentiment classification pipeline."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=max_length,
        return_all_scores=False,
    )


def run_batch_inference(
    pipe: Pipeline,
    texts: list[str],
    metrics: InferenceMetrics,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    """
    Run inference on a list of texts in batches. Updates metrics in place.
    Returns list of results (label + score) per input; failed rows get a sentinel result.
    """
    results: list[dict[str, Any]] = []
    # Truncate/clean and avoid empty strings for the model (use "." as placeholder)
    cleaned = [_truncate_and_clean(_sanitize_text(t)) or "." for t in texts]
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        metrics.batches_run += 1
        try:
            out = pipe(batch)
            # pipe returns one dict per item: {"label": "POSITIVE/NEGATIVE", "score": float}
            for j, item in enumerate(out):
                metrics.total_processed += 1
                results.append(item)
        except Exception:
            for _ in batch:
                metrics.total_failed += 1
                results.append({"label": "ERROR", "score": 0.0})
    return results


def infer_dataframe(
    df: pd.DataFrame,
    text_column: str,
    pipe: Pipeline,
    metrics: InferenceMetrics,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame. Expects a column `text_column`.
    Adds columns: sentiment_label, sentiment_score.
    """
    texts = df[text_column].astype(object).fillna("").tolist()
    results = run_batch_inference(pipe, texts, metrics, batch_size=batch_size)
    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    out = df.copy()
    out["sentiment_label"] = labels
    out["sentiment_score"] = scores
    return out


def read_csv_in_chunks(
    path: str,
    text_column: str,
    chunksize: int = 5000,
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield (chunk DataFrame, chunk_index) for memory-efficient processing."""
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        if text_column not in chunk.columns:
            # Sentiment140: columns are sentiment, id, date, query, user, text (last)
            cols = chunk.columns.tolist()
            if cols:
                text_column = cols[-1]
        yield chunk, i
