"""Tests for batch inference and metrics."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.inference import (
    InferenceMetrics,
    _sanitize_text,
    _truncate_and_clean,
    run_batch_inference,
)


def test_sanitize_text():
    assert _sanitize_text("hello") == "hello"
    assert _sanitize_text("  spaced  ") == "spaced"
    assert _sanitize_text(None) == ""
    assert _sanitize_text("") == ""
    assert _sanitize_text(123) == "123"


def test_truncate_and_clean():
    assert _truncate_and_clean("a  b   c") == "a b c"
    assert _truncate_and_clean("x" * 600)[:512] == "x" * 512
    assert _truncate_and_clean("") == ""


def test_inference_metrics_prometheus():
    m = InferenceMetrics(total_processed=10, total_failed=1, batches_run=2)
    out = m.to_prometheus()
    assert "inference_processed_total 10" in out
    assert "inference_failed_total 1" in out
    assert "inference_batches_total 2" in out


def test_run_batch_inference_mock():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {"label": "POSITIVE", "score": 0.99},
        {"label": "NEGATIVE", "score": 0.98},
    ]
    metrics = InferenceMetrics()
    results = run_batch_inference(mock_pipe, ["hello", "world"], metrics, batch_size=2)
    assert len(results) == 2
    assert results[0]["label"] == "POSITIVE"
    assert results[1]["label"] == "NEGATIVE"
    assert metrics.total_processed == 2
    assert metrics.total_failed == 0
    assert metrics.batches_run == 1


def test_run_batch_inference_failure():
    mock_pipe = MagicMock(side_effect=RuntimeError("model error"))
    metrics = InferenceMetrics()
    results = run_batch_inference(mock_pipe, ["a", "b"], metrics, batch_size=2)
    assert len(results) == 2
    assert all(r["label"] == "ERROR" and r["score"] == 0.0 for r in results)
    assert metrics.total_failed == 2
    assert metrics.batches_run == 1
