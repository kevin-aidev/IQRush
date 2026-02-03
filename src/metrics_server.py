"""
Optional HTTP server exposing Prometheus-style /metrics (processed, failed, batches).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn

if TYPE_CHECKING:
    from src.inference import InferenceMetrics

_app = FastAPI()
_metrics: InferenceMetrics | None = None


def set_metrics(m: InferenceMetrics) -> None:
    global _metrics
    _metrics = m


@_app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    if _metrics is None:
        return "# No metrics yet\n"
    return _metrics.to_prometheus()


def run_metrics_server(metrics: InferenceMetrics, host: str = "0.0.0.0", port: int = 9090) -> None:
    """Run /metrics server in the current process (blocks)."""
    set_metrics(metrics)
    uvicorn.run(_app, host=host, port=port, log_level="warning")
