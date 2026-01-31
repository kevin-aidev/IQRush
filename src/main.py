"""
CLI entrypoint: load CSV, run batch inference, export results, log metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog

from config import (
    BATCH_SIZE,
    DATA_DIR,
    INPUT_PATH,
    LOG_LEVEL,
    MAX_LENGTH,
    MODEL_NAME,
    OUTPUT_PATH,
    TEXT_COLUMN,
)
from src.inference import (
    InferenceMetrics,
    load_sentiment_pipeline,
    infer_dataframe,
)


def configure_logging(level: str = LOG_LEVEL) -> None:
    import logging
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    model_name: str = MODEL_NAME,
    no_header: bool = False,
) -> InferenceMetrics:
    """Run full pipeline: read CSV, batch inference, write output. Returns metrics."""
    log = structlog.get_logger()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        log.error("input_file_not_found", path=str(input_path))
        raise FileNotFoundError(str(input_path))

    metrics = InferenceMetrics()
    log.info("loading_model", model=model_name)
    pipe = load_sentiment_pipeline(model_name=model_name, max_length=max_length)
    log.info("reading_csv", path=str(input_path))
    df = pd.read_csv(input_path, header=None if no_header else 0)
    cols = list(df.columns)
    # When no_header, columns are 0,1,...,5; match "5" to column 5
    if text_column not in cols and text_column.isdigit():
        idx = int(text_column)
        if idx in cols:
            text_column = idx
    if text_column not in cols:
        text_column = cols[-1]
        log.info("text_column_inferred", column=str(text_column))
    df = infer_dataframe(df, text_column, pipe, metrics, batch_size=batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(
        "pipeline_complete",
        output=str(output_path),
        processed=metrics.total_processed,
        failed=metrics.total_failed,
        batches=metrics.batches_run,
    )
    return metrics


def main() -> int:
    configure_logging()
    log = structlog.get_logger()
    input_path = DATA_DIR / INPUT_PATH
    output_path = DATA_DIR / OUTPUT_PATH
    try:
        metrics = run_pipeline(
            input_path=input_path,
            output_path=output_path,
            text_column=TEXT_COLUMN,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            model_name=MODEL_NAME,
        )
        return 0
    except FileNotFoundError:
        return 1
    except Exception:
        log.exception("pipeline_failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
