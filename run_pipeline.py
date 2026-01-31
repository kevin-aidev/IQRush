#!/usr/bin/env python3
"""Run the batch inference pipeline. Usage: python run_pipeline.py [--input path] [--output path]."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BATCH_SIZE,
    DATA_DIR,
    INPUT_PATH,
    LOG_LEVEL,
    MAX_LENGTH,
    METRICS_ENABLED,
    METRICS_PORT,
    MODEL_NAME,
    OUTPUT_PATH,
    TEXT_COLUMN,
)
from src.main import configure_logging, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch sentiment inference pipeline")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / INPUT_PATH), help="Input CSV path")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / OUTPUT_PATH), help="Output CSV path")
    parser.add_argument("--text-column", "-t", default=TEXT_COLUMN, help="Column name for text")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Max token length")
    parser.add_argument("--model", "-m", default=MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--metrics", action="store_true", default=METRICS_ENABLED, help="Expose /metrics HTTP server")
    parser.add_argument("--metrics-port", type=int, default=METRICS_PORT, help="Metrics server port")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row (e.g. Sentiment140)")
    args = parser.parse_args()

    configure_logging(LOG_LEVEL)
    try:
        metrics = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_name=args.model,
            no_header=args.no_header,
        )
        if args.metrics:
            from src.metrics_server import run_metrics_server
            run_metrics_server(metrics, port=args.metrics_port)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
