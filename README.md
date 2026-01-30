# IQRush.ai – Batch Sentiment Inference Pipeline

A batch inference pipeline that reads a CSV of text inputs, runs a pre-trained sentiment model in batches, and writes predictions to an output CSV. Containerized with Docker; includes logging, simple metrics (processed/failed), and an optional `/metrics` HTTP endpoint.

## Deliverables

- **Repository**: This repo (or ZIP) with source, Dockerfile, and config.
- **Dockerfile** (and `docker-compose.yml`): See [Docker](#docker) and [Usage](#usage).
- **Output predictions file**: Produced by running the pipeline (e.g. `output_predictions.csv`). `sample_output_predictions.csv` shows the expected format (input columns + `sentiment_label`, `sentiment_score`).
- **README**: This file (setup, usage, architecture).
- **Optional**: `/metrics` endpoint, CI workflow (`.github/workflows/ci.yml`), test suite (`tests/`).

## Features

- **Batch inference** using a Hugging Face pre-trained model (`distilbert-base-uncased-finetuned-sst-2-english` by default)
- **Docker** image and `docker-compose` for one-command runs
- **Logging** (structlog) with structured fields (processed, failed, batches)
- **Metrics**: number processed, number failed, batches run; optional Prometheus-style `GET /metrics`
- **Output**: full input CSV plus `sentiment_label` and `sentiment_score` columns

## Setup

### Local (no Docker)

1. **Python 3.10+** and a virtualenv:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. **Data**: Place your input CSV in the project directory (or set `INPUT_PATH` / use `--input`). Example: `Reviews.csv` or a subset of `training.csv`.

### Docker

1. **Build** the image:

    ```bash
    docker build -t iqrush-pipeline .
    ```

2. **Data directory**: Create a folder (e.g. `data/`) and put your input CSV there (e.g. `data/Reviews.csv`).

## Usage

### Local

```bash
# Default: read Reviews.csv, write output_predictions.csv
python run_pipeline.py

# Custom paths and text column
python run_pipeline.py --input training_subset.csv --output out.csv --text-column "tweet text"

# Sentiment140 (no header): use last column (index 5) for text
python run_pipeline.py --input training_20k.csv --output out.csv --no-header --text-column 5

# Larger batches, optional metrics server (then open http://localhost:9090/metrics)
python run_pipeline.py --batch-size 64 --metrics --metrics-port 9090
```

**Environment variables** (optional): `INPUT_PATH`, `OUTPUT_PATH`, `TEXT_COLUMN`, `BATCH_SIZE`, `MAX_LENGTH`, `SENTIMENT_MODEL`, `LOG_LEVEL`, `DATA_DIR`, `METRICS_ENABLED`, `METRICS_PORT`.

### Docker

**Bind-mount data directory** (recommended):

```bash
mkdir -p data
cp Reviews.csv data/
docker run --rm -v "$(pwd)/data:/data" iqrush-pipeline
# Output: data/output_predictions.csv
```

**Override input/output**:

```bash
docker run --rm -v "$(pwd)/data:/data" iqrush-pipeline \
  --input /data/Reviews.csv --output /data/output_predictions.csv
```

**Docker Compose**:

```bash
cp Reviews.csv data/
docker compose run --rm pipeline
```

### Datasets

- **Reviews.csv**: Product reviews; text column is `Text`. Good for quick runs (~3k rows).
- **training.csv** (Sentiment140): Tweets; if the file has no header, the last column is used as text. Use a subset (e.g. 20k rows) for testing:

    ```bash
    head -n 20001 training.csv > data/training_20k.csv
    python run_pipeline.py --input data/training_20k.csv --output data/out_20k.csv --no-header --text-column 5
    ```

    For CSV with headers, use the column name that contains the tweet text (often the 6th column name).

## Output

- **File**: Same as input CSV with two extra columns:
    - `sentiment_label`: `POSITIVE` or `NEGATIVE` (or `ERROR` on failure)
    - `sentiment_score`: confidence in [0, 1]
- **Logs**: Structured log line at the end with `processed`, `failed`, `batches`.
- **Optional metrics**: With `--metrics`, a server listens on port 9090; `GET /metrics` returns Prometheus-style counters: `inference_processed_total`, `inference_failed_total`, `inference_batches_total`.

## Architecture

- **config.py**: Central config from env (paths, batch size, model name, etc.).
- **src/inference.py**: Loads the Hugging Face pipeline, runs batch inference, tracks `InferenceMetrics` (processed, failed, batches), and adds sentiment columns to a DataFrame.
- **src/main.py**: Sets up logging, loads CSV, calls inference, writes CSV, logs final metrics.
- **run_pipeline.py**: CLI (argparse) that calls `run_pipeline` and optionally starts the metrics server.
- **src/metrics_server.py**: Optional FastAPI app exposing `GET /metrics` in Prometheus format.
- **Dockerfile**: Multi-stage not required; single stage with `pip install`, `WORKDIR /app`, default `DATA_DIR=/data` and `ENTRYPOINT`/`CMD` for `run_pipeline.py`.

**Design choices**:

- **Batching**: Configurable `BATCH_SIZE` (default 32) to trade off memory and throughput.
- **Model**: DistilBERT SST-2 for speed and CPU use; override with `SENTIMENT_MODEL` or `--model`.
- **Metrics**: In-process counters plus optional HTTP endpoint to avoid extra processes.
- **Text column**: Configurable; if missing, the last column is used (e.g. Sentiment140).

## Optional: Metrics, CI, Tests

- **Metrics**: Run with `--metrics` (and optionally `--metrics-port 9090`); scrape `http://localhost:9090/metrics`.
- **CI**: See `.github/workflows/ci.yml` for a minimal lint/test workflow.
- **Tests**: `pytest tests/`; see `tests/test_inference.py` for inference and metrics tests.

## License

Internal assessment – IQRush.ai.
