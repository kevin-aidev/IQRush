import os
from pathlib import Path

INPUT_PATH = os.getenv("INPUT_PATH", "Reviews.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output_predictions.csv")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "Text")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MODEL_NAME = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "false").lower() in ("1", "true", "yes")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
