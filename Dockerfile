# IQRush batch sentiment inference pipeline
# Python 3.11 slim, CPU-friendly (no CUDA)
FROM python:3.11-slim

WORKDIR /app

# Install system deps only if needed (e.g. for some tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install in a layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY config.py run_pipeline.py ./
COPY src/ ./src/

# Default: mount data at /data, read Reviews.csv, write output_predictions.csv
ENV DATA_DIR=/data
ENV INPUT_PATH=Reviews.csv
ENV OUTPUT_PATH=output_predictions.csv
ENV BATCH_SIZE=32
ENV LOG_LEVEL=INFO

# Optional: expose metrics server port
EXPOSE 9090

# Run pipeline: expects /data/Reviews.csv (or INPUT_PATH) via volume
# Example: docker run -v $(pwd)/data:/data iqrush-pipeline
ENTRYPOINT ["python", "-u", "run_pipeline.py"]
CMD ["--input", "/data/Reviews.csv", "--output", "/data/output_predictions.csv"]
