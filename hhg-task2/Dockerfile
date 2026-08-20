# ============================================
# HHG Voice RAG — Production Dockerfile
# Multi-stage build for low-latency RAG
# ============================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Pre-cache HuggingFace SentenceTransformer model into image
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application source, scripts, and dataset
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Ensure numpy_store.pkl is built and resident in the image
RUN if [ ! -f data/numpy_store.pkl ]; then \
        echo "Building LocalNumpyStore index from dataset parquet..." && \
        python scripts/ingest.py --strategy fixed --max-records 10000 --fresh; \
    fi

# Create directories
RUN mkdir -p /app/docs

# Expose default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, httpx; port = os.environ.get('PORT', '8000'); r = httpx.get(f'http://localhost:{port}/health'); assert r.status_code == 200"

# Run the server with single worker and dynamic port binding
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
