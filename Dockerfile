# ============================================================
# Dockerfile — Backend (FastAPI)
# ============================================================
# Multi-stage build for the Python FastAPI backend.
# Base: python:3.12-slim
#
# Build: docker build -t crop-backend .
# Run:   docker run -p 8000:8000 crop-backend
# ============================================================

FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching: only re-install if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend/   ./backend/
COPY data/      ./data/
COPY mlops/     ./mlops/
COPY train_pipeline.py .

# Download the dataset and train models at build time
# (In production, mount pre-trained models as a volume instead)
RUN python data/download_data.py && \
    python train_pipeline.py

# Create logs directory
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Health check
# Uses Python, not curl: python:3.12-slim ships neither curl nor wget, so a
# curl-based probe always failed and the container never reported healthy.
# Probes /health/ready so an instance without usable models is taken out of
# rotation instead of serving unusable responses.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health/ready', timeout=5).status == 200 else 1)"]

# Run the FastAPI server.
#
# Multiple workers are required for real concurrency, not just cosmetic: the
# request handlers are synchronous so Starlette runs them in a threadpool, but
# SHAP tree traversal holds the GIL, so /explain still serialises within a
# single process. Each worker holds its own copy of the models (~100 MB), so
# raise CROP_WORKERS only if the container has the memory for it.
ENV CROP_WORKERS=2
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers ${CROP_WORKERS}"]
