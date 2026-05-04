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

# Generate dataset and train models at build time
# (In production, mount pre-trained models as a volume instead)
RUN python data/generate_data.py && \
    python train_pipeline.py

# Create logs directory
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
