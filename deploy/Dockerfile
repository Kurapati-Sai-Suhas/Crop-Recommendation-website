# ============================================================
# deploy/Dockerfile — single-container deployment
# ============================================================
# One image serving both the React UI and the FastAPI API on one
# port, which is what free container hosts (Hugging Face Spaces,
# Cloud Run, Render) expect: one process, one port, no compose.
#
# Two deliberate differences from the root Dockerfile, which is for
# local development:
#
#   1. It does NOT train at build time. The model artifacts are
#      committed, so training here would make every deploy depend on
#      a network fetch of the dataset and add minutes to the build --
#      and, worse, could silently ship a model nobody evaluated.
#      Deploys should ship a reviewed artifact, not produce a new one.
#
#   2. It does NOT copy data/. Since the explainer moved to
#      tree_path_dependent SHAP it needs no background sample, so
#      nothing at serving time reads the dataset.
#
# Build:  docker build -f deploy/Dockerfile -t crop-app .
# Run:    docker run -p 7860:7860 crop-app
# ============================================================

# ─── Stage 1: build the React bundle ─────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build

# package files first, so a source-only change reuses the install layer
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# No VITE_API_URL: the bundle then calls the API same-origin, which is
# correct here because this image serves both.
RUN npm run build


# ─── Stage 2: Python runtime ─────────────────────────────────
FROM python:3.12-slim

# Hosts set PORT; 7860 is the Hugging Face Spaces default.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# gcc/g++ are needed to build some wheels; removed afterwards so they
# do not ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++

# Application code and the trained artifacts
COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# Run as a non-root user. Hugging Face Spaces expects uid 1000 and a
# writable home; the app also needs to write prediction logs.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Readiness, not liveness: this returns non-200 when the model artifacts
# are missing, so a broken image is never routed traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen( \
'http://localhost:'+os.environ.get('PORT','7860')+'/api/v1/health/ready', \
timeout=5).status==200 else 1)"

# Single worker by default: free tiers give ~1-2 vCPU, and SHAP holds the
# GIL, so extra workers mostly add memory. Raise WEB_CONCURRENCY on a
# larger instance.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
