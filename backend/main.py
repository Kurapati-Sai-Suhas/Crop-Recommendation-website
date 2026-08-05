"""
============================================================
backend/main.py
============================================================
FastAPI application entry point.

Starts the server with:
  uvicorn backend.main:app --reload --port 8000

API docs available at:
  http://localhost:8000/docs    (Swagger UI)
  http://localhost:8000/redoc  (ReDoc)

Environment variables:
  CROP_CORS_ORIGINS  comma-separated allowlist of browser origins
                     (default: local Vite dev servers)
  CROP_LOG_LEVEL     logging level (default: INFO)
============================================================
"""

import logging
import math
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.services.predictor import is_ready, load_models

# ─── Logging ──────────────────────────────────────────────────────────────────
class _RequestIdFilter(logging.Filter):
    """Supply a default request_id so the format string never raises."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


logging.basicConfig(
    level=os.getenv("CROP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_RequestIdFilter())

logger = logging.getLogger("crop.api")


# ─── CORS allowlist ───────────────────────────────────────────────────────────
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _cors_origins() -> list[str]:
    """
    Read the browser origin allowlist from the environment.

    Never falls back to "*": a wildcard combined with credentialed requests
    makes Starlette reflect whatever Origin the caller sends, which lets any
    site read authenticated responses.
    """
    raw = os.getenv("CROP_CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise ValueError(
            "CROP_CORS_ORIGINS must not contain '*'. List the exact origins "
            "that are allowed to call this API."
        )
    return origins or DEFAULT_CORS_ORIGINS


# ─── Lifespan context: runs on startup + shutdown ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML models into memory when the server starts.

    Loading eagerly avoids cold-start latency and surfaces a broken
    deployment in the startup logs rather than in a user's first request.
    """
    logger.info("Starting Crop Recommendation API...")
    load_models()
    if is_ready():
        logger.info("Models loaded — ready to serve predictions")
    else:
        # Not fatal: the process still serves /health so an operator can see
        # what is wrong. Prediction endpoints return 503 until this is fixed.
        logger.error(
            "Startup completed WITHOUT usable models. Prediction endpoints "
            "will return 503. Run: python train_pipeline.py"
        )
    yield
    logger.info("Shutting down")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌾 Crop Recommendation API",
    description="""
An **Explainable AI** system for crop recommendation.

> **Status: academic prototype.** Models are trained on *synthetic* data.
> Reported accuracy measures how well each model recovers that generator,
> not agronomic accuracy. Not for real planting decisions.

## Features
- **Predict** the best crop based on soil and environmental data
- **Explain** predictions using SHAP (SHapley Additive exPlanations)
- **Compare** 3 ML models: Random Forest, Logistic Regression, Naive Bayes
- **Monitor** prediction logs and data drift

## Input Features
| Feature | Unit | Range |
|---------|------|-------|
| N | ratio | 0–200 |
| P | ratio | 0–200 |
| K | ratio | 0–210 |
| temperature | °C | 0–50 |
| humidity | % | 0–100 |
| ph | 0–14 | 0–14 |
| rainfall | mm | 0–500 |
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,      # flip on only alongside a real auth design
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


# ─── Request correlation IDs ──────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Attach a correlation ID to every request.

    Error responses quote this ID instead of internal exception text, so an
    operator can find the matching log line without the client learning
    anything about the internals.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Validation errors ────────────────────────────────────────────────────────
def _json_safe(value):
    """
    Replace non-finite floats with their text form, recursively.

    FastAPI echoes the offending input back in a 422 body. When that input is
    NaN or Infinity, json.dumps refuses to encode it and the 422 becomes an
    opaque 500 — reachable by any caller with a 4-byte payload.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": _json_safe(exc.errors())},
    )


# ─── Unhandled errors ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the detail; return only a correlation ID."""
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("unhandled error", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error. Reference: {request_id}"},
    )


# ─── Register all routes ─────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ─── Root ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "🌾 Crop Recommendation API is running!",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "notice":  "Academic prototype — trained on synthetic data. "
                   "Not for real planting decisions.",
    }


# ─── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
