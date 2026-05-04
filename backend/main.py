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
============================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api.routes import router
from backend.services.predictor import load_models


# ─── Lifespan context: runs on startup + shutdown ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML models into memory when the server starts.
    This avoids cold-start latency on the first request.
    """
    print("🚀 Starting Crop Recommendation API...")
    print("   Loading ML models...")
    load_models()
    print("   ✅ Ready to serve predictions!")
    yield
    print("👋 Shutting down...")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌾 Crop Recommendation API",
    description="""
A production-ready **Explainable AI** system for crop recommendation.

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

# ─── CORS middleware (allows frontend on any origin in dev) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register all routes ─────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ─── Root redirect to docs ───────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "🌾 Crop Recommendation API is running!",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }


# ─── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
