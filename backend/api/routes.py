"""
============================================================
backend/api/routes.py
============================================================
FastAPI router with all API endpoints.

Endpoints:
  GET  /health    → check if server + models are ready
  POST /predict   → predict crop from 7 input features
  POST /explain   → get SHAP explanation for a prediction
  GET  /metrics   → get model evaluation metrics
  GET  /monitoring → get recent predictions + drift report
============================================================
"""

import json
import os
import time

from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    CropInput, PredictResponse, ExplainResponse,
    MetricsResponse, HealthResponse, SingleModelResult, ModelMetric
)
from backend.services import predictor, explainer, monitoring

router = APIRouter()

# ─── Path to saved metrics ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(BASE_DIR, "backend", "models", "metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK with list of loaded models.
    Useful for Docker health checks and monitoring.
    """
    predictor.load_models()
    loaded = list(predictor._models.keys())
    return HealthResponse(
        status="ok" if loaded else "no_models",
        models_loaded=loaded,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_crop(input_data: CropInput):
    """
    Predict the best crop based on soil and environmental conditions.

    Uses Random Forest as the primary model.
    Also returns predictions from Logistic Regression and Naive Bayes
    for the model comparison panel in the UI.

    **Request body**: 7 feature values (N, P, K, temperature, humidity, ph, rainfall)

    **Response**:
    - `primary_prediction`: RF result with confidence
    - `all_models`: results from all 3 models
    - `input_summary`: echo of input for display
    """
    start_time = time.time()

    try:
        data = input_data.model_dump()

        # Run prediction on all 3 models
        all_results_raw = predictor.get_all_model_predictions(data)

        # Build response-model-compatible dicts
        all_results = {
            name: SingleModelResult(**res)
            for name, res in all_results_raw.items()
        }

        primary = all_results.get(
            "RandomForest",
            SingleModelResult(crop="unknown", confidence=0.0, model_used="RandomForest")
        )

        latency_ms = (time.time() - start_time) * 1000

        # Log prediction for monitoring
        monitoring.log_prediction(
            input_data=data,
            prediction=primary.model_dump(),
            latency_ms=latency_ms,
        )

        return PredictResponse(
            primary_prediction=primary,
            all_models=all_results,
            input_summary={k: round(v, 2) for k, v in data.items()},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /explain
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_prediction(input_data: CropInput):
    """
    Generate SHAP explanation for the crop prediction.

    Returns:
    - Per-feature SHAP values (positive = pushed toward this crop)
    - Human-readable text explanation
    - Base64-encoded SHAP bar chart image
    - Sorted feature importance list for frontend charts
    """
    try:
        data = input_data.model_dump()

        # First get the prediction
        prediction_result = predictor.predict(data, model_name="RandomForest")
        predicted_crop    = prediction_result["crop"]

        # Generate SHAP explanation
        explanation = explainer.explain_prediction(data, predicted_crop)

        return ExplainResponse(
            crop=predicted_crop,
            **explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/metrics", response_model=MetricsResponse, tags=["MLOps"])
async def get_metrics():
    """
    Return model evaluation metrics for all 3 trained models.

    Metrics are computed on the test split during training
    and stored in backend/models/metrics.json.
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run the training pipeline first: python train_pipeline.py"
        )

    with open(METRICS_PATH, "r") as f:
        raw_metrics = json.load(f)

    return MetricsResponse(
        models={name: ModelMetric(**vals) for name, vals in raw_metrics.items()}
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /monitoring
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/monitoring", tags=["MLOps"])
async def get_monitoring():
    """
    Return recent prediction logs and basic data drift indicators.
    Useful for post-deployment monitoring in an MLOps dashboard.
    """
    recent_logs  = monitoring.get_recent_logs(n=50)
    drift_report = monitoring.compute_drift_report()

    return {
        "recent_predictions": recent_logs,
        "drift_report":       drift_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /feature-importance
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/feature-importance", tags=["Explainability"])
async def get_global_feature_importance():
    """
    Return global feature importance from the RandomForest model.
    This is the built-in RF importance (not SHAP) — fast, no input needed.
    """
    try:
        importance = explainer.get_global_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
