"""
============================================================
backend/api/routes.py
============================================================
FastAPI router with all API endpoints.

Endpoints:
  GET  /health           → liveness: the process is up
  GET  /health/ready     → readiness: models loaded and servable
  POST /predict          → predict crop from 7 input features
  POST /explain          → get SHAP explanation for a prediction
  GET  /metrics          → get model evaluation metrics
  GET  /monitoring       → get recent predictions + drift report
  GET  /feature-importance → global RF feature importance

Handlers are declared `def`, not `async def`. Their bodies are
blocking, CPU-bound scikit-learn and SHAP work; declaring them
async ran that work directly on the event loop and serialised
every request in the process.
============================================================
"""

import json
import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError

from backend.api.schemas import (
    CropInput, PredictResponse, ExplainResponse,
    MetricsResponse, HealthResponse, SingleModelResult, ModelMetric
)
from backend.errors import ExplanationFailed, ModelsUnavailable
from backend.services import predictor, explainer, monitoring

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Path to saved metrics ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(BASE_DIR, "backend", "models", "metrics.json")


def _request_id(request: Request) -> str:
    """Correlation ID attached by the middleware in backend/main.py."""
    return getattr(request.state, "request_id", "-")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — liveness
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Liveness check: 200 whenever the process is serving.

    Deliberately does NOT gate on model availability — use /health/ready for
    that. An orchestrator should restart on liveness failure but merely stop
    routing traffic on readiness failure; those are different signals.
    """
    return HealthResponse(
        status="ok" if predictor.is_ready() else "no_models",
        models_loaded=predictor.loaded_model_names(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health/ready — readiness
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health/ready", response_model=HealthResponse, tags=["System"])
def readiness_check(response: Response):
    """
    Readiness check: 200 only when a real prediction can be served.

    Returns 503 when artifacts are missing, so load balancers and container
    healthchecks stop sending traffic to an instance that cannot answer.
    """
    ready = predictor.is_ready()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "no_models",
        models_loaded=predictor.loaded_model_names(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_crop(input_data: CropInput, request: Request):
    """
    Predict the best crop based on soil and environmental conditions.

    Uses Random Forest as the primary model. Also returns predictions from
    Logistic Regression and Naive Bayes for the model comparison panel.

    **Response**
    - `primary_prediction`: RF result with confidence
    - `all_models`: results from every loaded model
    - `input_summary`: echo of input for display

    Returns 503 if models are unavailable — never a placeholder crop.
    """
    start_time = time.time()
    data = input_data.model_dump()

    try:
        all_results_raw = predictor.get_all_model_predictions(data)
    except ModelsUnavailable:
        logger.error("prediction refused: models unavailable",
                     extra={"request_id": _request_id(request)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is unavailable. No recommendation can be made.",
        )
    except Exception:
        rid = _request_id(request)
        logger.exception("prediction failed", extra={"request_id": rid})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed. Reference: {rid}",
        )

    all_results = {
        name: SingleModelResult(**res)
        for name, res in all_results_raw.items()
    }
    primary = all_results[predictor.PRIMARY_MODEL]

    latency_ms = (time.time() - start_time) * 1000
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


# ─────────────────────────────────────────────────────────────────────────────
# POST /explain
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_prediction(input_data: CropInput, request: Request):
    """
    Generate a SHAP explanation for the crop prediction.

    Returns:
    - Per-feature SHAP values (positive = pushed toward this crop)
    - Human-readable text explanation
    - Base64-encoded SHAP bar chart image
    - Sorted feature importance list for frontend charts
    """
    data = input_data.model_dump()

    try:
        predicted_crop = predictor.predict(data)["crop"]
        explanation    = explainer.explain_prediction(data, predicted_crop)
    except ModelsUnavailable:
        logger.error("explanation refused: models unavailable",
                     extra={"request_id": _request_id(request)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is unavailable. No explanation can be produced.",
        )
    except ExplanationFailed:
        rid = _request_id(request)
        logger.exception("explanation failed", extra={"request_id": rid})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Explanation could not be produced. Reference: {rid}",
        )
    except Exception:
        rid = _request_id(request)
        logger.exception("explanation failed", extra={"request_id": rid})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed. Reference: {rid}",
        )

    return ExplainResponse(crop=predicted_crop, **explanation)


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/metrics", response_model=MetricsResponse, tags=["MLOps"])
def get_metrics(request: Request):
    """
    Return model evaluation metrics for every trained model.

    Metrics are computed on the test split during training and stored in
    backend/models/metrics.json.
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Run the training pipeline first: python train_pipeline.py",
        )

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            raw_metrics = json.load(f)
        models = {name: ModelMetric(**vals) for name, vals in raw_metrics.items()}
    except (json.JSONDecodeError, TypeError, ValidationError):
        # A metrics file missing precision/recall previously escaped as an
        # unhandled 500 from response-model validation.
        rid = _request_id(request)
        logger.exception("metrics file is malformed: %s", METRICS_PATH,
                         extra={"request_id": rid})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Stored metrics are malformed or incomplete. "
                    f"Re-run the training pipeline. Reference: {rid}"),
        )

    return MetricsResponse(models=models)


# ─────────────────────────────────────────────────────────────────────────────
# GET /monitoring
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/monitoring", tags=["MLOps"])
def get_monitoring():
    """
    Return recent prediction logs and basic data drift indicators.
    Useful for post-deployment monitoring in an MLOps dashboard.
    """
    return {
        "recent_predictions": monitoring.get_recent_logs(n=50),
        "drift_report":       monitoring.compute_drift_report(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /feature-importance
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/feature-importance", tags=["Explainability"])
def get_global_feature_importance(request: Request):
    """
    Return global feature importance from the RandomForest model.
    This is the built-in RF importance (not SHAP) — fast, no input needed.
    """
    try:
        return {"feature_importance": explainer.get_global_feature_importance()}
    except ModelsUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is unavailable.",
        )
    except Exception:
        rid = _request_id(request)
        logger.exception("feature importance failed", extra={"request_id": rid})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature importance failed. Reference: {rid}",
        )
