"""
============================================================
backend/api/schemas.py
============================================================
Pydantic models for input validation and API response shapes.

Pydantic automatically validates:
  - Data types (float, str, etc.)
  - Value ranges (ge=, le= constraints)
  - Required vs optional fields

If validation fails, FastAPI returns a 422 Unprocessable Entity
with clear error messages — no manual checks needed.
============================================================
"""

from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CropInput(BaseModel):
    """
    Input feature vector for crop recommendation.
    All 7 features are required with realistic agronomic bounds.

    `allow_inf_nan=False` rejects NaN/Infinity, which JSON permits and which
    previously slipped past the range checks and crashed response encoding.
    """
    N:           float = Field(..., ge=0,   le=200,  allow_inf_nan=False, description="Nitrogen content in soil (0–200)")
    P:           float = Field(..., ge=0,   le=200,  allow_inf_nan=False, description="Phosphorus content in soil (0–200)")
    K:           float = Field(..., ge=0,   le=210,  allow_inf_nan=False, description="Potassium content in soil (0–210)")
    temperature: float = Field(..., ge=0,   le=50,   allow_inf_nan=False, description="Temperature in Celsius (0–50)")
    humidity:    float = Field(..., ge=0,   le=100,  allow_inf_nan=False, description="Relative humidity % (0–100)")
    ph:          float = Field(..., ge=0.0, le=14.0, allow_inf_nan=False, description="Soil pH value (0–14)")
    rainfall:    float = Field(..., ge=0,   le=500,  allow_inf_nan=False, description="Annual rainfall in mm (0–500)")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "N": 80,
                "P": 40,
                "K": 40,
                "temperature": 23.5,
                "humidity": 82.0,
                "ph": 6.0,
                "rainfall": 200.0
            }
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SingleModelResult(BaseModel):
    """Prediction result from one model."""
    # `model_used` collides with Pydantic's protected `model_` namespace;
    # opting out keeps the field name and silences the warning on import.
    model_config = ConfigDict(protected_namespaces=())

    crop:       str
    confidence: float
    model_used: str


class FeatureImportanceItem(BaseModel):
    """One feature's SHAP importance."""
    feature:     str
    label:       str
    shap_value:  float
    input_value: float
    direction:   str   # "positive" or "negative"


class PredictResponse(BaseModel):
    """Full response from POST /predict."""
    primary_prediction: SingleModelResult
    all_models:         Dict[str, SingleModelResult]
    input_summary:      Dict[str, float]


class ExplainResponse(BaseModel):
    """Full response from POST /explain."""
    crop:                  str
    shap_values:           List[FeatureImportanceItem]
    text_explanation:      str
    summary_plot_base64:   str
    top_features:          List[dict]


class ModelMetric(BaseModel):
    """Evaluation metrics for one model."""
    accuracy:  float
    precision: float
    recall:    float
    f1_score:  float


class MetricsResponse(BaseModel):
    """Response from GET /metrics."""
    models: Dict[str, ModelMetric]


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:  str
    models_loaded: List[str]
    version: str = "1.0.0"
