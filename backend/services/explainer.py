# -*- coding: utf-8 -*-
"""
============================================================
backend/services/explainer.py
============================================================
Explainable AI service using SHAP (SHapley Additive exPlanations).

For each prediction it produces:
  1. SHAP values per feature (signed importance)
  2. A human-readable text explanation
  3. Bar chart data for the frontend (feature importance)

Why SHAP?
  - TreeExplainer is optimised for Random Forests
  - Provides both global and per-prediction explanations
  - Backed by Shapley values from game theory
============================================================
"""

import base64
import io
import logging
import os
import threading

import numpy as np
import pandas as pd
import shap
# The object-oriented Figure API is used below rather than pyplot: pyplot keeps
# a global figure registry that is not safe to touch from multiple threads, and
# request handlers now run in the threadpool.
from matplotlib.figure import Figure

from backend.errors import ExplanationFailed, ModelsUnavailable
from backend.services.predictor import (
    load_models, get_rf_model, get_label_encoder, get_feature_array
)

logger = logging.getLogger(__name__)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_LABELS = {
    "N":           "Nitrogen (N)",
    "P":           "Phosphorus (P)",
    "K":           "Potassium (K)",
    "temperature": "Temperature (C)",
    "humidity":    "Humidity (%)",
    "ph":          "pH Level",
    "rainfall":    "Rainfall (mm)",
}

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")

# Module-level SHAP explainer cache. Guarded by _lock: without it, concurrent
# first requests each built their own TreeExplainer over the same background set.
_explainer       = None
_background_data = None
_lock            = threading.RLock()


def _load_explainer():
    """Initialise the SHAP TreeExplainer once, thread-safely."""
    global _explainer, _background_data

    if _explainer is not None:
        return                                   # fast path, no lock

    with _lock:
        if _explainer is not None:               # re-check inside the lock
            return

        load_models()
        rf = get_rf_model()
        if rf is None:
            raise ModelsUnavailable(
                "RandomForest model not loaded; cannot build a SHAP explainer"
            )

        from backend.services import predictor as _predictor

        if os.path.exists(DATA_PATH):
            df     = pd.read_csv(DATA_PATH)
            sample = df[FEATURES].sample(100, random_state=42).values
            # The background set must live in the same feature space the model
            # was trained in, or the attributions are silently meaningless.
            if _predictor._scaler is None:
                raise ModelsUnavailable(
                    "Scaler not loaded; cannot place SHAP background data "
                    "in the model's feature space"
                )
            background = _predictor._scaler.transform(sample)
        else:
            logger.warning(
                "Training data not found at %s -- falling back to a zero "
                "background set. SHAP attributions will be less meaningful.",
                DATA_PATH,
            )
            background = np.zeros((10, len(FEATURES)))

        _background_data = background
        _explainer = shap.TreeExplainer(
            rf, data=background, feature_perturbation="interventional"
        )
        logger.info("SHAP TreeExplainer initialised")


def explain_prediction(data, predicted_crop):
    """
    Generate full XAI explanation for a single prediction.

    Args:
        data:            input feature dict (N, P, K, ...)
        predicted_crop:  the predicted crop label (string)

    Returns:
        {
          'shap_values':         [...],
          'text_explanation':    '...',
          'summary_plot_base64': '<base64>',
          'top_features':        [...]
        }
    """
    _load_explainer()

    raw, scaled = get_feature_array(data)
    # Use the scaled features: the explainer was initialised with a scaled
    # background, and the two must match.
    shap_vals = _explainer.shap_values(scaled)

    le = get_label_encoder()
    if le is None:
        raise ModelsUnavailable("Label encoder not loaded; cannot map SHAP classes")

    classes = list(le.classes_)
    if predicted_crop not in classes:
        raise ExplanationFailed(
            f"Predicted crop '{predicted_crop}' is not a known class"
        )
    crop_idx = classes.index(predicted_crop)

    # Extract SHAP values for predicted class
    if isinstance(shap_vals, list):
        class_shap = shap_vals[crop_idx][0]
    else:
        class_shap = shap_vals[0, :, crop_idx]

    feature_shap = []
    for i, feat in enumerate(FEATURES):
        feature_shap.append({
            "feature":     feat,
            "label":       FEATURE_LABELS[feat],
            "shap_value":  round(float(class_shap[i]), 4),
            "input_value": round(float(raw[0][i]), 2),
            "direction":   "positive" if class_shap[i] > 0 else "negative",
        })

    # Sort by absolute SHAP value
    feature_shap.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    text    = _build_text_explanation(feature_shap, predicted_crop)
    plot_b64 = _generate_shap_plot(feature_shap)

    top_features = [
        {"feature": fs["label"], "importance": round(abs(fs["shap_value"]), 4)}
        for fs in feature_shap
    ]

    return {
        "shap_values":         feature_shap,
        "text_explanation":    text,
        "summary_plot_base64": plot_b64,
        "top_features":        top_features,
    }


def _build_text_explanation(feature_shap, crop):
    """Build a human-readable explanation sentence."""
    if not feature_shap:
        return f"No feature attributions were available for {crop.capitalize()}."

    top3  = feature_shap[:3]
    parts = []
    for fs in top3:
        direction = "High" if fs["input_value"] > 50 else "Low"
        val  = fs["input_value"]
        unit = ""
        if fs["feature"] == "temperature":
            unit = "C"
        elif fs["feature"] == "humidity":
            unit = "%"
        elif fs["feature"] == "rainfall":
            unit = " mm"
        parts.append(f"{direction} {fs['label']} ({val}{unit})")

    if len(parts) >= 2:
        factors = ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        factors = parts[0] if parts else "input conditions"

    return (
        f"{factors} strongly influenced the recommendation of "
        f"**{crop.capitalize()}**. "
        f"The model is most sensitive to {feature_shap[0]['label'].lower()} "
        f"for this prediction."
    )


def _generate_shap_plot(feature_shap):
    """
    Generate a horizontal bar chart and return a base64-encoded PNG.

    Uses the object-oriented Figure API rather than pyplot: handlers now run
    in the threadpool, and pyplot's global figure registry would let
    concurrent renders corrupt each other's output.
    """
    try:
        labels = [fs["label"] for fs in feature_shap]
        values = [fs["shap_value"] for fs in feature_shap]
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in values]

        fig = Figure(figsize=(7, 4))
        ax  = fig.subplots()
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=9)
        ax.set_title("Feature Importance (SHAP)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        # A missing chart degrades the response; it must not fail the request.
        logger.warning("SHAP plot generation failed", exc_info=True)
        return ""


def get_global_feature_importance():
    """Return global RF feature importance (fast, no SHAP needed)."""
    load_models()
    rf = get_rf_model()
    if rf is None:
        raise ModelsUnavailable("RandomForest model not loaded")

    importances = rf.feature_importances_
    result = [
        {"feature": FEATURE_LABELS[f], "importance": round(float(v), 4)}
        for f, v in zip(FEATURES, importances)
    ]
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result
