# -*- coding: utf-8 -*-
"""
============================================================
backend/services/predictor.py
============================================================
Inference service: loads trained models and runs predictions.

Loading happens once, under a lock, and is published only when
the whole bundle (models + scaler + encoder) is consistent.
A half-loaded cache used to be observable by concurrent
requests, which silently skipped feature scaling and produced
confidently wrong crops.
============================================================
"""

import logging
import os
import threading

import numpy as np
import joblib

from backend.errors import ModelsUnavailable

logger = logging.getLogger(__name__)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")

MODEL_FILES = {
    "RandomForest":       "RandomForest.joblib",
    "LogisticRegression": "LogisticRegression.joblib",
    "NaiveBayes":         "NaiveBayes.joblib",
}

PRIMARY_MODEL = "RandomForest"

# ─── Module-level cache ───────────────────────────────────────────────────────
# _models is mutated in place (never rebound) so that callers holding a
# reference to it keep observing the live object.
_models  = {}
_scaler  = None
_encoder = None

# Set last, inside the lock, once _models/_scaler/_encoder are all consistent.
# Readers block on the lock until this flips, so a partially populated cache
# is never observable.
_loaded = False
_lock   = threading.RLock()


def load_models(force: bool = False):
    """
    Load all trained models + preprocessing objects into memory.

    Idempotent and thread-safe: concurrent callers block until the first
    load finishes rather than each repeating the work.
    """
    global _loaded, _scaler, _encoder

    if _loaded and not force:
        return                                   # fast path, no lock

    with _lock:
        if _loaded and not force:                # re-check inside the lock
            return

        models, missing = {}, []
        for name, filename in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                models[name] = joblib.load(path)
            else:
                missing.append(filename)

        scaler_path  = os.path.join(MODEL_DIR, "scaler.joblib")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

        scaler  = joblib.load(scaler_path)  if os.path.exists(scaler_path)  else None
        encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

        # Publish everything together, then flip the flag.
        _models.clear()
        _models.update(models)
        _scaler  = scaler
        _encoder = encoder
        _loaded  = True

        if missing:
            logger.error(
                "Model artifacts missing from %s: %s -- run train_pipeline.py. "
                "Prediction endpoints will return 503.",
                MODEL_DIR, ", ".join(missing),
            )
        if models and scaler is None:
            logger.error(
                "scaler.joblib missing -- models were trained on scaled features. "
                "Refusing to serve unscaled input."
            )
        logger.info("Models loaded: %s", sorted(_models))


def is_ready() -> bool:
    """True when the service can actually serve a prediction."""
    load_models()
    return PRIMARY_MODEL in _models and _scaler is not None and _encoder is not None


def loaded_model_names() -> list:
    """Names of the models currently held in memory."""
    load_models()
    return sorted(_models)


def _require_ready():
    """Raise rather than let a caller fall back to a placeholder result."""
    load_models()
    if PRIMARY_MODEL not in _models:
        raise ModelsUnavailable(
            f"Primary model '{PRIMARY_MODEL}' is not loaded "
            f"(available: {sorted(_models)})"
        )
    if _scaler is None or _encoder is None:
        raise ModelsUnavailable(
            "Preprocessing artifacts (scaler / label encoder) are not loaded"
        )


def _prepare_input(data):
    """Convert a dict of feature values to a scaled numpy array."""
    row = [data[f] for f in FEATURES]
    X   = np.array(row, dtype=float).reshape(1, -1)
    if _scaler is None:
        # Reachable only if _require_ready() was bypassed. Scaling silently
        # is how wrong-but-plausible predictions used to escape.
        raise ModelsUnavailable("Scaler not loaded; cannot prepare input")
    return _scaler.transform(X)


def predict(data, model_name=PRIMARY_MODEL):
    """
    Run prediction using the specified model (default: RandomForest).

    Args:
        data: dict with keys N, P, K, temperature, humidity, ph, rainfall
        model_name: one of 'RandomForest', 'LogisticRegression', 'NaiveBayes'

    Returns:
        {"crop": "rice", "confidence": 0.92, "model_used": "RandomForest"}

    Raises:
        ModelsUnavailable: if the requested model or preprocessing is missing.
    """
    _require_ready()

    if model_name not in _models:
        raise ModelsUnavailable(
            f"Model '{model_name}' is not loaded (available: {sorted(_models)})"
        )

    model    = _models[model_name]
    X        = _prepare_input(data)
    pred_idx = model.predict(X)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X)[0]
        confidence = round(float(proba[pred_idx]), 4)

    return {
        "crop":       str(_encoder.inverse_transform([pred_idx])[0]),
        "confidence": confidence,
        "model_used": model_name,
    }


def get_all_model_predictions(data):
    """Run prediction with every loaded model and return the results."""
    _require_ready()
    return {
        name: predict(data, model_name=name)
        for name in MODEL_FILES
        if name in _models
    }


def get_feature_array(data):
    """Return (raw, scaled) feature arrays for SHAP."""
    _require_ready()
    raw = np.array([data[f] for f in FEATURES], dtype=float).reshape(1, -1)
    return raw, _scaler.transform(raw)


def get_rf_model():
    """Return the RandomForest model (used by explainer service)."""
    load_models()
    return _models.get(PRIMARY_MODEL)


def get_label_encoder():
    """Return the LabelEncoder (used by explainer service)."""
    load_models()
    return _encoder
