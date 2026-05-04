# -*- coding: utf-8 -*-
"""
============================================================
backend/services/predictor.py
============================================================
Inference service: loads trained models and runs predictions.
============================================================
"""

import os
import numpy as np
import pandas as pd
import joblib

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")

# Module-level cache (loaded once at startup)
_models  = {}
_scaler  = None
_encoder = None


def load_models():
    """Load all trained models + preprocessing objects into memory."""
    global _models, _scaler, _encoder

    if _models:
        return  # already loaded

    model_files = {
        "RandomForest":       "RandomForest.joblib",
        "LogisticRegression": "LogisticRegression.joblib",
        "NaiveBayes":         "NaiveBayes.joblib",
    }

    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            _models[name] = joblib.load(path)
        else:
            print(f"[WARN] Model not found: {path} -- run train_pipeline.py first")

    scaler_path  = os.path.join(MODEL_DIR, "scaler.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    if os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
    if os.path.exists(encoder_path):
        _encoder = joblib.load(encoder_path)

    print(f"[OK] Models loaded: {list(_models.keys())}")


def _prepare_input(data):
    """Convert a dict of feature values to a scaled numpy array."""
    row = [data[f] for f in FEATURES]
    X   = np.array(row).reshape(1, -1)
    if _scaler is not None:
        X = _scaler.transform(X)
    return X


def predict(data, model_name="RandomForest"):
    """
    Run prediction using the specified model (default: RandomForest).

    Args:
        data: dict with keys N, P, K, temperature, humidity, ph, rainfall
        model_name: one of 'RandomForest', 'LogisticRegression', 'NaiveBayes'

    Returns:
        {"crop": "rice", "confidence": 0.92, "model_used": "RandomForest"}
    """
    load_models()

    if model_name not in _models:
        raise ValueError(f"Model '{model_name}' not loaded. Available: {list(_models.keys())}")

    model    = _models[model_name]
    X        = _prepare_input(data)
    pred_idx = model.predict(X)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X)[0]
        confidence = round(float(proba[pred_idx]), 4)

    crop_name = _encoder.inverse_transform([pred_idx])[0] if _encoder else str(pred_idx)

    return {
        "crop":       crop_name,
        "confidence": confidence,
        "model_used": model_name,
    }


def get_all_model_predictions(data):
    """Run prediction with all 3 models and return results."""
    load_models()
    results = {}
    for name in ["RandomForest", "LogisticRegression", "NaiveBayes"]:
        if name in _models:
            results[name] = predict(data, model_name=name)
    return results


def get_feature_array(data):
    """Return (raw, scaled) feature arrays for SHAP."""
    raw    = np.array([data[f] for f in FEATURES]).reshape(1, -1)
    scaled = _scaler.transform(raw) if _scaler else raw
    return raw, scaled


def get_rf_model():
    """Return the RandomForest model (used by explainer service)."""
    load_models()
    return _models.get("RandomForest")


def get_label_encoder():
    """Return the LabelEncoder (used by explainer service)."""
    load_models()
    return _encoder
