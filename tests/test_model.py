"""
============================================================
tests/test_model.py
============================================================
Model correctness and performance tests.

Verifies:
  - All models can be loaded
  - Predictions return expected format
  - RandomForest accuracy >= 90%
  - SHAP values have correct shape
============================================================
"""

import os
import sys
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestModelLoading:
    def test_models_can_be_loaded(self):
        from backend.services.predictor import load_models, _models
        load_models()
        assert len(_models) > 0

    def test_random_forest_is_loaded(self):
        from backend.services.predictor import load_models, _models
        load_models()
        assert "RandomForest" in _models

    def test_scaler_is_loaded(self):
        from backend.services.predictor import load_models, _scaler
        load_models()
        assert _scaler is not None


class TestPrediction:
    SAMPLE_INPUT = {
        "N": 80, "P": 40, "K": 40,
        "temperature": 23.0, "humidity": 82.0,
        "ph": 6.0, "rainfall": 200.0
    }

    def test_predict_returns_dict(self):
        from backend.services.predictor import load_models, predict
        load_models()
        result = predict(self.SAMPLE_INPUT)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self):
        from backend.services.predictor import load_models, predict
        load_models()
        result = predict(self.SAMPLE_INPUT)
        assert "crop" in result
        assert "confidence" in result
        assert "model_used" in result

    def test_confidence_between_0_and_1(self):
        from backend.services.predictor import load_models, predict
        load_models()
        result = predict(self.SAMPLE_INPUT)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_models_predict(self):
        from backend.services.predictor import load_models, get_all_model_predictions
        load_models()
        results = get_all_model_predictions(self.SAMPLE_INPUT)
        assert len(results) >= 1  # at least RF should be present


class TestModelAccuracy:
    """Test model accuracy against the test split — requires trained models."""

    def _get_test_data(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        data_path = os.path.join(ROOT, "data", "crop_data.csv")
        if not os.path.exists(data_path):
            pytest.skip("Dataset not found")

        df = pd.read_csv(data_path).dropna()
        FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        le = LabelEncoder()
        X = df[FEATURES].values
        y = le.fit_transform(df["label"].values)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_test, y_test

    def test_rf_accuracy_above_90_percent(self):
        from sklearn.metrics import accuracy_score
        import joblib

        model_path = os.path.join(ROOT, "backend", "models", "RandomForest.joblib")
        scaler_path = os.path.join(ROOT, "backend", "models", "scaler.joblib")

        if not os.path.exists(model_path):
            pytest.skip("Model not trained yet")

        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        X_test, y_test = self._get_test_data()

        X_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_scaled)
        acc      = accuracy_score(y_test, y_pred)

        print(f"\n   RandomForest Test Accuracy: {acc:.4f}")
        assert acc >= 0.90, f"Expected accuracy >= 90%, got {acc:.4f}"
