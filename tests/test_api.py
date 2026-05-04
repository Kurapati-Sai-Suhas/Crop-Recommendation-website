"""
============================================================
tests/test_api.py
============================================================
API endpoint tests using FastAPI's TestClient.

Run: pytest tests/ -v
============================================================
"""

import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# ─── Sample valid input ───────────────────────────────────────────────────────
VALID_INPUT = {
    "N": 80, "P": 40, "K": 40,
    "temperature": 23.0, "humidity": 82.0,
    "ph": 6.0, "rainfall": 200.0
}

INVALID_INPUT = {
    "N": -10,        # invalid: below 0
    "P": 40,
    "K": 40,
    "temperature": 200.0,  # invalid: above 50
    "humidity": 82.0,
    "ph": 6.0,
    "rainfall": 200.0
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200


class TestPredictEndpoint:
    def test_predict_with_valid_input(self):
        response = client.post("/api/v1/predict", json=VALID_INPUT)
        assert response.status_code == 200

    def test_predict_returns_crop_name(self):
        response = client.post("/api/v1/predict", json=VALID_INPUT)
        data = response.json()
        assert "primary_prediction" in data
        assert "crop" in data["primary_prediction"]
        assert isinstance(data["primary_prediction"]["crop"], str)

    def test_predict_returns_confidence(self):
        response = client.post("/api/v1/predict", json=VALID_INPUT)
        data = response.json()
        confidence = data["primary_prediction"]["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_predict_returns_all_models(self):
        response = client.post("/api/v1/predict", json=VALID_INPUT)
        data = response.json()
        assert "all_models" in data
        # All 3 models should be present
        models = data["all_models"]
        assert "RandomForest" in models

    def test_predict_invalid_input_returns_422(self):
        response = client.post("/api/v1/predict", json=INVALID_INPUT)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self):
        incomplete = {"N": 80, "P": 40}  # missing 5 fields
        response = client.post("/api/v1/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_crop_is_known_class(self):
        KNOWN_CROPS = {
            "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
            "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
            "banana", "mango", "grapes", "watermelon", "muskmelon",
            "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"
        }
        response = client.post("/api/v1/predict", json=VALID_INPUT)
        crop = response.json()["primary_prediction"]["crop"]
        assert crop in KNOWN_CROPS


class TestMetricsEndpoint:
    def test_metrics_returns_200_if_trained(self):
        # This test only passes after training
        import os
        metrics_path = os.path.join(ROOT, "backend", "models", "metrics.json")
        if os.path.exists(metrics_path):
            response = client.get("/api/v1/metrics")
            assert response.status_code == 200

    def test_metrics_structure(self):
        import os
        metrics_path = os.path.join(ROOT, "backend", "models", "metrics.json")
        if os.path.exists(metrics_path):
            response = client.get("/api/v1/metrics")
            data = response.json()
            assert "models" in data
            for model_name, metrics in data["models"].items():
                assert "accuracy" in metrics
                assert "f1_score" in metrics


class TestFeatureImportanceEndpoint:
    def test_feature_importance_returns_200(self):
        response = client.get("/api/v1/feature-importance")
        assert response.status_code == 200

    def test_feature_importance_has_7_features(self):
        response = client.get("/api/v1/feature-importance")
        data = response.json()
        assert "feature_importance" in data
        assert len(data["feature_importance"]) == 7
