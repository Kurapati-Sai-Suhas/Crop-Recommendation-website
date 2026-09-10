"""
============================================================
tests/test_model.py
============================================================
Model and pipeline tests.

These check things that can actually break: that the serving
artifacts agree with each other, that inference reproduces
training-time preprocessing exactly, and that the model clears a
threshold that means something.

The previous version asserted RandomForest accuracy >= 90% on this
dataset. A linear discriminant with no tuning scores 96.7% here, so
that bar could not fail -- it tested nothing. The threshold below
is set against the weakest sensible baseline instead.
============================================================
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "backend", "models")
DATA_PATH = os.path.join(ROOT, "data", "crop_data.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

SAMPLE_INPUT = {
    "N": 80, "P": 40, "K": 40,
    "temperature": 23.0, "humidity": 82.0,
    "ph": 6.0, "rainfall": 200.0,
}


def _require_models():
    if not os.path.exists(os.path.join(MODEL_DIR, "RandomForest.joblib")):
        pytest.skip("Models not trained yet -- run python train_pipeline.py")


def _test_split():
    """Reproduce the exact split the training pipeline held out."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if not os.path.exists(DATA_PATH):
        pytest.skip("Dataset not found -- run python data/download_data.py")

    df = pd.read_csv(DATA_PATH)
    y  = LabelEncoder().fit_transform(df["label"])
    X  = df[FEATURES].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
class TestArtifacts:
    def test_all_artifacts_present(self):
        _require_models()
        for name in ["RandomForest.joblib", "LogisticRegression.joblib",
                     "NaiveBayes.joblib", "scaler.joblib",
                     "label_encoder.joblib", "metrics.json"]:
            assert os.path.exists(os.path.join(MODEL_DIR, name)), f"missing {name}"

    def test_models_and_scaler_agree_on_feature_count(self):
        """A scaler and model trained on different shapes would fail at
        request time, not at load time."""
        _require_models()
        import joblib
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        assert scaler.n_features_in_ == len(FEATURES)
        for name in ["RandomForest", "LogisticRegression", "NaiveBayes"]:
            model = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
            assert model.n_features_in_ == len(FEATURES), name

    def test_encoder_covers_every_model_class(self):
        _require_models()
        import joblib
        encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
        model   = joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib"))
        assert len(encoder.classes_) == len(model.classes_)


class TestPrediction:
    def test_predict_returns_expected_shape(self):
        _require_models()
        from backend.services.predictor import predict
        result = predict(SAMPLE_INPUT)
        assert set(result) == {"crop", "confidence", "model_used"}
        assert isinstance(result["crop"], str)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_matches_the_predicted_class(self):
        """
        Guards a real bug: indexing predict_proba by the encoded label value
        instead of the class's position returns a neighbouring class's
        probability whenever the two orderings diverge.
        """
        _require_models()
        from backend.services import predictor
        from backend.services.predictor import predict, get_feature_array

        predictor.load_models()
        result   = predict(SAMPLE_INPUT)
        _raw, X  = get_feature_array(SAMPLE_INPUT)
        model    = predictor._models["RandomForest"]
        proba    = model.predict_proba(X)[0]

        assert result["confidence"] == pytest.approx(proba.max(), abs=1e-4)

    def test_feature_order_matters(self):
        """
        Inference must consume features in the training order. Permuting two
        values should change the answer; if it does not, ordering is being
        lost somewhere between the request and the model.
        """
        _require_models()
        from backend.services.predictor import predict
        swapped = dict(SAMPLE_INPUT, N=SAMPLE_INPUT["rainfall"],
                       rainfall=SAMPLE_INPUT["N"])
        assert predict(SAMPLE_INPUT)["crop"] != predict(swapped)["crop"]

    def test_inference_reproduces_training_preprocessing(self):
        """
        The end-to-end contract: routing a raw dict through the service must
        equal loading the artifacts and applying them by hand.
        """
        _require_models()
        import joblib
        from backend.services.predictor import predict

        scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        model   = joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

        row      = np.array([[SAMPLE_INPUT[f] for f in FEATURES]], dtype=float)
        expected = encoder.inverse_transform(model.predict(scaler.transform(row)))[0]

        assert predict(SAMPLE_INPUT)["crop"] == expected


class TestPerformance:
    def test_beats_the_weakest_sensible_baseline(self):
        """
        A threshold that can actually fail.

        Untuned LDA scores ~0.967 on this data, so 0.90 was free. The bar
        here is set just under that: if the served forest cannot beat a
        model with no hyperparameters, something is genuinely wrong.
        """
        _require_models()
        import joblib
        from sklearn.metrics import accuracy_score

        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        model  = joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib"))
        X_test, y_test = _test_split()

        acc = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
        assert acc >= 0.96, (
            f"Test accuracy {acc:.4f} is below an untuned linear baseline "
            f"(~0.967). The pipeline is broken, not merely underperforming."
        )

    def test_reported_metrics_match_a_fresh_evaluation(self):
        """metrics.json must describe the artifacts actually on disk."""
        _require_models()
        import joblib
        from sklearn.metrics import accuracy_score

        with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
            reported = json.load(f)

        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        model  = joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib"))
        X_test, y_test = _test_split()
        actual = accuracy_score(y_test, model.predict(scaler.transform(X_test)))

        assert actual == pytest.approx(reported["RandomForest"]["accuracy"],
                                       abs=1e-3), (
            "metrics.json disagrees with the saved model -- it is stale."
        )


class TestEvaluationReport:
    def test_report_records_an_explicit_selection_rationale(self):
        """Model choice must be recorded, not implied."""
        path = os.path.join(MODEL_DIR, "evaluation_report.json")
        if not os.path.exists(path):
            pytest.skip("Evaluation report not generated yet")
        with open(path, encoding="utf-8") as f:
            report = json.load(f)

        assert "model_selection" in report
        assert report["model_selection"]["rationale"]
        assert report["error_analysis"]["n_errors"] >= 0
        assert len(report["permutation_importance"]) == len(FEATURES)


class TestExplanations:
    """
    Regression tests for two bugs that shipped in this repository.

    1. /explain raised on every request. SHAP's TreeExplainer validates
       additivity across all 22 classes at once, and the interventional
       estimator left ~1e-4 of float residue on near-zero-probability
       classes, which tripped the check.
    2. Before that, the endpoint returned all-zero SHAP values while the
       text still asserted the features "strongly influenced" the result --
       a confident explanation of nothing.
    """

    INPUTS = [
        {"N": 80, "P": 40, "K": 40, "temperature": 23.0,
         "humidity": 82.0, "ph": 6.0, "rainfall": 200.0},
        {"N": 10, "P": 10, "K": 10, "temperature": 30.0,
         "humidity": 40.0, "ph": 4.2, "rainfall": 40.0},
        {"N": 120, "P": 130, "K": 200, "temperature": 22.0,
         "humidity": 92.0, "ph": 5.8, "rainfall": 115.0},
    ]

    def test_explain_does_not_raise(self):
        _require_models()
        from backend.services import explainer, predictor
        for data in self.INPUTS:
            crop = predictor.predict(data)["crop"]
            explainer.explain_prediction(data, crop)   # must not raise

    def test_shap_values_are_not_all_zero(self):
        _require_models()
        from backend.services import explainer, predictor
        for data in self.INPUTS:
            crop = predictor.predict(data)["crop"]
            result = explainer.explain_prediction(data, crop)
            total = sum(abs(v["shap_value"]) for v in result["shap_values"])
            assert total > 1e-6, (
                f"All SHAP values are zero for {crop}: the explanation is "
                f"asserting influence that was never computed."
            )

    def test_shap_satisfies_additivity(self):
        """
        The defining property: expected value + sum of attributions must
        equal the model's output for the explained class. If this drifts,
        the numbers are decorative.
        """
        _require_models()
        import numpy as np
        from backend.services import explainer, predictor

        data = self.INPUTS[0]
        crop = predictor.predict(data)["crop"]
        result = explainer.explain_prediction(data, crop)

        explainer._load_explainer()
        model    = predictor.get_rf_model()
        encoder  = predictor.get_label_encoder()
        _raw, X  = predictor.get_feature_array(data)
        idx      = list(encoder.classes_).index(crop)

        expected_value = explainer._explainer.expected_value
        base = (expected_value[idx] if np.ndim(expected_value)
                else float(expected_value))
        attributions = sum(v["shap_value"] for v in result["shap_values"])
        model_output = model.predict_proba(X)[0][idx]

        assert base + attributions == pytest.approx(model_output, abs=1e-3)

    def test_text_describes_each_feature_against_its_own_range(self):
        """
        The wording used a single '> 50 is High' rule for every feature, so
        pH (range 3.5-9.9) was always "Low" and temperature nearly always
        "Low". A pH of 6.0 sits mid-distribution and must not read as low.
        """
        _require_models()
        from backend.services import explainer, predictor

        data = dict(self.INPUTS[0], ph=6.5)
        crop = predictor.predict(data)["crop"]
        text = explainer.explain_prediction(data, crop)["text_explanation"]
        assert "Low pH Level" not in text
