# -*- coding: utf-8 -*-
"""
============================================================
backend/services/train.py
============================================================
Training pipeline for Crop Recommendation System.

What this does:
  1. Loads the crop dataset from CSV
  2. Preprocesses data (label encoding + feature scaling)
  3. Trains 3 models: Random Forest, Logistic Regression, Naive Bayes
  4. Evaluates all models (accuracy, precision, recall, F1)
  5. Saves models + scaler with joblib
  6. Logs everything to MLflow

Run directly: python -m backend.services.train
============================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crop_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "backend", "models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns and target ─────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"


def load_data():
    """Load and return features and labels from CSV."""
    print(f"[DATA] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    X = df[FEATURES]
    y = df[TARGET]
    print(f"   Loaded {len(df)} rows | {y.nunique()} crops")
    return X, y


def preprocess(X, y, test_size=0.2):
    """
    Preprocess data:
    - Encode labels to integers
    - Scale features with StandardScaler
    - Split into train/test sets
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print("   [OK] Scaler + LabelEncoder saved")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def evaluate_model(model, X_test, y_test, le):
    """Compute accuracy, precision, recall, and F1 score."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }


def train_all_models():
    """
    Main training function.
    Trains RF, LR, NB -> saves models -> logs to MLflow -> returns metrics.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler, le = preprocess(X, y)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "NaiveBayes": GaussianNB(),
    }

    all_metrics = {}

    # MLflow experiment
    mlruns_dir = os.path.join(BASE_DIR, "mlops", "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")
    mlflow.set_experiment("CropRecommendation")

    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, le)
            all_metrics[name] = metrics

            print(f"   Accuracy : {metrics['accuracy']:.4f}")
            print(f"   F1 Score : {metrics['f1_score']:.4f}")

            if name == "RandomForest":
                mlflow.log_params({"n_estimators": 100, "max_depth": "None", "random_state": 42})
            elif name == "LogisticRegression":
                mlflow.log_params({"max_iter": 1000, "multi_class": "ovr"})
            elif name == "NaiveBayes":
                mlflow.log_params({"var_smoothing": 1e-9})

            mlflow.log_metrics(metrics)

            model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
            joblib.dump(model, model_path)

            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=name,
                    registered_model_name=f"CropRecommender_{name}",
                )
            except Exception:
                pass  # Registry not always available locally

            print(f"   [OK] {name} saved -> {model_path}")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] Metrics saved -> {METRICS_PATH}")

    return all_metrics


if __name__ == "__main__":
    print("=" * 60)
    print("  Crop Recommendation -- Model Training Pipeline")
    print("=" * 60)
    results = train_all_models()
    print("\n[RESULTS]")
    for model_name, m in results.items():
        print(f"  {model_name:20s} | Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f}")
    print("\n[DONE] Training complete!")
