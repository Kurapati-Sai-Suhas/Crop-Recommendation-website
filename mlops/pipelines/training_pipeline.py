"""
============================================================
mlops/pipelines/training_pipeline.py
============================================================
Modular training pipeline with separate stages.

Each stage can be run independently, making the pipeline
easy to debug and extend.

Usage:
  python mlops/pipelines/training_pipeline.py --stage all
  python mlops/pipelines/training_pipeline.py --stage preprocess
  python mlops/pipelines/training_pipeline.py --stage train
  python mlops/pipelines/training_pipeline.py --stage evaluate
============================================================
"""

import argparse
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

FEATURES   = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET     = "label"
DATA_PATH  = os.path.join(ROOT, "data", "crop_data.csv")
MODEL_DIR  = os.path.join(ROOT, "backend", "models")


def stage_preprocess() -> tuple:
    """
    Stage 1: Load data and split into train/test sets.
    Returns X_train, X_test, y_train, y_test, scaler, le
    """
    print("\n[Stage 1] Preprocessing data...")
    df = pd.read_csv(DATA_PATH).dropna()

    le     = LabelEncoder()
    y      = le.fit_transform(df[TARGET])
    X      = df[FEATURES].values
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print(f"   Train: {len(X_train)} | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, scaler, le


def stage_train(X_train, y_train) -> dict:
    """Stage 2: Train all models."""
    print("\n[Stage 2] Training models...")

    models = {
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "NaiveBayes":         GaussianNB(),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, path)
        trained[name] = model
        print(f"   ✅ {name} trained + saved")

    return trained


def stage_evaluate(trained_models: dict, X_test, y_test) -> dict:
    """Stage 3: Evaluate all models and save metrics."""
    print("\n[Stage 3] Evaluating models...")

    metrics = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
        print(f"   {name:20s} → Acc: {metrics[name]['accuracy']:.4f} | F1: {metrics[name]['f1_score']:.4f}")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   📊 Metrics → {metrics_path}")

    return metrics


def run_full_pipeline():
    """Run all pipeline stages end-to-end."""
    X_tr, X_te, y_tr, y_te, scaler, le = stage_preprocess()
    trained = stage_train(X_tr, y_tr)
    metrics = stage_evaluate(trained, X_te, y_te)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Recommendation Training Pipeline")
    parser.add_argument("--stage", choices=["all", "preprocess", "train", "evaluate"],
                        default="all", help="Which stage to run")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MLOps Training Pipeline — Stage: {args.stage.upper()}")
    print("=" * 60)

    if args.stage == "all":
        run_full_pipeline()
    elif args.stage == "preprocess":
        stage_preprocess()
    elif args.stage in ("train", "evaluate"):
        X_tr, X_te, y_tr, y_te, _, _ = stage_preprocess()
        trained = stage_train(X_tr, y_tr)
        if args.stage == "evaluate":
            stage_evaluate(trained, X_te, y_te)

    print("\n✅ Done!")
