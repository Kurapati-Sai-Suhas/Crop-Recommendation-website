# -*- coding: utf-8 -*-
"""
============================================================
train_pipeline.py  (Project Root)
============================================================
Master training pipeline script.

Run this ONCE to:
  1. Download the crop dataset (if not already present)
  2. Train all 3 models (RF, LR, NB)
  3. Save models to backend/models/
  4. Log everything to MLflow

Usage:
  cd crop-recommendation
  python train_pipeline.py
============================================================
"""

import os
import sys

# Add project root to Python path so relative imports work
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def ensure_dataset():
    """Download and verify the dataset if it is not already present."""
    data_path = os.path.join(ROOT, "data", "crop_data.csv")
    if os.path.exists(data_path):
        print(f"[DATA] Dataset found: {data_path}")
        return

    print("[DATA] Dataset not found. Downloading...")
    from data.download_data import fetch, verify, SOURCE_URL
    payload = fetch(SOURCE_URL)
    verify(payload)                      # checksum + structure, before writing
    with open(data_path, "wb") as handle:
        handle.write(payload)
    print(f"   [OK] Dataset written: {data_path}")


def run_training():
    """Run the full model training pipeline."""
    from backend.services.train import train_all_models
    print("\n" + "-" * 60)
    print("  TRAINING PIPELINE")
    print("-" * 60)
    metrics = train_all_models()
    return metrics


def verify_models():
    """Quick sanity check: load models and run a test prediction."""
    from backend.services.predictor import load_models, predict

    load_models()
    test_input = {
        "N": 80, "P": 40, "K": 40,
        "temperature": 23.0, "humidity": 82.0,
        "ph": 6.0, "rainfall": 200.0
    }
    result = predict(test_input)
    print(f"\n[OK] Verification prediction: {result['crop'].upper()} "
          f"(confidence: {result['confidence']:.2%})")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Crop Recommendation System -- Training Pipeline")
    print("=" * 60)

    # Step 1: Data
    ensure_dataset()

    # Step 2: Train
    metrics = run_training()

    # Step 3: Verify
    verify_models()

    print("\n" + "=" * 60)
    print("  [DONE] Pipeline complete! Start the API server with:")
    print("     uvicorn backend.main:app --reload --port 8000")
    print("=" * 60)
