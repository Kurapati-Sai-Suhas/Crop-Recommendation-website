"""
============================================================
backend/services/monitoring.py
============================================================
Prediction monitoring service.

Logs every prediction to a JSONL file with:
  - timestamp
  - input features
  - prediction result
  - confidence
  - response time

Also computes basic drift indicators by comparing recent
predictions against training distribution statistics.
============================================================
"""

import os
import json
import time
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
from collections import Counter

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR     = os.path.join(BASE_DIR, "logs")
LOG_PATH    = os.path.join(LOG_DIR, "predictions.jsonl")
DRIFT_PATH  = os.path.join(LOG_DIR, "drift_report.json")

os.makedirs(LOG_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def log_prediction(input_data: dict, prediction: dict, latency_ms: float) -> None:
    """
    Append a prediction record to the JSONL log file.

    Args:
        input_data:  dict of feature values
        prediction:  dict with crop, confidence, model_used
        latency_ms:  response time in milliseconds
    """
    record = {
        "timestamp":   datetime.datetime.now(timezone.utc).isoformat(),
        "input":       input_data,
        "prediction":  prediction,
        "latency_ms":  round(latency_ms, 2),
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_recent_logs(n: int = 50) -> list:
    """Return the last n prediction records."""
    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass

    return list(reversed(records))


def compute_drift_report() -> dict:
    """
    Basic data drift detection:
    Compare mean of recent 50 predictions against training distribution.
    Returns a dict with per-feature drift indicators.
    """
    logs = get_recent_logs(50)
    if not logs:
        return {"status": "no_data", "features": {}}

    # Recent inputs
    recent_vals = {f: [] for f in FEATURES}
    for log in logs:
        for f in FEATURES:
            if f in log.get("input", {}):
                recent_vals[f].append(log["input"][f])

    # Training stats (hardcoded from known dataset distributions)
    # These would typically come from the training data statistics saved during training
    training_means = {
        "N": 50.6, "P": 53.4, "K": 48.1,
        "temperature": 25.6, "humidity": 71.5,
        "ph": 6.47, "rainfall": 103.5
    }
    training_stds = {
        "N": 36.9, "P": 32.9, "K": 50.7,
        "temperature": 5.0, "humidity": 22.3,
        "ph": 0.77, "rainfall": 54.9
    }

    drift_results = {}
    for f in FEATURES:
        vals = recent_vals[f]
        if not vals:
            drift_results[f] = {"status": "no_data"}
            continue

        recent_mean = np.mean(vals)
        train_mean  = training_means[f]
        train_std   = training_stds[f]

        # Simple z-score drift check
        z_score = abs(recent_mean - train_mean) / (train_std + 1e-10)
        drift_level = "high" if z_score > 2 else ("medium" if z_score > 1 else "low")

        drift_results[f] = {
            "recent_mean": round(recent_mean, 3),
            "train_mean":  round(train_mean, 3),
            "z_score":     round(z_score, 3),
            "drift_level": drift_level,
        }

    # Prediction distribution
    recent_preds  = [l["prediction"]["crop"] for l in logs]
    crop_counts   = Counter(recent_preds)

    report = {
        "status":       "ok",
        "n_predictions": len(logs),
        "features":     drift_results,
        "crop_distribution": dict(crop_counts),
        "generated_at": datetime.datetime.now(timezone.utc).isoformat(),
    }

    # Save drift report
    with open(DRIFT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
