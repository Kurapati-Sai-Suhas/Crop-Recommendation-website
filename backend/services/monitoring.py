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

# Written by the training pipeline from the training split. Previously these
# statistics were hardcoded literals in this file, which meant they silently
# described whatever dataset happened to be current when they were typed --
# and went stale the moment the data changed.
BASELINE_PATH = os.path.join(
    BASE_DIR, "backend", "models", "monitoring_baseline.json"
)


def load_baseline():
    """Load the training-set feature statistics, or None if not yet trained."""
    if not os.path.exists(BASELINE_PATH):
        return None
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    if not all(f in baseline.get("features", {}) for f in FEATURES):
        return None
    return baseline


def _drift_level(z_score: float) -> str:
    """
    Map a z-score to a coarse drift level.

    These thresholds are a triage heuristic, not a hypothesis test: the
    inputs are user-submitted and not independent samples from any
    population, so the z-score is a relative-movement signal rather than
    a calibrated p-value.
    """
    if z_score > 3:
        return "high"
    if z_score > 2:
        return "medium"
    return "low"


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

    baseline = load_baseline()
    if baseline is None:
        return {
            "status": "no_baseline",
            "detail": ("Training baseline not found. Run the training "
                       "pipeline to regenerate monitoring_baseline.json."),
            "features": {},
        }

    drift_results = {}
    for f in FEATURES:
        vals = recent_vals[f]
        if not vals:
            drift_results[f] = {"status": "no_data"}
            continue

        stats      = baseline["features"][f]
        train_mean = stats["mean"]
        train_std  = stats["std"]

        n           = len(vals)
        recent_mean = float(np.mean(vals))

        # Standard error of the mean, not the raw feature std. Comparing a
        # sample mean against a population mean using the population's own
        # spread understates the signal by a factor of sqrt(n): a genuine
        # shift across 50 requests would sit at z < 1 and never be flagged.
        standard_error = train_std / np.sqrt(n) if train_std > 0 else 0.0
        z_score = (abs(recent_mean - train_mean) / standard_error
                   if standard_error > 0 else 0.0)

        drift_results[f] = {
            "recent_mean":    round(recent_mean, 3),
            "train_mean":     round(train_mean, 3),
            "train_std":      round(train_std, 3),
            "n_samples":      n,
            "standard_error": round(float(standard_error), 4),
            "z_score":        round(float(z_score), 3),
            "drift_level":    _drift_level(z_score),
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
