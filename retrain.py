# -*- coding: utf-8 -*-
"""
============================================================
retrain.py  (project root)
============================================================
Retrain on new data, but only keep the result if it is better.

Retraining is the easy half. The half that matters is refusing to
promote a model that is not an improvement, and being able to get
the previous one back. This script does both:

    validate -> back up -> retrain -> compare -> promote or roll back

The promotion gate reuses the rule the training pipeline already
applies to model selection: a difference smaller than the
cross-validation fold standard deviation is noise, not an
improvement, and noise is not a reason to ship. Without that gate,
repeated retraining is a random walk that eventually ships a worse
model than the one it replaced.

Usage
-----
  python retrain.py                        # retrain on the current dataset
  python retrain.py --data path/to/new.csv # retrain on new data
  python retrain.py --dry-run              # validate + report, change nothing
  python retrain.py --rollback             # restore the previous model set

Trigger
-------
Nothing schedules this. It is meant to be run when something has
actually changed: the drift report at GET /api/v1/monitoring shows a
sustained shift, or new labelled data arrives. Retraining on a fixed
timetable against unchanged data burns compute and adds risk for no
information gain.
============================================================
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODEL_DIR   = os.path.join(ROOT, "backend", "models")
BACKUP_DIR  = os.path.join(ROOT, "backend", "models_previous")
DATA_PATH   = os.path.join(ROOT, "data", "crop_data.csv")
HISTORY     = os.path.join(MODEL_DIR, "retrain_history.json")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

ARTIFACTS = [
    "RandomForest.joblib", "LogisticRegression.joblib", "NaiveBayes.joblib",
    "scaler.joblib", "label_encoder.joblib", "metrics.json",
    "evaluation_report.json", "monitoring_baseline.json",
]

PRIMARY = "RandomForest"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validate before training on it
# ─────────────────────────────────────────────────────────────────────────────
def validate_dataset(path):
    """
    Reject data that would produce a broken model, before spending
    compute on it. Every check here corresponds to a failure that would
    otherwise surface at serving time as a confident wrong answer.
    """
    problems, warnings = [], []

    def report(problems, warnings, df=None):
        """Print before returning, on every path -- an early return that
        skipped this made the most basic failure (a missing column) exit
        silently with no reason given."""
        if df is not None and TARGET in df.columns:
            print(f"[VALIDATE] {len(df)} rows | {df[TARGET].nunique()} classes")
        elif df is not None:
            print(f"[VALIDATE] {len(df)} rows | schema check failed")
        else:
            print("[VALIDATE] dataset could not be read")
        for w in warnings:
            print(f"  [WARN]  {w}")
        for p in problems:
            print(f"  [ERROR] {p}")
        if not problems and not warnings:
            print("  clean")
        return problems, warnings

    if not os.path.exists(path):
        return report([f"Dataset not found: {path}"], [])

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return report([f"Could not read {path}: {exc}"], [])

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing required columns: {missing_cols}")
        # Every later check indexes those columns, so stop here.
        return report(problems, warnings, df)

    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column '{col}' is not numeric")

    n_missing = int(df[FEATURES + [TARGET]].isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing cells will be dropped")

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        warnings.append(f"{n_dupes} duplicate rows present")

    counts = df[TARGET].value_counts()
    if len(counts) < 2:
        problems.append(f"Only {len(counts)} class present; nothing to learn")
    if counts.min() < 10:
        problems.append(
            f"Class '{counts.idxmin()}' has {counts.min()} samples -- too few "
            f"for a stratified 5-fold split"
        )
    if counts.max() / counts.min() > 3:
        warnings.append(
            f"Class imbalance {counts.max()}:{counts.min()} -- plain accuracy "
            f"is no longer a fair headline metric"
        )

    # Compare against the ranges the current model was trained on. Values far
    # outside them are not necessarily wrong, but they change what the model
    # is, so they should be a decision rather than a surprise.
    baseline_path = os.path.join(MODEL_DIR, "monitoring_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as handle:
            baseline = json.load(handle)["features"]
        for col in FEATURES:
            stats = baseline.get(col)
            if not stats:
                continue
            span = stats["max"] - stats["min"]
            lo   = stats["min"] - 0.5 * span
            hi   = stats["max"] + 0.5 * span
            out  = int(((df[col] < lo) | (df[col] > hi)).sum())
            if out:
                warnings.append(
                    f"{out} rows have '{col}' far outside the previous "
                    f"training range [{stats['min']}, {stats['max']}]"
                )

    return report(problems, warnings, df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backup / restore
# ─────────────────────────────────────────────────────────────────────────────
def backup_current():
    """Copy the live artifacts aside so a bad promotion can be undone."""
    if not os.path.exists(os.path.join(MODEL_DIR, f"{PRIMARY}.joblib")):
        print("[BACKUP] No existing model to back up (first training run)")
        return False

    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name in ARTIFACTS:
        src = os.path.join(MODEL_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BACKUP_DIR, name))
    print(f"[BACKUP] Previous model set -> {BACKUP_DIR}")
    return True


def rollback():
    """Restore the backed-up artifacts."""
    if not os.path.isdir(BACKUP_DIR):
        print("[ROLLBACK] No backup directory; nothing to restore")
        return False

    restored = 0
    for name in ARTIFACTS:
        src = os.path.join(BACKUP_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODEL_DIR, name))
            restored += 1
    print(f"[ROLLBACK] Restored {restored} artifacts from {BACKUP_DIR}")
    return restored > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Promotion gate
# ─────────────────────────────────────────────────────────────────────────────
def read_metrics(directory):
    path = os.path.join(directory, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fold_std(directory):
    """The training pipeline's own estimate of run-to-run noise."""
    path = os.path.join(directory, "evaluation_report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return report.get("baseline_cv", {}).get(PRIMARY, {}).get("cv_std")


def decide(incumbent, candidate, noise):
    """
    Promote only on evidence.

    Three outcomes, and the middle one is the one people skip:
      - clearly better  -> promote
      - inside noise    -> keep the incumbent (a coin flip is not an upgrade)
      - clearly worse   -> roll back
    """
    if incumbent is None:
        return True, "No incumbent model; promoting the first trained set."

    old = incumbent[PRIMARY]["accuracy"]
    new = candidate[PRIMARY]["accuracy"]
    delta = new - old
    noise = noise if noise else 0.005

    if delta > noise:
        return True, (f"Candidate {new:.4f} beats incumbent {old:.4f} by "
                      f"{delta:+.4f}, more than fold noise ({noise:.4f}). "
                      f"Promoting.")
    if delta < -noise:
        return False, (f"Candidate {new:.4f} is worse than incumbent "
                       f"{old:.4f} by {delta:+.4f}, beyond fold noise "
                       f"({noise:.4f}). Rolling back.")
    return False, (f"Candidate {new:.4f} vs incumbent {old:.4f} differ by "
                   f"{delta:+.4f}, inside fold noise ({noise:.4f}). "
                   f"Not a measurable improvement -- keeping the incumbent.")


def record(entry):
    history = []
    if os.path.exists(HISTORY):
        try:
            with open(HISTORY, encoding="utf-8") as handle:
                history = json.load(handle)
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    with open(HISTORY, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"[HISTORY] {HISTORY}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Retrain with a promotion gate")
    parser.add_argument("--data", default=DATA_PATH,
                        help="CSV to retrain on (default: the current dataset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report; change nothing")
    parser.add_argument("--rollback", action="store_true",
                        help="Restore the previous model set and exit")
    parser.add_argument("--force", action="store_true",
                        help="Promote even if the gate says no")
    args = parser.parse_args()

    print("=" * 62)
    print("  Retraining with promotion gate")
    print("=" * 62)

    if args.rollback:
        return 0 if rollback() else 1

    problems, _warnings = validate_dataset(args.data)
    if problems:
        print("\n[ABORT] Dataset failed validation. Nothing was changed.")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Validation passed. No training performed.")
        return 0

    incumbent = read_metrics(MODEL_DIR)
    noise     = fold_std(MODEL_DIR)
    had_backup = backup_current()

    # Point the trainer at the requested data if it is not the default path.
    if os.path.abspath(args.data) != os.path.abspath(DATA_PATH):
        shutil.copy2(args.data, DATA_PATH)
        print(f"[DATA] Copied {args.data} -> {DATA_PATH}")

    print("\n[TRAIN] Running the full pipeline...")
    from backend.services.train import train_all_models
    candidate = train_all_models()

    promote, reason = decide(incumbent, candidate, noise)
    print("\n" + "-" * 62)
    print("  PROMOTION DECISION")
    print("-" * 62)
    print(f"  {reason}")

    if args.force and not promote:
        print("  [FORCE] Overriding the gate as requested.")
        promote = True

    if not promote and had_backup:
        rollback()

    record({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "data":       os.path.abspath(args.data),
        "incumbent_accuracy": (incumbent[PRIMARY]["accuracy"]
                               if incumbent else None),
        "candidate_accuracy": candidate[PRIMARY]["accuracy"],
        "fold_noise": noise,
        "promoted":   promote,
        "reason":     reason,
    })

    print(f"\n[RESULT] {'PROMOTED' if promote else 'KEPT INCUMBENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
