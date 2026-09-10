# -*- coding: utf-8 -*-
"""
============================================================
backend/services/train.py
============================================================
Training and evaluation pipeline for the Crop Recommendation System.

Protocol
--------
1. Hold out a stratified 20% test set. It is touched exactly once,
   at the very end, after every modelling decision has been made.
2. Compare candidate models by 5-fold stratified cross-validation on
   the training set only.
3. Tune hyperparameters with GridSearchCV, again on the training set
   only, using the same CV splitter.
4. Refit the selected configuration on the full training set and score
   it once on the held-out test set.

Every step that touches feature scaling does so inside a Pipeline, so
the scaler is refit within each CV fold. Fitting one scaler on the whole
training set before cross-validating would leak fold-validation
statistics into fold-training -- a small effect here, but the habit is
what matters.

Artifacts written to backend/models/
------------------------------------
  RandomForest.joblib / LogisticRegression.joblib / NaiveBayes.joblib
  scaler.joblib, label_encoder.joblib   (the serving contract)
  metrics.json              headline test metrics, read by GET /metrics
  evaluation_report.json    CV table, tuning results, per-class scores,
                            confusion pairs, permutation importance
  monitoring_baseline.json  training feature statistics for drift checks

Run directly: python -m backend.services.train
============================================================
"""

import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH    = os.path.join(BASE_DIR, "data", "crop_data.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "backend", "models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
REPORT_PATH  = os.path.join(MODEL_DIR, "evaluation_report.json")
BASELINE_PATH = os.path.join(MODEL_DIR, "monitoring_baseline.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
FEATURES  = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET    = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS  = 5

# One splitter object, reused for model comparison and for tuning, so every
# candidate is scored on identical folds.
CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def _candidates():
    """
    Candidate models and their search grids.

    The three are deliberately different in inductive bias rather than
    three flavours of the same idea:
      - GaussianNB      : generative, assumes per-class independent Gaussians
      - LogisticRegression : linear decision boundaries
      - RandomForest    : axis-aligned, non-linear, models interactions

    Comparing them answers a real question -- does this problem need a
    non-linear boundary? -- instead of padding the report.
    """
    return {
        "RandomForest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {
                # n_estimators: variance reduction; more trees never hurts
                #   accuracy, only runtime, so this checks where it plateaus.
                "model__n_estimators":     [200, 400],
                # max_depth / min_samples_leaf: the actual capacity controls.
                #   With 22 classes and 1760 training rows, an unbounded tree
                #   can memorise; this is the overfitting knob that matters.
                "model__max_depth":        [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {
                # C is inverse regularisation strength -- the only meaningful
                # knob for a linear model on 7 standardised features.
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
        "NaiveBayes": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]),
            {
                # var_smoothing floors the per-feature variance, guarding
                # against near-zero-variance features in a class.
                "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
            },
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load the dataset and report what was actually read."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Run: python data/download_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    missing = df.isna().sum().sum()
    dupes   = df.duplicated().sum()
    print(f"[DATA] {DATA_PATH}")
    print(f"       {len(df)} rows x {df.shape[1]} cols | "
          f"{df[TARGET].nunique()} crops")
    print(f"       missing cells: {missing} | duplicate rows: {dupes}")

    if missing:
        # Documented rather than silently dropped: on this dataset the count
        # is zero, and a sudden non-zero value should be noticed, not hidden.
        print(f"[WARN] Dropping {missing} rows with missing values")
        df = df.dropna()

    counts = df[TARGET].value_counts()
    print(f"       class balance: min {counts.min()}, max {counts.max()} "
          f"per crop -- {'balanced' if counts.min() == counts.max() else 'IMBALANCED'}")

    return df


def split_data(df):
    """Stratified train/test split. The test set is not touched until the end."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    X = df[FEATURES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[SPLIT] train {len(X_train)} | test {len(X_test)} (stratified)")
    return X_train, X_test, y_train, y_test, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — model comparison by cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def compare_models(X_train, y_train):
    """
    Score every untuned candidate by cross-validation on the training set.

    This is the honest comparison: no model has seen the test set, and each
    is scored on identical folds.
    """
    print("\n" + "-" * 62)
    print(f"  STAGE 1 - baseline comparison ({N_SPLITS}-fold CV, train only)")
    print("-" * 62)

    results = {}
    for name, (pipe, _grid) in _candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=CV,
                                 scoring="accuracy", n_jobs=-1)
        results[name] = {
            "cv_mean": round(float(scores.mean()), 4),
            "cv_std":  round(float(scores.std()), 4),
            "folds":   [round(float(s), 4) for s in scores],
        }
        print(f"  {name:20s} {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_models(X_train, y_train):
    """
    Grid-search each candidate on the training set.

    GridSearchCV refits the whole Pipeline inside every fold, so the scaler
    never sees a fold's validation rows before scoring them.
    """
    print("\n" + "-" * 62)
    print(f"  STAGE 2 - hyperparameter tuning (GridSearchCV, train only)")
    print("-" * 62)

    tuned = {}
    for name, (pipe, grid) in _candidates().items():
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        print(f"  {name}: {n_combos} configs x {N_SPLITS} folds "
              f"= {n_combos * N_SPLITS} fits")

        search = GridSearchCV(
            pipe, grid, cv=CV, scoring="accuracy", n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)

        tuned[name] = {
            "estimator":   search.best_estimator_,
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_mean":     round(float(search.best_score_), 4),
        }
        print(f"    -> best CV {search.best_score_:.4f} | {search.best_params_}")

    return tuned


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — single evaluation on the held-out test set
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(tuned, X_test, y_test, encoder):
    """Score every tuned model once on the held-out test set."""
    print("\n" + "-" * 62)
    print("  STAGE 3 - held-out test set (touched once)")
    print("-" * 62)

    metrics = {}
    for name, entry in tuned.items():
        y_pred = entry["estimator"].predict(X_test)
        metrics[name] = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(
                y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall":    round(float(recall_score(
                y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score":  round(float(f1_score(
                y_test, y_pred, average="weighted", zero_division=0)), 4),
            "cv_mean":     entry["cv_mean"],
            "best_params": entry["best_params"],
        }
        print(f"  {name:20s} acc {metrics[name]['accuracy']:.4f} | "
              f"F1 {metrics[name]['f1_score']:.4f} "
              f"(CV was {entry['cv_mean']:.4f})")

    return metrics


def select_model(baseline_cv, tuned, served="RandomForest"):
    """
    Decide which model to serve, and record why.

    On this dataset the top candidates land within a fraction of a fold
    standard deviation of each other. Declaring a "winner" from that gap
    would be reading noise. So the rule is explicit:

      - If the best model beats the served model by more than one fold
        standard deviation, the gap is treated as real and reported as a
        recommendation to switch.
      - Otherwise the models are called statistically indistinguishable
        and RandomForest is kept -- not because it scored higher, but
        because the product requires per-prediction SHAP attribution and
        TreeExplainer computes that exactly for a forest. GaussianNB would
        need KernelExplainer: approximate and orders of magnitude slower.

    Recording the tie-break is the point. "Accuracy picked it" would be a
    false statement about a 0.1pp difference.
    """
    ranked = sorted(tuned.items(), key=lambda kv: -kv[1]["cv_mean"])
    best_name, best = ranked[0]

    served_cv = tuned[served]["cv_mean"]
    fold_std  = baseline_cv[served]["cv_std"]
    gap       = best["cv_mean"] - served_cv
    decisive  = gap > fold_std

    print("\n" + "-" * 62)
    print("  MODEL SELECTION")
    print("-" * 62)
    for name, entry in ranked:
        print(f"  {name:20s} CV {entry['cv_mean']:.4f}")
    print(f"\n  highest CV     : {best_name} ({best['cv_mean']:.4f})")
    print(f"  served model   : {served} ({served_cv:.4f})")
    print(f"  gap            : {gap:+.4f} vs fold std {fold_std:.4f}")

    if decisive:
        print(f"  -> gap exceeds fold noise. Consider serving {best_name}.")
    else:
        print(f"  -> within fold noise: statistically indistinguishable.")
        print(f"     Keeping {served} for exact TreeSHAP attribution.")

    return {
        "highest_cv_model": best_name,
        "highest_cv_score": best["cv_mean"],
        "served_model":     served,
        "served_cv_score":  served_cv,
        "gap":              round(float(gap), 4),
        "fold_std":         fold_std,
        "gap_exceeds_fold_noise": bool(decisive),
        "rationale": (
            f"{best_name} scored highest in cross-validation, but the "
            f"{gap:+.4f} gap over {served} is within one fold standard "
            f"deviation ({fold_std:.4f}), so the two are not "
            f"distinguishable on this data. {served} is served because the "
            f"application requires per-prediction SHAP values, and "
            f"TreeExplainer computes those exactly for a forest."
        ) if not decisive else (
            f"{best_name} beats {served} by {gap:+.4f}, more than one fold "
            f"standard deviation ({fold_std:.4f}). This gap is worth acting on."
        ),
    }


def error_analysis(estimator, X_test, y_test, encoder):
    """
    Which crops does the model actually confuse, and how badly?

    A 99% headline hides the only interesting part of this dataset. The
    confusion pairs below are the part worth talking about.
    """
    y_pred = estimator.predict(X_test)
    labels = list(encoder.classes_)

    cm = confusion_matrix(y_test, y_pred)
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)

    pairs = []
    for i, j in zip(*np.nonzero(off_diagonal)):
        pairs.append({
            "true":      labels[i],
            "predicted": labels[j],
            "count":     int(off_diagonal[i, j]),
        })
    pairs.sort(key=lambda p: -p["count"])

    report = classification_report(
        y_test, y_pred, target_names=labels,
        output_dict=True, zero_division=0,
    )
    per_class = {
        crop: {
            "precision": round(float(scores["precision"]), 4),
            "recall":    round(float(scores["recall"]), 4),
            "f1_score":  round(float(scores["f1-score"]), 4),
            "support":   int(scores["support"]),
        }
        for crop, scores in report.items() if crop in labels
    }

    n_errors = int((y_test != y_pred).sum())
    print(f"\n[ERRORS] {n_errors} misclassified of {len(y_test)} test rows")
    for p in pairs[:5]:
        print(f"         {p['true']:12s} -> {p['predicted']:12s} x{p['count']}")

    weakest = sorted(per_class.items(), key=lambda kv: kv[1]["recall"])[:3]
    print("[WEAKEST RECALL] " +
          ", ".join(f"{c} {s['recall']:.2f}" for c, s in weakest))

    return {
        "n_test_rows":    int(len(y_test)),
        "n_errors":       n_errors,
        "confusion_pairs": pairs,
        "per_class":      per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def feature_importance(estimator, X_test, y_test):
    """
    Permutation importance on the test set.

    Preferred over RandomForest's built-in `feature_importances_`, which is
    impurity-based and biased toward high-cardinality continuous features.
    Permutation importance measures what the model actually relies on.
    """
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1,
    )
    ranked = sorted(
        (
            {
                "feature": f,
                "importance_mean": round(float(m), 4),
                "importance_std":  round(float(s), 4),
            }
            for f, m, s in zip(FEATURES,
                               result.importances_mean,
                               result.importances_std)
        ),
        key=lambda d: -d["importance_mean"],
    )
    print("\n[PERMUTATION IMPORTANCE] " +
          ", ".join(f"{d['feature']} {d['importance_mean']:.3f}"
                    for d in ranked[:4]))
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — persist the serving artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(tuned, encoder, X_train):
    """
    Save models in the shape the inference service expects.

    predictor.py loads a bare estimator plus a separate scaler and applies
    them in sequence. Each tuned Pipeline already holds a scaler fitted on
    the training set and a model fitted on that scaler's output, so
    splitting the pipeline preserves exactly the train-time transform.
    """
    scaler = None
    for name, entry in tuned.items():
        pipeline = entry["estimator"]
        joblib.dump(pipeline.named_steps["model"],
                    os.path.join(MODEL_DIR, f"{name}.joblib"))
        if scaler is None:
            scaler = pipeline.named_steps["scaler"]

    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print(f"\n[SAVED] models + scaler + label encoder -> {MODEL_DIR}")

    # Drift baseline computed from the training split, not hardcoded.
    baseline = {
        "source":   "training split",
        "n_rows":   int(len(X_train)),
        "features": {
            f: {
                "mean": round(float(X_train[:, i].mean()), 4),
                "std":  round(float(X_train[:, i].std(ddof=1)), 4),
                "min":  round(float(X_train[:, i].min()), 4),
                "max":  round(float(X_train[:, i].max()), 4),
            }
            for i, f in enumerate(FEATURES)
        },
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2)
    print(f"[SAVED] monitoring baseline -> {BASELINE_PATH}")


def _log_to_mlflow(metrics, baseline_cv):
    """Log one MLflow run per model. Never fails the training run."""
    try:
        import mlflow
    except ImportError:
        print("[MLFLOW] not installed -- skipping experiment tracking")
        return

    try:
        mlruns_dir = os.path.join(BASE_DIR, "mlops", "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")
        mlflow.set_experiment("CropRecommendation")

        for name, m in metrics.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_params(m["best_params"])
                mlflow.log_metrics({
                    "test_accuracy":  m["accuracy"],
                    "test_precision": m["precision"],
                    "test_recall":    m["recall"],
                    "test_f1":        m["f1_score"],
                    "cv_accuracy_tuned":   m["cv_mean"],
                    "cv_accuracy_untuned": baseline_cv[name]["cv_mean"],
                })
        print("[MLFLOW] runs logged")
    except Exception as exc:
        print(f"[MLFLOW] tracking skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def train_all_models():
    """Run the full protocol and return the headline test metrics."""
    df = load_data()
    X_train, X_test, y_train, y_test, encoder = split_data(df)

    baseline_cv = compare_models(X_train, y_train)
    tuned       = tune_models(X_train, y_train)
    metrics     = evaluate_on_test(tuned, X_test, y_test, encoder)

    selection = select_model(baseline_cv, tuned)
    primary   = selection["served_model"]

    errors     = error_analysis(tuned["RandomForest"]["estimator"],
                                X_test, y_test, encoder)
    importance = feature_importance(tuned["RandomForest"]["estimator"],
                                    X_test, y_test)

    save_artifacts(tuned, encoder, X_train)

    # metrics.json keeps the flat shape GET /metrics validates against.
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    report = {
        "protocol": {
            "test_size":     TEST_SIZE,
            "cv_folds":      N_SPLITS,
            "random_state":  RANDOM_STATE,
            "n_train":       int(len(X_train)),
            "n_test":        int(len(X_test)),
            "scaling":       "StandardScaler inside Pipeline, refit per CV fold",
            "selection":     "cross-validated accuracy on the training set, "
                             "with an explicit tie-break when the gap is "
                             "inside fold noise (see model_selection)",
        },
        "baseline_cv":          baseline_cv,
        "tuned_test_metrics":   metrics,
        "model_selection":      selection,
        "selected_model":       primary,
        "error_analysis":       errors,
        "permutation_importance": importance,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[SAVED] metrics -> {METRICS_PATH}")
    print(f"[SAVED] evaluation report -> {REPORT_PATH}")

    _log_to_mlflow(metrics, baseline_cv)
    return metrics


if __name__ == "__main__":
    print("=" * 62)
    print("  Crop Recommendation -- training & evaluation")
    print("=" * 62)
    train_all_models()
    print("\n[DONE]")
