# -*- coding: utf-8 -*-
"""
============================================================
eda/full_evaluation.py
============================================================
Every evaluation number for the three models, with its provenance.

The training pipeline reports a headline accuracy and writes a
per-class report. This script is the complete audit: training
accuracy, out-of-bag, per-fold cross-validation, the held-out test
set, threshold-free metrics, calibration, and confidence intervals.

Three things here are deliberately not in the main pipeline:

  * Confidence intervals. With 440 test rows and 2 errors, quoting
    "0.9955" implies four-digit precision the sample size cannot
    support. The Wilson interval says what the number actually is.

  * ROC-AUC and log loss. Listed as a known gap in the README;
    computed here so the gap is closed and the reason they add
    little on this data is visible rather than asserted.

  * Calibration (ECE, Brier). The README quotes an ECE figure; this
    measures it rather than repeating it.

Nothing here re-tunes anything. The hyperparameters are the ones the
grid search already selected, so the test set is still being read
after every modelling decision, not before.

Run: python eda/full_evaluation.py
============================================================
"""

import io
import json
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RANDOM_STATE, TEST_SIZE, N_SPLITS = 42, 0.2, 5
RULE = "=" * 74

# Exactly what GridSearchCV selected in the training pipeline.
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1, oob_score=True),
    "LogisticRegression": LogisticRegression(
        C=100.0, max_iter=2000, random_state=RANDOM_STATE),
    "NaiveBayes": GaussianNB(var_smoothing=1e-11),
}


def wilson(successes, n, z=1.96):
    """Wilson score interval -- correct near 1.0, where normal approx is not."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def expected_calibration_error(y_true, proba, n_bins=10):
    """
    ECE: average gap between confidence and accuracy, binned by confidence.

    A perfectly calibrated model predicting 0.8 is right 80% of the time.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, avg_conf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - avg_conf)
        rows.append((lo, hi, int(m.sum()), avg_conf, acc, acc - avg_conf))
    return ece, rows


def main():
    df = pd.read_csv("data/crop_data.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df[FEATURES].values
    names = list(encoder.classes_)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(RULE)
    print("  0. PROTOCOL")
    print(RULE)
    print(f"  dataset            data/crop_data.csv  ({len(df)} rows, "
          f"{len(FEATURES)} features, {len(names)} classes)")
    print(f"  split              stratified {int((1-TEST_SIZE)*100)}/"
          f"{int(TEST_SIZE*100)}, random_state={RANDOM_STATE}")
    print(f"  train / test       {len(Xtr)} / {len(Xte)}")
    print(f"  cross-validation   StratifiedKFold({N_SPLITS}, shuffle=True, "
          f"random_state={RANDOM_STATE}) on TRAIN only")
    print(f"  preprocessing      StandardScaler inside a Pipeline, refit per fold")
    print(f"  hyperparameters    as selected by GridSearchCV in the training run")

    fitted, results = {}, {}
    for name, est in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe

    # ── 1. train vs CV vs test ───────────────────────────────────────────
    print("\n" + RULE)
    print("  1. TRAIN  vs  CROSS-VALIDATION  vs  TEST")
    print(RULE)
    print(f"  {'model':20s} {'train':>8s} {'CV mean':>9s} {'CV sd':>7s}"
          f" {'test':>8s} {'train-test':>11s}")
    for name, pipe in fitted.items():
        tr = accuracy_score(ytr, pipe.predict(Xtr))
        te = accuracy_score(yte, pipe.predict(Xte))
        folds = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = {"train": tr, "test": te, "folds": folds}
        print(f"  {name:20s} {tr:8.4f} {folds.mean():9.4f} {folds.std():7.4f}"
              f" {te:8.4f} {tr-te:+11.4f}")

    print("\n  per-fold cross-validation accuracy (train split only)")
    print(f"  {'model':20s} " + " ".join(f"{'fold '+str(i+1):>8s}"
                                          for i in range(N_SPLITS)))
    for name, r in results.items():
        print(f"  {name:20s} " + " ".join(f"{v:8.4f}" for v in r["folds"]))

    oob = fitted["RandomForest"].named_steps["model"].oob_score_
    print(f"\n  RandomForest out-of-bag score  {oob:.4f}"
          f"   (held out by construction, computed during fit)")

    # ── 2. test metrics ──────────────────────────────────────────────────
    print("\n" + RULE)
    print("  2. HELD-OUT TEST METRICS")
    print(RULE)
    print("  macro = unweighted mean over classes; weighted = by support.")
    print("  With 20 test rows per class these coincide almost exactly.\n")
    header = (f"  {'model':20s} {'acc':>7s} {'bal acc':>8s} {'P mac':>7s}"
              f" {'R mac':>7s} {'F1 mac':>7s} {'F1 wtd':>7s} {'kappa':>7s} {'MCC':>7s}")
    print(header)
    for name, pipe in fitted.items():
        p = pipe.predict(Xte)
        print(f"  {name:20s} {accuracy_score(yte,p):7.4f}"
              f" {balanced_accuracy_score(yte,p):8.4f}"
              f" {precision_score(yte,p,average='macro',zero_division=0):7.4f}"
              f" {recall_score(yte,p,average='macro',zero_division=0):7.4f}"
              f" {f1_score(yte,p,average='macro',zero_division=0):7.4f}"
              f" {f1_score(yte,p,average='weighted',zero_division=0):7.4f}"
              f" {cohen_kappa_score(yte,p):7.4f}"
              f" {matthews_corrcoef(yte,p):7.4f}")

    # ── 3. probability-based metrics ─────────────────────────────────────
    print("\n" + RULE)
    print("  3. THRESHOLD-FREE AND PROBABILITY METRICS")
    print(RULE)
    print(f"  {'model':20s} {'ROC-AUC ovr':>12s} {'log loss':>10s}"
          f" {'Brier':>8s} {'top-2 acc':>10s} {'top-3 acc':>10s}")
    for name, pipe in fitted.items():
        pr = pipe.predict_proba(Xte)
        onehot = np.eye(len(names))[yte]
        brier = float(np.mean(np.sum((pr - onehot) ** 2, axis=1)))
        print(f"  {name:20s}"
              f" {roc_auc_score(yte,pr,multi_class='ovr',average='macro'):12.4f}"
              f" {log_loss(yte,pr,labels=range(len(names))):10.4f}"
              f" {brier:8.4f}"
              f" {top_k_accuracy_score(yte,pr,k=2,labels=range(len(names))):10.4f}"
              f" {top_k_accuracy_score(yte,pr,k=3,labels=range(len(names))):10.4f}")
    print("\n  ROC-AUC is near 1.0 for every model, which is why the README")
    print("  called it uninformative here: it cannot separate models that all")
    print("  rank correctly. Log loss does separate them, because it punishes")
    print("  confident errors -- note NaiveBayes.")

    # ── 4. confidence intervals ──────────────────────────────────────────
    print("\n" + RULE)
    print("  4. HOW PRECISE IS 0.9955, REALLY?")
    print(RULE)
    print(f"  95% Wilson intervals on {len(yte)} test rows.\n")
    print(f"  {'model':20s} {'correct':>9s} {'accuracy':>9s} {'95% CI':>20s}")
    for name, pipe in fitted.items():
        p = pipe.predict(Xte)
        k = int((p == yte).sum())
        lo, hi = wilson(k, len(yte))
        print(f"  {name:20s} {k:4d}/{len(yte):<4d} {k/len(yte):9.4f}"
              f"   [{lo:.4f}, {hi:.4f}]")
    print("\n  The intervals overlap heavily. Quoting four decimal places on a")
    print("  440-row test set implies precision the sample size cannot give:")
    print("  0.9955 honestly means 'somewhere around 98.4-99.9%'.")

    # ── 5. calibration ───────────────────────────────────────────────────
    print("\n" + RULE)
    print("  5. CALIBRATION  (is a confidence of 0.8 right 80% of the time?)")
    print(RULE)
    for name, pipe in fitted.items():
        ece, _ = expected_calibration_error(yte, pipe.predict_proba(Xte))
        print(f"  {name:20s} ECE {ece:.4f}")
    print("\n  RandomForest reliability table:")
    ece, rows = expected_calibration_error(yte, fitted["RandomForest"].predict_proba(Xte))
    print(f"  {'confidence bin':>16s} {'n':>5s} {'mean conf':>10s}"
          f" {'accuracy':>9s} {'gap':>8s}")
    for lo, hi, n, c, a, g in rows:
        print(f"  {f'{lo:.1f}-{hi:.1f}':>16s} {n:5d} {c:10.4f} {a:9.4f} {g:+8.4f}")
    print("\n  A negative gap means over-confidence, positive means under-.")

    # ── 6. per class ─────────────────────────────────────────────────────
    print("\n" + RULE)
    print("  6. PER-CLASS REPORT  (served model: RandomForest)")
    print(RULE)
    pred = fitted["RandomForest"].predict(Xte)
    rep = classification_report(yte, pred, target_names=names,
                                output_dict=True, zero_division=0)
    print(f"  {'crop':14s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for crop in names:
        r = rep[crop]
        print(f"  {crop:14s} {r['precision']:10.3f} {r['recall']:8.3f}"
              f" {r['f1-score']:8.3f} {int(r['support']):8d}")
    print(f"  {'-'*52}")
    for avg in ("macro avg", "weighted avg"):
        r = rep[avg]
        print(f"  {avg:14s} {r['precision']:10.3f} {r['recall']:8.3f}"
              f" {r['f1-score']:8.3f} {int(r['support']):8d}")

    # ── 7. confusion ─────────────────────────────────────────────────────
    print("\n" + RULE)
    print("  7. CONFUSION  (off-diagonal only -- 22x22 is unreadable in full)")
    print(RULE)
    cm = confusion_matrix(yte, pred)
    off = cm.copy(); np.fill_diagonal(off, 0)
    total = int(off.sum())
    print(f"  misclassified {total} of {len(yte)} test rows\n")
    for i, j in zip(*np.nonzero(off)):
        print(f"    {names[i]:14s} -> {names[j]:14s}  x{off[i,j]}")
    print(f"\n  every other cell of the 22x22 matrix is zero off the diagonal")

    # ── 8. save ──────────────────────────────────────────────────────────
    out = {
        "protocol": {"n_train": len(Xtr), "n_test": len(Xte),
                     "cv_folds": N_SPLITS, "random_state": RANDOM_STATE},
        "models": {},
    }
    for name, pipe in fitted.items():
        p, pr = pipe.predict(Xte), pipe.predict_proba(Xte)
        k = int((p == yte).sum())
        lo, hi = wilson(k, len(yte))
        e, _ = expected_calibration_error(yte, pr)
        out["models"][name] = {
            "train_accuracy": round(float(results[name]["train"]), 4),
            "cv_mean": round(float(results[name]["folds"].mean()), 4),
            "cv_std": round(float(results[name]["folds"].std()), 4),
            "cv_folds": [round(float(v), 4) for v in results[name]["folds"]],
            "test_accuracy": round(float(results[name]["test"]), 4),
            "test_accuracy_95ci": [round(lo, 4), round(hi, 4)],
            "balanced_accuracy": round(float(balanced_accuracy_score(yte, p)), 4),
            "f1_macro": round(float(f1_score(yte, p, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(yte, p, average="weighted", zero_division=0)), 4),
            "cohen_kappa": round(float(cohen_kappa_score(yte, p)), 4),
            "mcc": round(float(matthews_corrcoef(yte, p)), 4),
            "roc_auc_ovr_macro": round(float(roc_auc_score(yte, pr, multi_class="ovr", average="macro")), 4),
            "log_loss": round(float(log_loss(yte, pr, labels=range(len(names)))), 4),
            "ece": round(float(e), 4),
        }
    out["models"]["RandomForest"]["oob_score"] = round(float(oob), 4)
    path = os.path.join("backend", "models", "full_evaluation.json")
    with open(path, "w", encoding="utf-8") as h:
        json.dump(out, h, indent=2)
    print(f"\n[SAVED] {path}")


if __name__ == "__main__":
    main()
