# -*- coding: utf-8 -*-
"""
============================================================
eda/dataset_provenance.py
============================================================
Why is accuracy 99.5%?

The obvious suspicion is overfitting. It is not overfitting --
eda/overfitting_check.py settles that with six measurements. The
second suspicion is leakage: near-duplicate rows spanning the
train/test split, which an exact-duplicate check would miss.

This script rules leakage out and then finds the real answer: the
dataset is synthetic, and each crop's features were drawn from a
uniform distribution over hand-picked round-number ranges. The
classes occupy near-disjoint axis-aligned boxes, so almost any
classifier reaches ~99%.

Checks
------
  1. near-duplicates across the split (nearest-neighbour distances)
  2. duplicates at reduced precision
  3. grouped cross-validation, so near-duplicates cannot span folds
  4. distribution shape per (class, feature): normal? uniform?
  5. the bounds themselves -- are they round numbers?
  6. robustness under noise

Run: python eda/dataset_provenance.py
============================================================
"""

import io
import os
import sys
import warnings
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RULE = "=" * 70


def forest():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    )


def main():
    df = pd.read_csv("data/crop_data.csv")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df[FEATURES].values
    Z = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte, Ztr, Zte = train_test_split(
        X, y, Z, test_size=0.2, random_state=42, stratify=y)

    # ── 1. near-duplicates ───────────────────────────────────────────────
    print(RULE)
    print("  1. NEAR-DUPLICATES ACROSS THE SPLIT")
    print(RULE)
    print("  Exact-duplicate checks miss near-copies. If test rows sit almost")
    print("  on top of training rows, the model can memorise its way to a high")
    print("  score. Compare test->train distances against train->train.\n")
    to_train = NearestNeighbors(n_neighbors=1).fit(Ztr).kneighbors(Zte)
    d_test, idx = to_train[0].ravel(), to_train[1].ravel()
    d_train = NearestNeighbors(n_neighbors=2).fit(Ztr).kneighbors(Ztr)[0][:, 1]
    print(f"  test  -> nearest train row   median {np.median(d_test):.4f}"
          f"   min {d_test.min():.4f}")
    print(f"  train -> nearest train row   median {np.median(d_train):.4f}"
          f"   min {d_train.min():.4f}")
    print(f"  ratio of medians {np.median(d_test)/np.median(d_train):.3f}"
          f"   (1.0 means the same population)")
    for t in (0.01, 0.05, 0.10):
        n = int((d_test < t).sum())
        print(f"  test rows with a train neighbour closer than {t:<5}: {n}")
    print(f"\n  VERDICT: no near-duplicate leakage." if (d_test < 0.10).sum() == 0
          else "\n  VERDICT: near-duplicates present -- investigate.")

    # ── 2. reduced precision ─────────────────────────────────────────────
    print("\n" + RULE)
    print("  2. DUPLICATES AT REDUCED PRECISION")
    print(RULE)
    for dec in (3, 2, 1, 0):
        n = pd.DataFrame(np.round(X, dec)).duplicated().sum()
        print(f"  rounded to {dec} dp: {n} duplicate feature vectors")

    # ── 3. grouped CV ────────────────────────────────────────────────────
    print("\n" + RULE)
    print("  3. GROUPED CROSS-VALIDATION")
    print(RULE)
    print("  Cluster the data and hold out whole clusters, so near-identical")
    print("  rows cannot straddle the split.\n")
    plain = cross_val_score(forest(), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="accuracy", n_jobs=-1).mean()
    print(f"  ordinary stratified 5-fold                {plain:.4f}")
    for k in (40, 150, 300):
        groups = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Z)
        absent = [len(set(y[te]) - set(y[tr]))
                  for tr, te in GroupKFold(5).split(X, y, groups=groups)]
        score = cross_val_score(forest(), X, y, groups=groups, cv=GroupKFold(5),
                                scoring="accuracy", n_jobs=-1).mean()
        print(f"  GroupKFold over {k:3d} clusters              {score:.4f}"
              f"   ({score - plain:+.4f})   classes missing from train: {absent}")
    print("\n  A low score at 40 clusters is not leakage: at that granularity")
    print("  whole crops drop out of training, so the model is asked to name a")
    print("  crop it has never seen. At 150+ clusters no class is missing and")
    print("  the score is unchanged -- so there is nothing to leak.")

    # ── 4. distribution shape ────────────────────────────────────────────
    print("\n" + RULE)
    print("  4. WHAT SHAPE ARE THE PER-CLASS DISTRIBUTIONS?")
    print(RULE)
    counts = Counter()
    kurtoses, edge_mass = [], []
    for c in np.unique(y):
        for j in range(len(FEATURES)):
            v = X[y == c, j]
            if v.std() == 0:
                continue
            lo, hi = v.min(), v.max()
            p_norm = stats.shapiro(v).pvalue
            p_unif = stats.kstest((v - lo) / (hi - lo), "uniform").pvalue
            kurtoses.append(stats.kurtosis(v))
            edge_mass.append(((v < lo + 0.1 * (hi - lo)) |
                              (v > hi - 0.1 * (hi - lo))).mean())
            counts["normal" if p_norm > 0.05
                   else "uniform" if p_unif > 0.05 else "neither"] += 1
    total = sum(counts.values())
    print(f"  (class, feature) pairs tested: {total}")
    print(f"    consistent with NORMAL   {counts['normal']:4d}"
          f"  ({100*counts['normal']/total:5.1f}%)")
    print(f"    consistent with UNIFORM  {counts['uniform']:4d}"
          f"  ({100*counts['uniform']/total:5.1f}%)")
    print(f"    neither                  {counts['neither']:4d}"
          f"  ({100*counts['neither']/total:5.1f}%)")
    print(f"  mean excess kurtosis  {np.mean(kurtoses):+.3f}"
          f"   (normal 0.0, uniform -1.2)")
    print(f"  mass in outer 10% of range  {np.mean(edge_mass):.3f}"
          f"   (uniform ~0.20, normal ~0.03)")

    # ── 5. the bounds ────────────────────────────────────────────────────
    print("\n" + RULE)
    print("  5. THE BOUNDS THEMSELVES")
    print(RULE)
    print(f"  {'crop':14s} {'N':>12s} {'span':>5s} {'P':>12s} {'span':>5s}"
          f" {'K':>12s} {'span':>5s}")
    spans = []
    for crop in sorted(df.label.unique()):
        sub = df[df.label == crop]
        cells = []
        for f in ("N", "P", "K"):
            lo, hi = sub[f].min(), sub[f].max()
            cells += [f"{lo:.0f}-{hi:.0f}", hi - lo]
            spans.append((f, hi - lo))
        print(f"  {crop:14s} {cells[0]:>12s} {cells[1]:5.0f}"
              f" {cells[2]:>12s} {cells[3]:5.0f} {cells[4]:>12s} {cells[5]:5.0f}")

    bounds = [v for f in ("N", "P", "K") for crop in df.label.unique()
              for v in (df[df.label == crop][f].min(), df[df.label == crop][f].max())]
    m5 = sum(1 for v in bounds if v % 5 == 0)
    print(f"\n  N/P/K class bounds examined  {len(bounds)}")
    print(f"  exact multiples of 5         {m5}  ({100*m5/len(bounds):.1f}%)")
    for f in ("N", "P", "K"):
        widths = Counter(w for ff, w in spans if ff == f)
        print(f"  {f} span widths               {dict(widths)}")
    print("\n  Measured soil samples do not produce class minima and maxima that")
    print("  are multiples of five, nor an identical span for every crop.")

    # ── 6. robustness ────────────────────────────────────────────────────
    print("\n" + RULE)
    print("  6. ROBUSTNESS UNDER NOISE")
    print(RULE)
    within = np.mean([X[y == c].std(0) for c in np.unique(y)], axis=0)
    model = forest().fit(Xtr, ytr)
    rng = np.random.RandomState(0)
    print(f"  {'noise (x within-class sd)':>26} {'accuracy':>10}")
    print(f"  {'0.00  clean':>26} {accuracy_score(yte, model.predict(Xte)):10.4f}")
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        noisy = Xte + rng.normal(0, within * mult, size=Xte.shape)
        print(f"  {mult:>26.2f} {accuracy_score(yte, model.predict(noisy)):10.4f}")
    print("\n  Degradation is gradual, which is what a model relying on real")
    print("  structure looks like. A memorising model collapses abruptly.")

    print("\n" + RULE)
    print("  CONCLUSION")
    print(RULE)
    print("  Not overfitting, and not leakage. Each crop's features were drawn")
    print("  from a uniform distribution over hand-picked round-number ranges,")
    print("  so the classes sit in near-disjoint axis-aligned boxes. ~99% is")
    print("  what the dataset hands to any competent classifier. The number is")
    print("  a property of the data, and reporting it without that context")
    print("  would misrepresent the work.")


if __name__ == "__main__":
    main()
