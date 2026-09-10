# -*- coding: utf-8 -*-
"""
============================================================
eda/overfitting_check.py
============================================================
Is the served Random Forest overfitting?

The question comes up because training accuracy is 1.0000, which
looks alarming. It is the expected behaviour of a forest grown with
min_samples_leaf=1: every tree is grown to purity on its bootstrap
sample, so the ensemble fits the training set exactly. What matters
is not that number on its own but the gap to held-out performance,
and whether constraining capacity would help.

Six checks, none of which relies on the test score alone:

  1. train vs test          the direct gap
  2. out-of-bag score       held-out by construction, computed during fit
  3. tree capacity          how much memorisation is even available
  4. validation curve       does limiting max_depth improve CV?
  5. learning curve         is the gap closing as data grows?
  6. shuffled-label control can the model memorise noise if asked?

Check 6 is the important control. If a model scores at chance on
permuted labels, it cannot be fitting label noise on the real ones.

Note: joblib.load below deserializes pickles. These are first-party
artifacts written by this project's own training pipeline.

Run: python eda/overfitting_check.py
============================================================
"""
import io, os, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split,
                                     validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

F = ["N","P","K","temperature","humidity","ph","rainfall"]
df = pd.read_csv("data/crop_data.csv")
le = LabelEncoder(); y = le.fit_transform(df["label"]); X = df[F].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

print("=" * 66)
print("  1. TRAIN vs TEST  — the direct overfitting test")
print("=" * 66)
scaler = joblib.load("backend/models/scaler.joblib")
model  = joblib.load("backend/models/RandomForest.joblib")
tr_acc = accuracy_score(ytr, model.predict(scaler.transform(Xtr)))
te_acc = accuracy_score(yte, model.predict(scaler.transform(Xte)))
print(f"  training accuracy   {tr_acc:.4f}")
print(f"  test accuracy       {te_acc:.4f}")
print(f"  generalisation gap  {tr_acc - te_acc:+.4f}")

print("\n" + "=" * 66)
print("  2. OUT-OF-BAG estimate — held-out by construction")
print("=" * 66)
oob = Pipeline([("s", StandardScaler()),
                ("m", RandomForestClassifier(n_estimators=200, max_depth=None,
                                             min_samples_leaf=1, oob_score=True,
                                             random_state=42, n_jobs=-1))])
oob.fit(Xtr, ytr)
print(f"  OOB score           {oob.named_steps['m'].oob_score_:.4f}")
print(f"  test accuracy       {te_acc:.4f}")
print("  (OOB uses the ~37% of rows each tree never saw. If the forest were")
print("   memorising, OOB would collapse while training stayed at 1.0.)")

print("\n" + "=" * 66)
print("  3. TREE CAPACITY — how much memorisation is even available")
print("=" * 66)
depths = [e.get_depth() for e in model.estimators_]
leaves = [e.get_n_leaves() for e in model.estimators_]
print(f"  trees               {len(model.estimators_)}")
print(f"  depth  min/mean/max {min(depths)} / {np.mean(depths):.1f} / {max(depths)}")
print(f"  leaves min/mean/max {min(leaves)} / {np.mean(leaves):.1f} / {max(leaves)}")
print(f"  training rows       {len(Xtr)}")
print(f"  mean rows per leaf  {len(Xtr)/np.mean(leaves):.1f}")

print("\n" + "=" * 66)
print("  4. VALIDATION CURVE over max_depth — does constraining help?")
print("=" * 66)
depth_grid = [3, 5, 8, 10, 15, 20, None]
tr_s, va_s = validation_curve(
    Pipeline([("s", StandardScaler()),
              ("m", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    Xtr, ytr, param_name="m__max_depth", param_range=depth_grid,
    cv=cv, scoring="accuracy", n_jobs=-1)
print(f"  {'max_depth':>10} {'train':>8} {'val (CV)':>10} {'gap':>8}")
for d, t, v in zip(depth_grid, tr_s.mean(1), va_s.mean(1)):
    print(f"  {str(d):>10} {t:8.4f} {v:10.4f} {t-v:8.4f}")

print("\n" + "=" * 66)
print("  5. LEARNING CURVE — would more data help?")
print("=" * 66)
sizes, tr_l, va_l = learning_curve(
    Pipeline([("s", StandardScaler()),
              ("m", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    Xtr, ytr, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=cv,
    scoring="accuracy", n_jobs=-1)
print(f"  {'n_train':>8} {'train':>8} {'val (CV)':>10} {'gap':>8}")
for n, t, v in zip(sizes, tr_l.mean(1), va_l.mean(1)):
    print(f"  {n:8d} {t:8.4f} {v:10.4f} {t-v:8.4f}")

print("\n" + "=" * 66)
print("  6. LABEL-SHUFFLE CONTROL — could it memorise noise if asked?")
print("=" * 66)
rng = np.random.RandomState(0)
y_shuf = rng.permutation(ytr)
p = Pipeline([("s", StandardScaler()),
              ("m", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
p.fit(Xtr, y_shuf)
sh_tr = accuracy_score(y_shuf, p.predict(Xtr))
sh_cv = cross_val_score(p, Xtr, y_shuf, cv=cv, scoring="accuracy", n_jobs=-1).mean()
print(f"  shuffled labels — training accuracy {sh_tr:.4f}")
print(f"  shuffled labels — CV accuracy       {sh_cv:.4f}  (chance = {1/22:.4f})")
print("  (High train + chance CV on shuffled labels = the model CAN memorise.")
print("   That it does NOT on real labels is the point.)")
