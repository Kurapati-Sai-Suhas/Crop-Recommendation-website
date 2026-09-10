# -*- coding: utf-8 -*-
"""
============================================================
deploy/sagemaker/build_container_native_model.py
============================================================
Rebuild the served model using the library versions the SageMaker
container already ships, and write it to a separate artifact set.

Why this exists
---------------
The project trains on scikit-learn 1.4.2. The newest managed SageMaker
SKLearn container is 1.2-1, running Python 3.9 with scikit-learn 1.2.1.
A 1.4.2 pickle cannot be relied on in that container.

The obvious fix -- ship a requirements.txt so the container pip-installs
1.4.2 at startup -- was tried and failed twice, in two different ways:

  1. Pinning scikit-learn and numpy but not scipy upgraded numpy under
     the container's pre-compiled scipy. scipy's C extensions then failed
     to import, the model never loaded, and the endpoint died with a
     misleading "180 second timeout".

  2. Pinning all three let the model load, but invocation raised
     "Cannot convert numpy.ndarray to numpy.ndarray" from inside the
     forest's Cython code -- the signature of two numpy builds coexisting
     in one process.

Installing a different numpy into a container whose scientific stack was
compiled against another one is fighting the image. The robust answer is
to match it: train with 1.2.1 and install nothing at runtime.

That also makes the endpoint start far faster, because there is no pip
step inside the 180-second cold-start budget at all.

Run with the 3.9 environment, not the project's:
  .venv-sm/Scripts/python deploy/sagemaker/build_container_native_model.py
============================================================
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(os.path.dirname(HERE))
DATA_PATH = os.path.join(ROOT, "data", "crop_data.csv")
OUT_DIR   = os.path.join(HERE, "artifacts")
MAIN_DIR  = os.path.join(ROOT, "backend", "models")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

# Identical to the main pipeline, so this is the same model expressed under
# an older library -- not a differently-tuned one.
RANDOM_STATE = 42
TEST_SIZE    = 0.2
BEST_PARAMS  = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1}


def main():
    print("=" * 62)
    print("  Container-native model build")
    print("=" * 62)
    print(f"python       {sys.version.split()[0]}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"numpy        {np.__version__}")

    if not sklearn.__version__.startswith("1.2"):
        print("\n[ABORT] This must run under scikit-learn 1.2.x to match the "
              "SageMaker container. Use the .venv-sm environment.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    X = df[FEATURES].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"\n[SPLIT] train {len(X_train)} | test {len(X_test)} "
          f"(same seed and stratification as the main pipeline)")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"[EVAL]  accuracy {acc:.4f} | F1 {f1:.4f}")

    # Compare against the 1.4.2 model this replaces. A large gap would mean
    # the two are not the same model, which would make the endpoint and the
    # web service disagree.
    main_metrics = os.path.join(MAIN_DIR, "metrics.json")
    if os.path.exists(main_metrics):
        with open(main_metrics, encoding="utf-8") as handle:
            reference = json.load(handle)["RandomForest"]["accuracy"]
        delta = acc - reference
        print(f"[CHECK] scikit-learn 1.4.2 model scored {reference:.4f} "
              f"-> delta {delta:+.4f}")
        if abs(delta) > 0.01:
            print("[WARN]  The two models differ by more than a point. "
                  "Investigate before serving this one.")
        else:
            print("[CHECK] Within a point -- same model, older library.")

    joblib.dump(pipe.named_steps["model"],
                os.path.join(OUT_DIR, "RandomForest.joblib"))
    joblib.dump(pipe.named_steps["scaler"],
                os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(OUT_DIR, "label_encoder.joblib"))

    provenance = {
        "built_with": {
            "python":       sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy":        np.__version__,
            "joblib":       joblib.__version__,
        },
        "target_container": "SageMaker SKLearn 1.2-1 (py3)",
        "hyperparameters":  BEST_PARAMS,
        "random_state":     RANDOM_STATE,
        "test_accuracy":    round(float(acc), 4),
        "test_f1":          round(float(f1), 4),
        "note": ("Built to match the container's library versions so nothing "
                 "is pip-installed at startup. The project's own service uses "
                 "backend/models/, built with scikit-learn 1.4.2."),
    }
    with open(os.path.join(OUT_DIR, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print(f"\n[SAVED] {OUT_DIR}")
    for name in ["RandomForest.joblib", "scaler.joblib",
                 "label_encoder.joblib", "provenance.json"]:
        size = os.path.getsize(os.path.join(OUT_DIR, name)) / 1024
        print(f"        {name:26s} {size:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
