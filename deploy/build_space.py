# -*- coding: utf-8 -*-
"""
============================================================
deploy/build_space.py
============================================================
Assemble a Hugging Face Space from this repository.

A Space is its own git repo and expects a Dockerfile plus a README
carrying YAML frontmatter at its root. This project's root already
has a different Dockerfile (for local development) and a README with
no frontmatter, so rather than contort the repo to match the host,
this script stages a clean deployment directory containing only what
the running service needs.

Deliberately not copied:
  data/    -- nothing at serving time reads the dataset
  eda/     -- analysis output, not runtime code
  tests/   -- run in CI, not in the image
  mlops/   -- experiment tracking, not serving

The trained model artifacts under backend/models/ ARE copied. The
image does not retrain at build time: a deploy should ship a model
that was reviewed, not silently produce a new one nobody evaluated.

Usage:
  python deploy/build_space.py                 # stage into .space-build/
  python deploy/build_space.py --check         # verify prerequisites only
============================================================
"""

import argparse
import os
import shutil
import sys

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE    = os.path.join(ROOT, ".space-build")
DEPLOY   = os.path.join(ROOT, "deploy")

REQUIRED_ARTIFACTS = [
    "RandomForest.joblib", "LogisticRegression.joblib", "NaiveBayes.joblib",
    "scaler.joblib", "label_encoder.joblib", "metrics.json",
    "monitoring_baseline.json",
]

SPACE_README = """---
title: Crop Recommendation
emoji: 🌾
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Crop Recommendation System

Recommends one of 22 crops from seven soil and climate measurements, and
explains each recommendation with SHAP attributions rather than returning a
bare label.

- **UI** — the dashboard at `/`
- **API docs** — [`/docs`](/docs)
- **Health** — [`/api/v1/health/ready`](/api/v1/health/ready)

## Read the accuracy carefully

Trained on the public Crop Recommendation Dataset (2,200 rows, 22 crops,
perfectly balanced). An untuned linear discriminant scores **0.967** in 5-fold
cross-validation; the grid-searched Random Forest reaches **0.9955** on a
held-out test set. The gap between the leading models (+0.0011 CV) is smaller
than the fold standard deviation (0.0058), so this dataset cannot support a
claim that one model beats another.

The served model is the Random Forest — chosen not for accuracy, but because
the app needs exact per-prediction SHAP values, which `TreeExplainer` provides
for a forest.

**The agronomy is not validated. Do not use this to decide what to plant.**

Source and full methodology:
<https://github.com/Kurapati-Sai-Suhas/Crop-Recommendation-website>
"""


def check():
    """Verify everything the image needs is present before staging."""
    problems = []

    models = os.path.join(ROOT, "backend", "models")
    for name in REQUIRED_ARTIFACTS:
        if not os.path.exists(os.path.join(models, name)):
            problems.append(f"Missing model artifact: backend/models/{name}")

    for path in ["requirements.txt", "backend/main.py",
                 "frontend/package.json", "frontend/package-lock.json",
                 "deploy/Dockerfile"]:
        if not os.path.exists(os.path.join(ROOT, path)):
            problems.append(f"Missing: {path}")

    if problems:
        print("[CHECK] Not ready to deploy:")
        for p in problems:
            print(f"  [ERROR] {p}")
        print("\nRun `python train_pipeline.py` if model artifacts are missing.")
        return False

    print("[CHECK] All prerequisites present")
    return True


def stage():
    """Copy the deployable subset into .space-build/."""
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # Backend, excluding caches and the previous-model backups.
    shutil.copytree(
        os.path.join(ROOT, "backend"),
        os.path.join(STAGE, "backend"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "models_previous"),
    )

    # Frontend source; the bundle is built inside the image.
    shutil.copytree(
        os.path.join(ROOT, "frontend"),
        os.path.join(STAGE, "frontend"),
        ignore=shutil.ignore_patterns("node_modules", "dist"),
    )

    shutil.copy2(os.path.join(ROOT, "requirements.txt"), STAGE)
    shutil.copy2(os.path.join(DEPLOY, "Dockerfile"),
                 os.path.join(STAGE, "Dockerfile"))

    with open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(SPACE_README)

    with open(os.path.join(STAGE, ".gitignore"), "w", encoding="utf-8") as handle:
        handle.write("__pycache__/\n*.pyc\nnode_modules/\ndist/\nlogs/\n")

    total = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _dirs, files in os.walk(STAGE)
        for f in files
    )
    n_files = sum(len(files) for _d, _s, files in os.walk(STAGE))

    print(f"[STAGE] {STAGE}")
    print(f"        {n_files} files, {total / 1_048_576:.1f} MB")
    print()
    print("Next — create the Space at https://huggingface.co/new-space")
    print("  (SDK: Docker, Blank template), then:")
    print()
    print("    cd .space-build")
    print("    git init -b main")
    print("    git add -A")
    print('    git commit -m "Deploy crop recommendation service"')
    print("    git remote add space \\")
    print("      https://huggingface.co/spaces/<your-username>/crop-recommendation")
    print("    git push --force space main")
    print()
    print("The first build takes ~5-8 minutes. Watch the Space's Logs tab.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage a Hugging Face Space")
    parser.add_argument("--check", action="store_true",
                        help="Verify prerequisites without staging")
    args = parser.parse_args()

    print("=" * 62)
    print("  Staging deployment bundle")
    print("=" * 62)

    if not check():
        return 1
    if args.check:
        return 0
    return 0 if stage() else 1


if __name__ == "__main__":
    sys.exit(main())
