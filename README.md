# 🌾 Crop Recommendation System — Explainable AI + MLOps

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange?logo=mlflow)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-yellow)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

> A crop recommendation system that tells you **what** to plant **and why**, backed by Explainable AI (SHAP), an MLOps pipeline, and a modern React dashboard.

---

> ### ⚠️ Status: portfolio project — not for real planting decisions
>
> The models are trained on the public **Crop Recommendation Dataset**
> (2,200 rows, 22 crops), fetched and checksum-verified by
> `data/download_data.py`. Read the accuracy figures with two caveats:
>
> **1. The task is close to solved by construction.** An untuned linear
> discriminant scores **0.967** and 1-NN scores **0.974** in 5-fold CV; the
> grid-searched Random Forest reaches **0.9955** on the held-out test set.
> The gap between the top models (+0.0011 CV) is smaller than the
> fold-to-fold standard deviation (0.0058), so "which model is best" is not
> a question this dataset can answer. A ~99% headline here is a property of
> the data, not evidence of modelling skill.
>
> **2. The dataset is a benchmark, not field measurements.** Each crop's
> feature values are tightly bounded with no tails — zero within-class IQR
> outliers across all seven features, where ~15 would be expected from
> measured data — which indicates it was at least partly generated. See
> [`eda/EDA_REPORT.md`](eda/EDA_REPORT.md).
>
> The engineering is real and works: leakage-safe pipelines, cross-validated
> selection, SHAP attribution, model serving, drift monitoring. The agronomy
> is not validated. **Do not use this system to decide what to plant.**

---

## 📸 Features

| Feature | Details |
|---------|---------|
| 🤖 ML Models | Random Forest (primary), Logistic Regression, Naive Bayes |
| 🧠 Explainability | SHAP TreeExplainer — per-feature Shapley values + human text |
| ⚡ Backend | FastAPI + Pydantic validation + Uvicorn |
| 🎨 Frontend | React 18 + Tailwind CSS + Chart.js (dark agricultural theme) |
| 📊 MLOps | MLflow experiment tracking + DVC data versioning |
| 🐳 Deployment | Docker + docker-compose + GitHub Actions CI/CD |
| 🔒 Monitoring | Prediction logging (JSONL) + basic drift detection |

---

## 🗂️ Project Structure

```
crop-recommendation/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          ← All API endpoints
│   │   └── schemas.py         ← Pydantic validation models
│   ├── models/                ← Trained .joblib files (auto-generated)
│   ├── services/
│   │   ├── train.py           ← Training pipeline
│   │   ├── predictor.py       ← Inference service
│   │   ├── explainer.py       ← SHAP XAI service
│   │   └── monitoring.py      ← Prediction logging + drift
│   ├── errors.py              ← Typed application exceptions
│   └── main.py                ← FastAPI entry point
│
├── frontend/
│   ├── src/
│   │   ├── api/client.js      ← Axios API client
│   │   ├── components/        ← Reusable UI components
│   │   ├── pages/             ← Home + Dashboard pages
│   │   └── index.css          ← Tailwind + design tokens
│   ├── package.json
│   └── vite.config.js
│
├── mlops/
│   └── dvc.yaml               ← DVC pipeline stages (see Known Limitations)
│
├── data/
│   ├── crop_data.csv          ← Dataset (downloaded, checksum-verified)
│   └── download_data.py       ← Fetch + verify the real dataset
│
├── eda/
│   ├── run_eda.py             ← Exploratory analysis (regenerates everything)
│   ├── EDA_REPORT.md          ← Findings, all figures computed at run time
│   └── figures/               ← Generated plots
│
├── tests/
│   ├── test_api.py            ← FastAPI endpoint tests
│   └── test_model.py          ← Model accuracy tests
│
├── logs/                      ← Prediction logs (auto-created)
├── train_pipeline.py          ← Master training entrypoint
├── requirements.txt
├── Dockerfile                 ← Backend container
├── frontend/Dockerfile.frontend
├── docker-compose.yml
├── .github/workflows/ci_cd.yml
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

The backend reads these environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CROP_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowlist of browser origins. `*` is rejected — a wildcard makes the server reflect any caller's origin. |
| `CROP_LOG_LEVEL` | `INFO` | Python logging level. |

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.12+
- Node.js 18+
- Git

### 1. Clone & Install Backend

```bash
cd crop-recommendation

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Train Models (one-time setup)

```bash
# Downloads + verifies the dataset, then trains, tunes and evaluates
python train_pipeline.py
```

Actual output:
```
[DATA] 2200 rows x 8 cols | 22 crops
       missing cells: 0 | duplicate rows: 0
       class balance: min 100, max 100 per crop -- balanced
[SPLIT] train 1760 | test 440 (stratified)

  STAGE 1 - baseline comparison (5-fold CV, train only)
  RandomForest         0.9926 +/- 0.0058
  LogisticRegression   0.9682 +/- 0.0066
  NaiveBayes           0.9949 +/- 0.0042

  STAGE 2 - hyperparameter tuning (GridSearchCV, train only)
  RandomForest: 18 configs x 5 folds = 90 fits
    -> best CV 0.9938 | {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}

  STAGE 3 - held-out test set (touched once)
  RandomForest         acc 0.9955 | F1 0.9955 (CV was 0.9938)
  LogisticRegression   acc 0.9841 | F1 0.9840 (CV was 0.9778)
  NaiveBayes           acc 0.9955 | F1 0.9954 (CV was 0.9949)

  MODEL SELECTION
  highest CV     : NaiveBayes (0.9949)
  served model   : RandomForest (0.9938)
  gap            : +0.0011 vs fold std 0.0058
  -> within fold noise: statistically indistinguishable.
     Keeping RandomForest for exact TreeSHAP attribution.

[ERRORS] 2 misclassified of 440 test rows
         blackgram    -> maize        x1
         rice         -> jute         x1
[PERMUTATION IMPORTANCE] humidity 0.320, N 0.229, rainfall 0.179, K 0.167
```

Read those figures against the disclaimer above. The interesting line is not
the accuracy — it is the model-selection block admitting the top two models
cannot be told apart on this data.

### 3. Start Backend API

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Install & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: [http://localhost:5173](http://localhost:5173)

### 5. (Optional) Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlops/mlruns --port 5000
```

MLflow dashboard: [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker Deployment

```bash
# Build and start all 3 services
docker-compose up --build

# Services:
#   Frontend  → http://localhost:3000
#   Backend   → http://localhost:8000
#   MLflow UI → http://localhost:5000
```

---

## 📡 API Reference

### `POST /api/v1/predict`
Get crop recommendation from all 3 models.

**Request body:**
```json
{
  "N": 80,
  "P": 40,
  "K": 40,
  "temperature": 23.0,
  "humidity": 82.0,
  "ph": 6.0,
  "rainfall": 200.0
}
```

**Response:**
```json
{
  "primary_prediction": {
    "crop": "rice",
    "confidence": 0.97,
    "model_used": "RandomForest"
  },
  "all_models": {
    "RandomForest":       { "crop": "rice", "confidence": 0.97 },
    "LogisticRegression": { "crop": "rice", "confidence": 0.91 },
    "NaiveBayes":         { "crop": "rice", "confidence": 0.99 }
  },
  "input_summary": { "N": 80, "P": 40, "K": 40, ... }
}
```

---

### `POST /api/v1/explain`
Get SHAP explanation for the prediction.

**Response:**
```json
{
  "crop": "rice",
  "shap_values": [
    { "feature": "N", "label": "Nitrogen (N)", "shap_value": 0.28, "input_value": 80, "direction": "positive" },
    ...
  ],
  "text_explanation": "High Nitrogen (N) (80) and Rainfall (mm) (200 mm) strongly influenced the recommendation of **rice**.",
  "top_features": [
    { "feature": "Nitrogen (N)", "importance": 0.28 },
    ...
  ],
  "summary_plot_base64": "<base64 PNG>"
}
```

---

### `GET /api/v1/metrics`
Returns evaluation metrics for all 3 models.
`503` if the stored metrics file is malformed or incomplete.

### `GET /api/v1/health`
Liveness — `200` whenever the process is serving.

### `GET /api/v1/health/ready`
Readiness — `200` only when models are loaded and a prediction can actually be
served, `503` otherwise. This is what the container healthcheck probes.

### `GET /api/v1/monitoring`
Returns recent prediction logs + basic drift report.

### Error responses

| Status | Meaning |
|--------|---------|
| `422` | Input failed validation (out of range, missing, non-numeric, or non-finite such as `NaN`/`Infinity`). |
| `503` | Models are unavailable. **The API never invents a placeholder crop** — if it cannot predict, it says so. |
| `500` | Unexpected failure. The body carries only a correlation ID, also returned as the `X-Request-ID` header; the detail is in the server log. |

---

## 🧪 Running Tests

```bash
# Make sure backend models are trained first
python train_pipeline.py

# Run full test suite
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_with_valid_input PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_invalid_input_returns_422 PASSED
tests/test_model.py::TestModelAccuracy::test_rf_accuracy_above_90_percent PASSED
...
```

---

## 📊 Input Features

| Feature | Unit | Range | Description |
|---------|------|-------|-------------|
| N | ratio | 0–200 | Nitrogen content in soil |
| P | ratio | 0–200 | Phosphorus content in soil |
| K | ratio | 0–210 | Potassium content in soil |
| temperature | °C | 0–50 | Average temperature |
| humidity | % | 0–100 | Relative humidity |
| ph | — | 0–14 | Soil pH value |
| rainfall | mm | 0–500 | Annual rainfall |

---

## 🌱 Supported Crops (22 classes)

`rice` · `maize` · `chickpea` · `kidneybeans` · `pigeonpeas` · `mothbeans` · `mungbean` · `blackgram` · `lentil` · `pomegranate` · `banana` · `mango` · `grapes` · `watermelon` · `muskmelon` · `apple` · `orange` · `papaya` · `coconut` · `cotton` · `jute` · `coffee`

---

## 🔬 MLOps Architecture

```
Download + verify → DVC tracking → CV + tuning → Test eval → MLflow
                                      ↓
                              Model Registry
                                      ↓
                        FastAPI Inference Server
                                      ↓
                   SHAP Explanation → Monitoring Logs
```

### MLflow tracks

Each training run logs one MLflow run per model to `mlops/mlruns/`:

| Logged | Why it is there |
|---|---|
| `cv_accuracy_untuned` | the baseline before any tuning |
| `cv_accuracy_tuned` | what the grid search actually bought |
| `test_accuracy` / `precision` / `recall` / `f1` | the single held-out evaluation |
| best hyperparameters | the configuration that produced those numbers |

Logging **both** the untuned and tuned CV scores is the point. A run that
records only the final number cannot answer "was the tuning worth it?" — here
it shows that grid search moved Logistic Regression 0.9682 → 0.9778 and the
Random Forest only 0.9926 → 0.9938.

```bash
mlflow ui --backend-store-uri ./mlops/mlruns --port 5000
```

Verified: 3 runs logged and rendering in the UI. Note that `mlflow==2.12.2`
imports `pkg_resources`, removed in setuptools 81+, so `setuptools<81` is
pinned in `requirements.txt`. Without that pin `import mlflow` fails outright
with `ModuleNotFoundError: No module named 'pkg_resources'`.

### DVC Manages:
- `data/crop_data.csv` versioning
- Pipeline stages: `download_data → train → test`
- Reproducible runs with `dvc repro`

---

## 🔬 Modelling protocol

The training pipeline (`backend/services/train.py`) runs four stages. The
ordering is the point: every decision is made before the test set is opened.

| Stage | What happens | Data used |
|---|---|---|
| 1. Baseline comparison | 5-fold stratified CV over three models with different inductive biases | training split only |
| 2. Tuning | `GridSearchCV` on the same folds | training split only |
| 3. Selection | Best cross-validated accuracy, with an explicit tie-break | training split only |
| 4. Final evaluation | Scored **once** | held-out test split |

**Leakage control.** Scaling happens inside a scikit-learn `Pipeline`, so the
`StandardScaler` is refit within every CV fold. Fitting one scaler on the
whole training set before cross-validating would leak each fold's validation
statistics into its own training data.

**Why these three models.** Not three variations on one idea — three
different assumptions. `GaussianNB` is generative and assumes per-class
independent Gaussians; `LogisticRegression` draws linear boundaries;
`RandomForest` is non-linear and models interactions. Comparing them answers
a real question: does this problem need a non-linear boundary?

### Results

| Model | CV (untuned) | CV (tuned) | Test accuracy | Test F1 |
|---|---|---|---|---|
| RandomForest | 0.9926 ± 0.0058 | 0.9938 | **0.9955** | 0.9955 |
| LogisticRegression | 0.9682 ± 0.0066 | 0.9778 | **0.9841** | 0.9840 |
| NaiveBayes | 0.9949 ± 0.0042 | 0.9949 | **0.9955** | 0.9954 |

### Which model is served, and why

`NaiveBayes` scored highest in cross-validation
(0.9949), but its lead over `RandomForest`
is **+0.0011** against a fold standard deviation of
**0.0058**. That gap is noise, not a result.

`RandomForest` is served — not because it scored higher, but because
the product requires per-prediction SHAP values, and `TreeExplainer` computes
those exactly for a forest. `GaussianNB` would need `KernelExplainer`:
approximate, and orders of magnitude slower. **The tie-break is documented
because "accuracy chose it" would be a false statement about a 0.1pp gap.**

### Error analysis

2 of 440 test rows are misclassified:

- `blackgram` → `maize` ×1
- `rice` → `jute` ×1

The rice ↔ jute confusion is the one that persists across models and shows up
in EDA too (rice recall drops to 0.76 under a linear discriminant). Both are
wet-season crops of the same river-delta conditions and their
humidity/rainfall envelopes genuinely overlap. **This is a property of the
world, not a modelling defect** — separating them needs a feature the dataset
does not contain: soil texture, season, or geography.

### What the model relies on

Permutation importance on the test set, preferred over the forest's built-in
`feature_importances_`, which is impurity-based and biased toward
high-cardinality continuous features:

- **humidity** — 0.320 ± 0.018
- **N** — 0.229 ± 0.018
- **rainfall** — 0.179 ± 0.011
- **K** — 0.167 ± 0.013
- **P** — 0.108 ± 0.015

Full detail, including the per-class report and confusion matrix, is written
to `backend/models/evaluation_report.json` on every training run.

---

## 🔍 Is it overfitting?

```bash
python eda/overfitting_check.py
```

Training accuracy is **1.0000**, which looks alarming and is the expected
behaviour of a forest grown with `min_samples_leaf=1` — every tree is grown to
purity on its bootstrap sample. What matters is the gap and whether
constraining capacity would help. Six checks say no:

| Check | Result |
|---|---|
| Train vs test | 1.0000 vs 0.9955 — **gap +0.0045** |
| Out-of-bag score | **0.9949**, within 0.0006 of test |
| `max_depth` 15 / 20 / None | all CV 0.9938 — constraining does not help |
| Learning-curve gap | 0.0415 → 0.0062 as training data grows |
| Shuffled labels | train 1.0000, CV **0.0381** (chance 0.0455) |
| Tree size | ~75 leaves for 22 classes, 23 rows per leaf |

The out-of-bag score is the most convincing single number: it is computed
during fitting from the ~37% of rows each tree never saw, and it lands within
0.0006 of the held-out test score. A memorising forest would show training
accuracy at 1.0 with OOB collapsing.

The shuffled-label control is the one that settles it. Given permuted labels
the same model still reaches 1.0000 on training data but scores at chance on
CV — so it *can* memorise noise, and demonstrably is not doing so on the real
labels.

**A separate concern, often confused with this one:** if 0.9955 seems too good,
the cause is not the model — an untuned linear discriminant reaches 0.967 on
this data. That is a property of the dataset, not overfitting, and it is
covered under *Exploratory analysis* below.

---

## 📊 Exploratory analysis

```bash
python eda/run_eda.py
```

Regenerates every figure and rewrites [`eda/EDA_REPORT.md`](eda/EDA_REPORT.md).
Nothing in that report is typed in by hand — every number is computed from the
dataset at run time, so it cannot drift out of sync with the data.

Headline findings:

- **2,200 rows, 7 numeric features, 22 crops, exactly 100 samples each.** No
  missing values, no duplicate rows. Perfect balance means plain accuracy is a
  fair headline metric and resampling would be unjustified.
- **No outlier removal is applied, deliberately.** K flags 9% of rows as
  global IQR outliers and *zero* within their own crop. Those points are the
  high-potassium crops sitting where agronomy says they should; a global
  filter would delete the signal that separates the classes.
- **P ↔ K correlate at +0.74** — agronomically real (compound fertiliser).
  Every other pair sits below |0.3|, so there is nothing for PCA to gain.
- **A depth-5 tree scores only 0.409, but LDA scores 0.967.** Separability
  does not come from any single feature crossing a threshold — it comes from
  the joint configuration of all seven.

---

## 🚢 Deployment

One container serves the React UI and the FastAPI API on a single port.
Full instructions, the cloud-platform comparison, and the SageMaker / Vertex AI
mapping are in [`deploy/DEPLOY.md`](deploy/DEPLOY.md).

```
Browser → Container ─┬─ /          React bundle
                     ├─ /docs      OpenAPI UI
                     └─ /api/v1/*  validate → scale → predict → SHAP → log
```

The image bakes in the trained artifacts. It **never trains at build time**
and **never reads the dataset** — a deploy ships a model that was reviewed,
not one produced silently during a build.

```bash
python deploy/build_space.py --check   # verify artifacts exist
python deploy/build_space.py           # stage .space-build/
```

Recommended target is **Hugging Face Spaces** (free, no card, public HTTPS).
Google Cloud Run is a good alternative if you already have GCP billing.
SageMaker and Vertex endpoints bill hourly for an idle node and are not
worth it for a portfolio project — `deploy/DEPLOY.md` explains how this
architecture maps onto them regardless.

---

## 🔁 Retraining

```bash
python retrain.py --dry-run              # validate new data, change nothing
python retrain.py --data path/to/new.csv # retrain
python retrain.py --rollback             # restore the previous model set
```

Nothing schedules this. It runs when something actually changed — the drift
report shows a sustained shift, or new labelled data arrives. Retraining on a
timetable against unchanged data burns compute and adds risk for no gain.

The gate reuses the rule the training pipeline applies to model selection: a
difference smaller than the CV fold standard deviation is noise, and noise is
not a reason to ship. Three outcomes:

| Candidate vs incumbent | Action |
|---|---|
| better by more than fold noise | promote |
| inside fold noise | **keep the incumbent** — a coin flip is not an upgrade |
| worse by more than fold noise | roll back |

Every run appends to `backend/models/retrain_history.json`. Data is validated
before any compute is spent: missing columns, non-numeric features, too few
samples per class for a stratified 5-fold split, and values far outside the
previous training range are all reported, and the fatal ones abort the run.

---

## 📝 Viva / Presentation Notes

**Q: Why SHAP over LIME?**
SHAP uses Shapley values from game theory — mathematically guaranteed to be fair and consistent. TreeExplainer is also 1000× faster than model-agnostic SHAP for Random Forests.

**Q: Why Random Forest as primary model?**
Not for accuracy — GaussianNB actually edged it in cross-validation
(0.9949 vs 0.9938), and that gap is smaller than the fold standard deviation
(0.0058), so the two are statistically indistinguishable here. RF is served
because the product requires per-prediction SHAP values and `TreeExplainer`
computes those exactly for a forest, where GaussianNB would need the
approximate, far slower `KernelExplainer`. The rationale is recorded in
`evaluation_report.json` under `model_selection`.

**Q: Why `tree_path_dependent` rather than interventional SHAP?**
The interventional estimator broke `/explain` on every request. SHAP checks
additivity across all 22 classes at once, and summing float contributions
over 200 trees of depth ~23 leaves ~1e-4 of residue on the near-zero-probability
classes, which trips the check — even though the predicted class's own
attributions were accurate to ~1e-9. `tree_path_dependent` satisfies
additivity to ~1e-16, needs no background set, and is faster. The trade-off
is that it conditions on tree structure, so credit can be shared between
correlated features — which here means only P and K (r = 0.74).

**Q: What is the MLflow experiment tracking?**
Each training run logs one run per model to a local file store, with the best
hyperparameters and both the untuned and tuned cross-validation scores
alongside the held-out test metrics. Logging both CV numbers is deliberate: it
makes "was the tuning worth it?" answerable from the tracking UI instead of
requiring a rerun. It also means the claim in this README can be checked
against the recorded runs rather than taken on trust.

**Q: How does monitoring work?**
Every prediction is appended to `logs/predictions.jsonl`. The `/monitoring` endpoint computes z-score drift by comparing recent prediction input distributions against training data statistics.

---

## ⚠️ Known limitations

Tracked, not hidden. These are real and currently unfixed:

| Area | Limitation |
|------|------------|
| **Data** | The dataset is a public benchmark with no tails in any per-class distribution (zero within-class IQR outliers), which indicates it was at least partly generated. No validation against field observations has been done. |
| **Evaluation** | 5-fold CV, grid search and a single held-out test evaluation are in place, with confusion matrix and per-class report in `evaluation_report.json`. Still missing: ROC-AUC and a calibration check. A random split also cannot measure generalisation to a *new region*, which is the deployment question that would actually matter. |
| **Confidence** | `predict_proba` is uncalibrated (measured ECE ≈ 0.05) and systematically under-confident. The UI's High/Medium/Low bands are not derived from measured reliability. |
| **Out-of-distribution input** | Physically absurd inputs (pH 0, all-zero soil) still return a confident crop. There is no novelty detection or abstain path. |
| **Feature importance** | `/feature-importance` still serves the forest's impurity-based importance, which is biased toward high-cardinality continuous features. The training pipeline now computes permutation importance (in `evaluation_report.json`) and it ranks the features differently; the endpoint has not been switched over. |
| **DVC** | `mlops/dvc.yaml` is not runnable as written — its paths are root-relative but the file lives in `mlops/`, and the repo has no initialised `.dvc/`. |
| **Scale** | Prediction logs are a local JSONL file with no rotation, so the service is not yet safe to run as multiple replicas. |
| **Security** | No authentication, no rate limiting, and no security headers. `/monitoring` is publicly readable. |
| **Dependencies** | `mlflow==2.12.2` imports `pkg_resources`, removed in setuptools 81+, so `setuptools<81` is pinned. Confirmed by reproducing the failure: with setuptools 84, `import mlflow` raises `ModuleNotFoundError`. Upgrading mlflow to ≥3.11.1 removes both the pin and 43 known advisories. |

---

## 📄 License

MIT — see [LICENSE](LICENSE).
