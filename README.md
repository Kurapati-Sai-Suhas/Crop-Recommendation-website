# 🌾 Crop Recommendation System — Explainable AI + MLOps

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange?logo=mlflow)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-yellow)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

> A **production-ready** crop recommendation system that tells you **what** to plant **and why**, backed by Explainable AI (SHAP), a full MLOps pipeline, and a modern React dashboard.

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
│   ├── mlflow/mlflow_config.py
│   ├── pipelines/training_pipeline.py
│   └── dvc.yaml               ← DVC pipeline stages
│
├── data/
│   ├── crop_data.csv          ← Auto-generated dataset
│   └── generate_data.py       ← Synthetic data generator
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
└── README.md
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
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
# Generates dataset + trains RF, LR, NB + saves models + logs to MLflow
python train_pipeline.py
```

You'll see output like:
```
✅ Dataset generated: 2200 rows
✅ RandomForest   | Acc: 0.9841 | F1: 0.9840
✅ LogisticRegression | Acc: 0.9523 | F1: 0.9519
✅ NaiveBayes     | Acc: 0.9977 | F1: 0.9977
✅ Verification prediction: RICE (confidence: 97.00%)
```

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

### `GET /api/v1/health`
Health check — confirms models are loaded.

### `GET /api/v1/monitoring`
Returns recent prediction logs + basic drift report.

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
Data Generation → DVC tracking → Model Training → MLflow logging
                                      ↓
                              Model Registry
                                      ↓
                        FastAPI Inference Server
                                      ↓
                   SHAP Explanation → Monitoring Logs
```

### MLflow Tracks:
- Hyperparameters per model
- Accuracy, Precision, Recall, F1 per run
- Model artifacts (saved to Registry)
- Run history (compare all experiments)

### DVC Manages:
- `data/crop_data.csv` versioning
- Pipeline stages: `generate_data → train → test`
- Reproducible runs with `dvc repro`

---

## 🚢 Deployment

### Backend → Render / Railway
1. Push to GitHub
2. Connect Render to your repo
3. Set build command: `pip install -r requirements.txt && python train_pipeline.py`
4. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Frontend → Vercel / Netlify
1. Set `VITE_API_URL=https://your-backend.onrender.com`
2. Build command: `npm run build`
3. Publish directory: `dist`

---

## 📝 Viva / Presentation Notes

**Q: Why SHAP over LIME?**
SHAP uses Shapley values from game theory — mathematically guaranteed to be fair and consistent. TreeExplainer is also 1000× faster than model-agnostic SHAP for Random Forests.

**Q: Why Random Forest as primary model?**
RF achieves ~98% accuracy on this dataset, handles non-linear feature interactions, and works natively with SHAP's fast TreeExplainer.

**Q: What is the MLflow experiment tracking?**
Every training run logs hyperparameters + metrics + model artifact to a local SQLite database viewable in the MLflow UI. Enables reproducibility and model versioning.

**Q: How does monitoring work?**
Every prediction is appended to `logs/predictions.jsonl`. The `/monitoring` endpoint computes z-score drift by comparing recent prediction input distributions against training data statistics.

---

## 📄 License

MIT — free for academic and personal use.
