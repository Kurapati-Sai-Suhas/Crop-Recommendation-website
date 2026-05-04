"""
============================================================
mlops/mlflow/mlflow_config.py
============================================================
MLflow configuration and utility helpers.

MLflow tracks:
  - Parameters (model hyperparameters)
  - Metrics (accuracy, F1, etc.)
  - Artifacts (model files)
  - Model versions (via Model Registry)

View the MLflow UI:
  mlflow ui --backend-store-uri ./mlops/mlruns --port 5000
  Open: http://localhost:5000
============================================================
"""

import os
import mlflow
import mlflow.sklearn

# ─── MLflow configuration ──────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLRUNS_DIR    = os.path.join(BASE_DIR, "mlops", "mlruns")
EXPERIMENT_NAME = "CropRecommendation"


def setup_mlflow() -> None:
    """Configure MLflow to store runs locally in mlops/mlruns/."""
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"📊 MLflow tracking URI: {MLRUNS_DIR}")
    print(f"   Experiment: {EXPERIMENT_NAME}")


def log_training_run(model_name: str, params: dict, metrics: dict, model) -> str:
    """
    Log a training run to MLflow.

    Args:
        model_name: Name of the model (e.g., "RandomForest")
        params:     Hyperparameters dict
        metrics:    Evaluation metrics dict
        model:      Trained sklearn model object

    Returns:
        run_id: MLflow run ID (useful for model versioning)
    """
    setup_mlflow()

    with mlflow.start_run(run_name=model_name) as run:
        # Log hyperparameters
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log model + register in Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=f"CropRecommender_{model_name}",
        )

        run_id = run.info.run_id
        print(f"   📝 MLflow run logged: {run_id}")
        return run_id


def get_best_model_version(model_name: str) -> dict:
    """
    Query the MLflow Model Registry for the latest version of a model.

    Returns:
        {"version": "3", "stage": "Production", "run_id": "..."}
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file:///{MLRUNS_DIR}")
    registered_name = f"CropRecommender_{model_name}"

    try:
        versions = client.get_latest_versions(registered_name)
        if versions:
            latest = versions[-1]
            return {
                "version": latest.version,
                "stage":   latest.current_stage,
                "run_id":  latest.run_id,
            }
    except Exception as e:
        print(f"⚠️ Could not fetch model version: {e}")

    return {}


def list_all_runs() -> list:
    """Return a list of all MLflow runs for the experiment."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )
    return [
        {
            "run_id":    r.info.run_id,
            "run_name":  r.data.tags.get("mlflow.runName", ""),
            "accuracy":  r.data.metrics.get("accuracy", 0),
            "f1_score":  r.data.metrics.get("f1_score", 0),
            "status":    r.info.status,
        }
        for r in runs
    ]
