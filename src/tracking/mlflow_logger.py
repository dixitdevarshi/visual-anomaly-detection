import mlflow
from pathlib import Path
from typing import List, Dict, Optional

TRACKING_URI = "sqlite:///mlflow.db"


def setup_experiment(experiment_name: str = "visual-anomaly-detection") -> str:
    """
    Create or load an MLflow experiment.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")

    return experiment_id


def log_pipeline_run(
    experiment_name: str,
    run_name: str,
    params: Dict,
    category_results: List[Dict],
    overall: Dict,
    visualization_dir: Optional[str] = None
):
    """
    Log a full pipeline run to MLflow.

    Args:
        experiment_name: MLflow experiment name
        run_name: name for this specific run
        params: pipeline parameters (coreset_ratio, batch_size, model etc.)
        category_results: list of per-category metric dicts
        overall: overall metrics dict
        visualization_dir: optional path to heatmap images to log as artifacts
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    setup_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # log parameters
        mlflow.log_params(params)

        # log overall metrics
        mlflow.log_metric("mean_auroc", overall["mean_auroc"])
        mlflow.log_metric("std_auroc", overall["std_auroc"])
        mlflow.log_metric("mean_average_precision", overall["mean_average_precision"])
        mlflow.log_metric("std_average_precision", overall["std_average_precision"])
        mlflow.log_metric("n_categories", overall["n_categories"])

        # log per-category metrics
        for result in category_results:
            cat = result["category"]
            mlflow.log_metric(f"{cat}_auroc", result["auroc"])
            mlflow.log_metric(f"{cat}_ap", result["average_precision"])

        # log heatmap visualizations as artifacts
        if visualization_dir and Path(visualization_dir).exists():
            for category_dir in Path(visualization_dir).iterdir():
                if category_dir.is_dir():
                    for img_file in category_dir.glob("*.png"):
                        mlflow.log_artifact(
                            str(img_file),
                            artifact_path=f"visualizations/{category_dir.name}"
                        )

        # log results json
        results_path = "experiments/results/results_all_15.json"
        if Path(results_path).exists():
            mlflow.log_artifact(results_path, artifact_path="results")

        print(f"MLflow run logged: {run_name}")
        print(f"Mean AUROC: {overall['mean_auroc']}")


def log_category_run(
    experiment_name: str,
    category: str,
    params: Dict,
    result: Dict
):
    """
    Log a single category run to MLflow.
    Useful for tracking individual category experiments.

    Args:
        experiment_name: MLflow experiment name
        category: MVTec category name
        params: pipeline parameters
        result: category metrics dict
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    setup_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{category}_coreset{params.get('coreset_ratio', 0.1)}"):
        mlflow.log_params({**params, "category": category})
        mlflow.log_metric("auroc", result["auroc"])
        mlflow.log_metric("average_precision", result["average_precision"])
        mlflow.log_metric("n_normal", result["n_normal"])
        mlflow.log_metric("n_anomaly", result["n_anomaly"])

        print(f"Logged {category}: AUROC {result['auroc']}")