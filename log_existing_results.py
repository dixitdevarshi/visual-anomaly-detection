import json
from src.tracking.mlflow_logger import log_pipeline_run

with open("experiments/results/results.json") as f:
    results = json.load(f)

params = {
    "coreset_ratio": 0.1,
    "batch_size": 16,
    "model": "dinov2_vitb14",
    "image_size": 224,
    "n_categories": 15
}

log_pipeline_run(
    experiment_name="visual-anomaly-detection",
    run_name="patchcore_dinov2_vitb14_coreset0.1",
    params=params,
    category_results=results["per_category"],
    overall=results["overall"],
    visualization_dir="experiments/results/visualizations"
)

print("Done. Run: mlflow ui")