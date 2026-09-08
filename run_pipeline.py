import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import PipelineConfig
from src.data.dataset import MVTecDataset
from src.model.extractor import DINOv2Extractor
from src.model.patchcore import PatchCore
from src.evaluation.metrics import compute_category_metrics, compute_overall_metrics, save_results
from src.visualization.gradcam import visualize_batch
from src.tracking.mlflow_logger import log_pipeline_run


def run_category(
    category: str,
    config: PipelineConfig,
    extractor: DINOv2Extractor,
) -> dict:
    print(f"\n{'='*50}")
    print(f"Running category: {category.upper()}")
    print(f"{'='*50}")

    train_dataset = MVTecDataset(config.data_root, category, split="train")
    test_dataset = MVTecDataset(config.data_root, category, split="test")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    patchcore = PatchCore(extractor=extractor, coreset_ratio=config.coreset_ratio)
    patchcore.fit(train_loader)

    artifact_path = f"artifacts/{category}_memory_bank.pt"
    patchcore.save(artifact_path)

    all_labels: list[int] = []
    all_image_scores: list[float] = []
    all_patch_scores: list[torch.Tensor] = []
    all_paths: list[str] = []

    for batch in test_loader:
        images: torch.Tensor = batch["image"]
        labels: torch.Tensor = batch["label"]
        paths: list[str] = batch["path"]

        image_scores, patch_scores = patchcore.predict(images)

        all_labels.extend(labels.tolist())
        all_image_scores.extend(image_scores.tolist())
        all_patch_scores.extend([patch_scores[i] for i in range(len(images))])
        all_paths.extend(paths)

    result = compute_category_metrics(all_labels, all_image_scores, category)

    if config.visualize:
        vis_dir = f"{config.save_dir}/visualizations/{category}"
        visualize_batch(
            image_paths=all_paths,
            patch_scores_list=all_patch_scores,
            anomaly_scores=all_image_scores,
            labels=all_labels,
            save_dir=vis_dir,
            n_samples=config.n_samples_visualize
        )

    return result


def main() -> None:
    config = PipelineConfig()

    extractor = DINOv2Extractor(model_name=config.model_name)

    all_results = []

    for category in config.categories:
        result = run_category(
            category=category,
            config=config,
            extractor=extractor,
        )
        all_results.append(result)

    overall = compute_overall_metrics(all_results)

    save_results(
        category_results=all_results,
        overall=overall,
        save_path=f"{config.save_dir}/results.json"
    )

    log_pipeline_run(
        experiment_name=config.experiment_name,
        run_name=f"patchcore_{config.model_name}_coreset{config.coreset_ratio}",
        params=config.model_dump(),
        category_results=all_results,
        overall=overall,
        visualization_dir=f"{config.save_dir}/visualizations"
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
