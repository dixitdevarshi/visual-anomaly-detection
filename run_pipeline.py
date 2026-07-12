import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset, MVTEC_CATEGORIES
from src.model.extractor import DINOv2Extractor
from src.model.patchcore import PatchCore
from src.evaluation.metrics import compute_category_metrics, compute_overall_metrics, save_results
from src.visualization.gradcam import visualize_batch


def run_category(
    category: str,
    data_root: str,
    extractor: DINOv2Extractor,
    coreset_ratio: float = 0.1,
    batch_size: int = 16,
    visualize: bool = True,
    save_dir: str = "experiments/results"
):
    print(f"\n{'='*50}")
    print(f"Running category: {category.upper()}")
    print(f"{'='*50}")

    train_dataset = MVTecDataset(data_root, category, split="train")
    test_dataset = MVTecDataset(data_root, category, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    patchcore = PatchCore(extractor=extractor, coreset_ratio=coreset_ratio)
    patchcore.fit(train_loader)

    artifact_path = f"artifacts/{category}_memory_bank.pt"
    patchcore.save(artifact_path)

    all_labels = []
    all_image_scores = []
    all_patch_scores = []
    all_paths = []

    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"]
        paths = batch["path"]

        image_scores, patch_scores = patchcore.predict(images)

        all_labels.extend(labels.tolist())
        all_image_scores.extend(image_scores.tolist())
        all_patch_scores.extend([patch_scores[i] for i in range(len(images))])
        all_paths.extend(paths)

    result = compute_category_metrics(all_labels, all_image_scores, category)

    if visualize:
        vis_dir = f"{save_dir}/visualizations/{category}"
        visualize_batch(
            image_paths=all_paths,
            patch_scores_list=all_patch_scores,
            anomaly_scores=all_image_scores,
            labels=all_labels,
            save_dir=vis_dir,
            n_samples=5
        )

    return result


def main():
    DATA_ROOT = "data/mvtec"
    SAVE_DIR = "experiments/results"
    BATCH_SIZE = 16
    CORESET_RATIO = 0.1

    CATEGORIES_TO_RUN = [
        "cable", "capsule", "carpet", "grid",
        "metal_nut", "pill", "screw", "tile",
        "toothbrush", "transistor", "wood", "zipper"
    ]

    extractor = DINOv2Extractor()

    all_results = []

    for category in CATEGORIES_TO_RUN:
        result = run_category(
            category=category,
            data_root=DATA_ROOT,
            extractor=extractor,
            coreset_ratio=CORESET_RATIO,
            batch_size=BATCH_SIZE,
            visualize=True,
            save_dir=SAVE_DIR
        )
        all_results.append(result)

    overall = compute_overall_metrics(all_results)

    save_results(
        category_results=all_results,
        overall=overall,
        save_path=f"{SAVE_DIR}/results.json"
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()