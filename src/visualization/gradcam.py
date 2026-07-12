import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no window popups
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path


def patch_scores_to_heatmap(
    patch_scores: torch.Tensor,
    image_size: int = 224,
    patch_size: int = 14
) -> np.ndarray:
    """
    Convert patch-level anomaly scores to a heatmap.
    Args:
        patch_scores: (num_patches,) anomaly scores per patch
        image_size: original image size
        patch_size: DINOv2 patch size (14 for vitb14)
    Returns:
        heatmap as numpy array (image_size, image_size) normalized to 0-1
    """
    n_patches = int(image_size // patch_size)
    scores_2d = patch_scores.reshape(n_patches, n_patches).numpy()

    heatmap = torch.nn.functional.interpolate(
        torch.tensor(scores_2d).unsqueeze(0).unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    Args:
        image: original image as numpy array (H, W, 3) in 0-255
        heatmap: normalized heatmap (H, W) in 0-1
        alpha: heatmap transparency
        colormap: matplotlib colormap name
    Returns:
        overlaid image as numpy array (H, W, 3)
    """
    cmap = cm.get_cmap(colormap)
    colored_heatmap = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlaid = (alpha * colored_heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlaid


def visualize_anomaly(
    image_path: str,
    patch_scores: torch.Tensor,
    anomaly_score: float,
    label: int,
    save_path: str = None,
    image_size: int = 224,
    patch_size: int = 14
):
    """
    Full visualization: original image + heatmap side by side.
    Saves to disk silently — no window popup.
    """
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_np = np.array(image)

    heatmap = patch_scores_to_heatmap(patch_scores, image_size, patch_size)
    overlaid = overlay_heatmap(image_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlaid)
    axes[2].set_title(f"Overlay | Score: {anomaly_score:.4f} | {'Anomaly' if label else 'Normal'}")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.close()


def visualize_batch(
    image_paths: list,
    patch_scores_list: list,
    anomaly_scores: list,
    labels: list,
    save_dir: str = None,
    n_samples: int = 5
):
    """
    Visualize a batch of predictions.
    All images saved to disk — no manual closing required.
    """
    indices = np.random.choice(len(image_paths), min(n_samples, len(image_paths)), replace=False)

    for i, idx in enumerate(indices):
        save_path = None
        if save_dir:
            save_path = str(Path(save_dir) / f"sample_{i}.png")

        visualize_anomaly(
            image_path=image_paths[idx],
            patch_scores=patch_scores_list[idx],
            anomaly_score=anomaly_scores[idx],
            label=labels[idx],
            save_path=save_path
        )