import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from src.model.extractor import DINOv2Extractor


class PatchCore:
    def __init__(self, extractor: DINOv2Extractor, coreset_ratio: float = 0.1) -> None:
        """
        PatchCore anomaly detection using a memory bank of normal features.

        Args:
            extractor: DINOv2Extractor instance
            coreset_ratio: fraction of features to keep in memory bank
        """
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.memory_bank: Optional[torch.Tensor] = None

    def fit(self, dataloader: DataLoader) -> None:
        """
        Build memory bank from normal training images.

        Args:
            dataloader: DataLoader returning normal images only
        """
        print("Building memory bank from normal images...")
        all_features: list[torch.Tensor] = []

        for batch in tqdm(dataloader):
            images: torch.Tensor = batch["image"]
            features: torch.Tensor = self.extractor.extract_features(images)

            B, N, D = features.shape
            features = features.reshape(B * N, D)
            all_features.append(features.cpu())

        combined: torch.Tensor = torch.cat(all_features, dim=0)
        print(f"Total features before coreset: {combined.shape}")

        self.memory_bank = self._coreset_subsampling(combined)
        print(f"Memory bank size after coreset: {self.memory_bank.shape}")

    def _coreset_subsampling(self, features: torch.Tensor) -> torch.Tensor:
        """
        Greedy coreset subsampling to reduce memory bank size.

        Args:
            features: tensor of shape (N, embed_dim)

        Returns:
            coreset tensor of shape (n_samples, embed_dim)
        """
        n_samples = max(int(len(features) * self.coreset_ratio), 100)

        selected: list[int] = [int(np.random.randint(0, len(features)))]

        for _ in tqdm(range(n_samples - 1), desc="Coreset subsampling"):
            selected_features = features[selected]
            dists = torch.cdist(features, selected_features)
            min_dists = dists.min(dim=1).values
            next_idx = int(min_dists.argmax().item())
            selected.append(next_idx)

        return features[selected]

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores for a batch of images.

        Args:
            images: tensor of shape (B, 3, 224, 224)

        Returns:
            image_scores: (B,) anomaly score per image
            patch_scores: (B, num_patches) score per patch for heatmap
        """
        assert self.memory_bank is not None, "Call fit() before predict()"

        features = self.extractor.extract_features(images)
        B, N, D = features.shape
        features_flat = features.reshape(B * N, D).cpu()

        dists = torch.cdist(features_flat, self.memory_bank)
        min_dists = dists.min(dim=1).values

        patch_scores = min_dists.reshape(B, N)
        image_scores = patch_scores.max(dim=1).values

        return image_scores, patch_scores

    def save(self, path: str) -> None:
        """Save memory bank to disk."""
        torch.save(self.memory_bank, path)
        print(f"Memory bank saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank from disk."""
        self.memory_bank = torch.load(path, weights_only=True)
        print(f"Memory bank loaded from {path}")
