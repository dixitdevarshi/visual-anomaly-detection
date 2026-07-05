import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model.extractor import DINOv2Extractor


class PatchCore:
    def __init__(self, extractor: DINOv2Extractor, coreset_ratio: float = 0.1):
        """
        PatchCore anomaly detection using a memory bank of normal features.
        Args:
            extractor: DINOv2Extractor instance
            coreset_ratio: fraction of features to keep in memory bank (0.1 = 10%)
        """
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None

    def fit(self, dataloader: DataLoader):
        """
        Build memory bank from normal training images.
        Args:
            dataloader: DataLoader returning normal images only (train split)
        """
        print("Building memory bank from normal images...")
        all_features = []

        for batch in tqdm(dataloader):
            images = batch["image"]
            features = self.extractor.extract_features(images)

            # flatten patches: (B, num_patches, embed_dim) -> (B*num_patches, embed_dim)
            B, N, D = features.shape
            features = features.reshape(B * N, D)
            all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        print(f"Total features before coreset: {all_features.shape}")

        self.memory_bank = self._coreset_subsampling(all_features)
        print(f"Memory bank size after coreset: {self.memory_bank.shape}")

    def _coreset_subsampling(self, features: torch.Tensor) -> torch.Tensor:
        """
        Greedy coreset subsampling to reduce memory bank size.
        Keeps the most representative subset of features.
        """
        n_samples = int(len(features) * self.coreset_ratio)
        n_samples = max(n_samples, 100)  # minimum 100 samples

        selected = [np.random.randint(0, len(features))]

        for _ in tqdm(range(n_samples - 1), desc="Coreset subsampling"):
            selected_features = features[selected]

            # distance from each feature to nearest selected feature
            dists = torch.cdist(features, selected_features)
            min_dists = dists.min(dim=1).values

            # pick the feature furthest from all selected
            next_idx = min_dists.argmax().item()
            selected.append(next_idx)

        return features[selected]

    def predict(self, images: torch.Tensor):
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

        # distance from each patch to nearest memory bank entry
        dists = torch.cdist(features_flat, self.memory_bank)
        min_dists = dists.min(dim=1).values  # (B*N,)

        patch_scores = min_dists.reshape(B, N)
        image_scores = patch_scores.max(dim=1).values  # max patch score = image score

        return image_scores, patch_scores

    def save(self, path: str):
        """Save memory bank to disk."""
        torch.save(self.memory_bank, path)
        print(f"Memory bank saved to {path}")

    def load(self, path: str):
        """Load memory bank from disk."""
        self.memory_bank = torch.load(path)
        print(f"Memory bank loaded from {path}")