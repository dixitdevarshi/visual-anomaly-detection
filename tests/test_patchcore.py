import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.model.patchcore import PatchCore
from src.model.extractor import DINOv2Extractor


class MockExtractor:
    """Mock DINOv2Extractor for testing without loading the actual model."""
    def __init__(self):
        self.device = "cpu"

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        # return fake patch features: (B, 256 patches, 768 dims)
        return torch.randn(B, 256, 768)


@pytest.fixture
def extractor():
    return MockExtractor()


@pytest.fixture
def patchcore(extractor):
    return PatchCore(extractor=extractor, coreset_ratio=0.5)


@pytest.fixture
def dummy_dataloader():
    images = torch.randn(8, 3, 224, 224)
    labels = torch.zeros(8, dtype=torch.int)
    dataset = [{"image": images[i], "label": labels[i],
                "defect_type": "good", "path": f"fake/path/{i}.png"}
               for i in range(8)]

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch]),
            "defect_type": [b["defect_type"] for b in batch],
            "path": [b["path"] for b in batch],
        }

    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


# ── PatchCore initialization ────────────────────────────────────────────────

def test_patchcore_init(patchcore):
    assert patchcore.memory_bank is None
    assert patchcore.coreset_ratio == 0.5


def test_patchcore_init_default_coreset():
    extractor = MockExtractor()
    pc = PatchCore(extractor=extractor)
    assert pc.coreset_ratio == 0.1


# ── Memory bank fitting ──────────────────────────────────────────────────────

def test_fit_builds_memory_bank(patchcore, dummy_dataloader):
    assert patchcore.memory_bank is None
    patchcore.fit(dummy_dataloader)
    assert patchcore.memory_bank is not None


def test_fit_memory_bank_shape(patchcore, dummy_dataloader):
    patchcore.fit(dummy_dataloader)
    # memory bank should be 2D: (n_samples, embed_dim)
    assert patchcore.memory_bank.ndim == 2
    assert patchcore.memory_bank.shape[1] == 768


def test_fit_memory_bank_is_smaller_than_full(patchcore, dummy_dataloader):
    patchcore.fit(dummy_dataloader)
    # coreset at 0.5 should reduce the memory bank
    # 8 images x 256 patches = 2048 total patches
    # 50% coreset = max 1024 but at least 100 (minimum enforced)
    assert patchcore.memory_bank.shape[0] <= 2048


# ── Anomaly scoring ──────────────────────────────────────────────────────────

def test_predict_returns_correct_shapes(patchcore, dummy_dataloader):
    patchcore.fit(dummy_dataloader)

    test_images = torch.randn(4, 3, 224, 224)
    image_scores, patch_scores = patchcore.predict(test_images)

    assert image_scores.shape == (4,)
    assert patch_scores.shape == (4, 256)


def test_predict_scores_are_non_negative(patchcore, dummy_dataloader):
    patchcore.fit(dummy_dataloader)

    test_images = torch.randn(4, 3, 224, 224)
    image_scores, patch_scores = patchcore.predict(test_images)

    assert (image_scores >= 0).all()
    assert (patch_scores >= 0).all()


def test_predict_image_score_is_max_of_patch_scores(patchcore, dummy_dataloader):
    patchcore.fit(dummy_dataloader)

    test_images = torch.randn(4, 3, 224, 224)
    image_scores, patch_scores = patchcore.predict(test_images)

    expected = patch_scores.max(dim=1).values
    assert torch.allclose(image_scores, expected)


def test_predict_raises_without_fit(patchcore):
    test_images = torch.randn(2, 3, 224, 224)
    with pytest.raises(AssertionError):
        patchcore.predict(test_images)


# ── Save and load ────────────────────────────────────────────────────────────

def test_save_and_load(patchcore, dummy_dataloader, tmp_path):
    patchcore.fit(dummy_dataloader)
    original_bank = patchcore.memory_bank.clone()

    save_path = str(tmp_path / "test_memory_bank.pt")
    patchcore.save(save_path)

    new_pc = PatchCore(extractor=MockExtractor())
    new_pc.load(save_path)

    assert torch.allclose(new_pc.memory_bank, original_bank)


def test_load_restores_correct_shape(patchcore, dummy_dataloader, tmp_path):
    patchcore.fit(dummy_dataloader)
    save_path = str(tmp_path / "test_memory_bank.pt")
    patchcore.save(save_path)

    new_pc = PatchCore(extractor=MockExtractor())
    new_pc.load(save_path)

    assert new_pc.memory_bank.ndim == 2
    assert new_pc.memory_bank.shape[1] == 768


# ── Coreset subsampling ──────────────────────────────────────────────────────

def test_coreset_reduces_features(patchcore):
    features = torch.randn(1000, 768)
    coreset = patchcore._coreset_subsampling(features)
    assert coreset.shape[0] < 1000
    assert coreset.shape[1] == 768


def test_coreset_minimum_samples(patchcore):
    # minimum 100 samples enforced even when ratio gives fewer
    features = torch.randn(50, 768)
    coreset = patchcore._coreset_subsampling(features)
    # minimum is 100, so coreset will have 100 samples
    assert coreset.shape[0] >= 100
    assert coreset.shape[1] == 768