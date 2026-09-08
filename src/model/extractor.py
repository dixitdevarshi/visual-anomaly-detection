import torch
import torch.nn as nn
from typing import Optional
from torchvision import transforms
from PIL import Image


class DINOv2Extractor(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", device: Optional[str] = None) -> None:
        """
        Frozen DINOv2 feature extractor for patch-level features.

        Args:
            model_name: DINOv2 variant. vitb14 is the best balance of speed and quality.
            device: cuda or cpu. Auto-detected if not provided.
        """
        super().__init__()

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("No GPU detected, running on CPU")

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"DINOv2 loaded on {self.device}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from a batch of images.

        Args:
            images: tensor of shape (B, 3, 224, 224)

        Returns:
            patch features of shape (B, num_patches, embed_dim)
        """
        images = images.to(self.device)

        with torch.no_grad():
            features: torch.Tensor = self.model.get_intermediate_layers(
                images, n=1, return_class_token=False
            )[0]

        return features

    def extract_from_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from disk and extract features.

        Args:
            image_path: path to image file

        Returns:
            patch features of shape (1, num_patches, embed_dim)
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return self.extract_features(tensor)


if __name__ == "__main__":
    extractor = DINOv2Extractor()
    print(f"Running on: {extractor.device}")
    dummy = torch.randn(2, 3, 224, 224)
    features = extractor.extract_features(dummy)
    print(f"Feature shape: {features.shape}")
