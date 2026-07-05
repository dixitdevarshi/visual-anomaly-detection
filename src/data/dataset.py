import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


class MVTecDataset(Dataset):
    def __init__(self, root_dir: str, category: str, split: str = "train", image_size: int = 224):
        """
        Args:
            root_dir: path to MVTec AD root folder
            category: one of MVTEC_CATEGORIES
            split: "train" or "test"
            image_size: resize target
        """
        assert category in MVTEC_CATEGORIES, f"Invalid category. Choose from {MVTEC_CATEGORIES}"
        assert split in ("train", "test"), "Split must be 'train' or 'test'"

        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.image_paths, self.labels, self.defect_types = self._load_dataset()

    def _load_dataset(self):
        image_paths = []
        labels = []
        defect_types = []

        split_dir = self.root_dir / self.category / self.split

        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            if not defect_dir.is_dir():
                continue

            label = 0 if defect_type == "good" else 1

            for img_file in sorted(defect_dir.iterdir()):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    image_paths.append(img_file)
                    labels.append(label)
                    defect_types.append(defect_type)

        return image_paths, labels, defect_types

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": self.labels[idx],
            "defect_type": self.defect_types[idx],
            "path": str(self.image_paths[idx])
        }