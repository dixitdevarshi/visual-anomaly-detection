from pydantic import BaseModel, Field
from typing import List


class PipelineConfig(BaseModel):
    data_root: str = Field(default="data/mvtec", description="Path to MVTec dataset root")
    save_dir: str = Field(default="experiments/results", description="Directory to save results")
    batch_size: int = Field(default=16, ge=1, le=64, description="Batch size for dataloader")
    coreset_ratio: float = Field(default=0.1, gt=0.0, le=1.0, description="Fraction of features to keep in memory bank")
    model_name: str = Field(default="dinov2_vitb14", description="DINOv2 model variant")
    image_size: int = Field(default=224, description="Image size for resizing")
    visualize: bool = Field(default=True, description="Whether to save heatmap visualizations")
    n_samples_visualize: int = Field(default=5, description="Number of samples to visualize per category")
    categories: List[str] = Field(
        default=[
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper"
        ],
        description="MVTec categories to run"
    )
    experiment_name: str = Field(default="visual-anomaly-detection", description="MLflow experiment name")

    class Config:
        validate_assignment = True