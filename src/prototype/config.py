"""
src/prototype/config.py

Configuration parameters and defaults for the SIH26012 inference backend.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class InferenceConfig:
    """Configuration for building segmentation inference and polygonization."""
    # Checkpoint
    checkpoint_path: str = "results/experiments/open_cities_adaptation/checkpoints/imagenet_ablation_best_model.pth"
    
    # Model architecture
    arch: str = "unet"
    encoder_name: str = "resnet18"
    in_channels: int = 3
    num_classes: int = 3
    
    # Preprocessing & Tiling
    tile_size: int = 256
    tile_stride: int = 256
    batch_size: int = 16
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Normalization (ImageNet standards)
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Reconstruction & Polygonization
    # Frozen setting: Class-1 interior + 1.0 px Euclidean boundary growth
    growth_distance: float = 1.0
    min_area_pixels: float = 15.0
    simplify_tolerance: float = 0.5
    
    # Output structure
    output_dir: str = "results/prototype"
    predictions_subdir: str = "predictions"
    visualizations_subdir: str = "visualizations"
    statistics_subdir: str = "statistics"

    def get_predictions_dir(self) -> str:
        return os.path.join(self.output_dir, self.predictions_subdir)

    def get_visualizations_dir(self) -> str:
        return os.path.join(self.output_dir, self.visualizations_subdir)

    def get_statistics_dir(self) -> str:
        return os.path.join(self.output_dir, self.statistics_subdir)


DEFAULT_CONFIG = InferenceConfig()
