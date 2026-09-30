"""
src/road_dl/

Deep-Learning Road and Access Corridor Extraction Module for SIH26012.
Provides dataset preparation, neural network models (U-Net with ResNet18 encoder),
compound BCE+Dice loss functions, connectivity-aware evaluation, sliding-window inference,
and vector graph conversion.
"""

from .model import get_road_model
from .losses import BCEDiceLoss
from .dataset import RoadPatchDataset, prepare_benchmark_patches
from .inference import predict_orthomosaic_roads
from .evaluation import compute_road_metrics, compute_connectivity_metrics
from .visualization import plot_road_dl_comparisons

__all__ = [
    "get_road_model",
    "BCEDiceLoss",
    "RoadPatchDataset",
    "prepare_benchmark_patches",
    "predict_orthomosaic_roads",
    "compute_road_metrics",
    "compute_connectivity_metrics",
    "plot_road_dl_comparisons",
]
