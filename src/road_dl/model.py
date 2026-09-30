"""
src/road_dl/model.py

Deep-Learning Architecture for Road & Access Corridor Segmentation.
Implements U-Net with ResNet18 encoder, outputting binary road logits.
Reuses validated model infrastructure from src/model.py.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import get_model, SimpleUNet


def get_road_model(
    encoder_name: str = "resnet18",
    classes: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Constructs a U-Net model with ResNet18 encoder for road extraction.
    
    Args:
        encoder_name: Encoder backbone (default 'resnet18').
        classes: Number of output channels (1 for binary road segmentation).
        pretrained: Whether to load ImageNet pre-trained encoder weights.
        
    Returns:
        PyTorch nn.Module.
    """
    return get_model(
        architecture="unet",
        encoder_name=encoder_name,
        in_channels=3,
        classes=classes,
        pretrained=pretrained,
    )
