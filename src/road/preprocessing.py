"""
src/road/preprocessing.py

Feature extraction and preprocessing utilities for physical road corridor identification.
Candidate signals include:
  - Excess Green Index (ExG = 2G - R - B) for vegetation suppression
  - HSV / Grayscale spectral characteristics
  - Local texture / variance filters
  - Building footprint buffer exclusion masks
"""

from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter, binary_dilation


def compute_exg(image_rgb: np.ndarray) -> np.ndarray:
    """
    Computes the Excess Green index (ExG = 2*G - R - B) normalized to float.
    Positive high values strongly correlate with living vegetation / canopy.
    Low / negative values indicate bare soil, asphalt, or concrete.
    """
    img_f = image_rgb.astype(np.float32)
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]
    exg = 2.0 * g - r - b
    return exg


def compute_color_features(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extracts color representations from RGB imagery.
    Returns:
        dict containing 'gray', 'hue', 'saturation', 'value' arrays.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "gray": gray,
        "hue": hsv[:, :, 0],
        "saturation": hsv[:, :, 1],
        "value": hsv[:, :, 2],
    }


def compute_local_variance(image_gray: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Computes local texture variance within a sliding window.
    Road surfaces exhibit uniform spectral responses (low variance),
    whereas vegetative canopies and complex building roofs exhibit high texture variance.
    """
    img_f = image_gray.astype(np.float32)
    mean = uniform_filter(img_f, size=window_size)
    mean_sq = uniform_filter(img_f ** 2, size=window_size)
    variance = np.maximum(0.0, mean_sq - mean ** 2)
    return variance


def create_building_exclusion_mask(
    building_mask: np.ndarray,
    buffer_px: int = 5,
) -> np.ndarray:
    """
    Generates a dilated building exclusion mask.
    Prevents road corridors from encroaching on structural footprints or roof overhangs.
    
    Args:
        building_mask: 2D binary uint8 or boolean array where 1 = building.
        buffer_px: Dilation radius in pixels (e.g. 5 px = 1.0 m at 0.2m GSD).
    """
    if building_mask is None or np.sum(building_mask) == 0:
        return np.zeros_like(building_mask, dtype=bool) if building_mask is not None else np.zeros((1, 1), dtype=bool)

    b_bin = (building_mask > 0).astype(np.uint8)
    if buffer_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_px + 1, 2 * buffer_px + 1))
        dilated = cv2.dilate(b_bin, kernel)
        return dilated > 0
    return b_bin > 0
