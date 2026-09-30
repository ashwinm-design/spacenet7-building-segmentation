"""
src/road/extraction.py

Classical computer-vision baseline for physical road and access corridor extraction.
Combines spectral indices, local texture variance, building exclusion buffering,
and morphological connected-component filtering.

Configurable and modular so that a future deep-learning road segmentation model
can replace this baseline without altering the parcel engine interface.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

from .preprocessing import (
    compute_exg,
    compute_color_features,
    compute_local_variance,
    create_building_exclusion_mask,
)


@dataclass
class RoadExtractionConfig:
    """
    Configuration parameters for road corridor extraction.
    Ensures zero hardcoding and complete reproducibility across runs.
    """
    # Morphological parameters
    morph_opening_size: int = 3
    morph_closing_size: int = 7
    building_exclusion_buffer_px: int = 5  # 1.0 m at 0.2 m/px GSD
    
    # Area & width constraints
    min_road_component_area_px: int = 250  # ~10 m² at 0.2 m GSD
    min_road_width_px: int = 4            # ~0.8 m minimum access corridor
    max_road_width_px: int = 60           # ~12 m maximum urban street width
    min_elongation_ratio: float = 1.8     # Length / Width or Major / Minor axis ratio
    
    # Spectral & texture thresholds
    max_exg_threshold: float = 8.0        # Vegetation suppression limit
    max_saturation_threshold: float = 75.0 # Low saturation (asphalt, concrete, packed dirt)
    min_value_threshold: float = 65.0     # Moderate/high brightness for unpaved/paved roads
    max_value_threshold: float = 245.0    # Avoid specular roof reflections
    max_local_variance: float = 160.0     # Smooth texture requirement
    min_building_distance_px: float = 4.0 # Distance buffer from building walls
    
    # Confidence and smoothing
    confidence_threshold: float = 0.45
    gaussian_blur_sigma: float = 1.0


def extract_road_mask(
    image: np.ndarray,
    building_mask: Optional[np.ndarray] = None,
    config: Optional[RoadExtractionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts road / access corridor binary mask and continuous probability map
    from high-resolution aerial imagery.

    Args:
        image: HxWx3 uint8 RGB numpy array.
        building_mask: Optional HxW binary array where 1 = building footprint.
        config: RoadExtractionConfig instance with tunable hyperparameters.

    Returns:
        (binary_mask: np.ndarray [HxW uint8], probability_map: np.ndarray [HxW float32])
    """
    if config is None:
        config = RoadExtractionConfig()

    h, w = image.shape[:2]

    # 1. Compute spectral & texture signals
    color_feats = compute_color_features(image)
    gray = color_feats["gray"]
    sat = color_feats["saturation"]
    val = color_feats["value"]
    exg = compute_exg(image)
    variance = compute_local_variance(gray, window_size=5)

    # 2. Building distance transform & exclusion mask
    if building_mask is not None and np.sum(building_mask) > 0:
        bldg_exclusion = create_building_exclusion_mask(
            building_mask, buffer_px=config.building_exclusion_buffer_px
        )
        # Distance from buildings
        bldg_inv = (~(building_mask > 0)).astype(np.uint8)
        dist_from_bldg = distance_transform_edt(bldg_inv)
    else:
        bldg_exclusion = np.zeros((h, w), dtype=bool)
        dist_from_bldg = np.full((h, w), 50.0, dtype=np.float32)

    # 3. Soft Likelihood Computation
    # Vegetation penalty (high ExG -> 0 probability)
    veg_prob = np.clip(1.0 - (exg - 0.0) / max(1.0, config.max_exg_threshold), 0.0, 1.0)
    
    # Saturation score (lower saturation -> higher road probability)
    sat_prob = np.clip(1.0 - (sat - 10.0) / max(1.0, config.max_saturation_threshold), 0.0, 1.0)

    # Value / brightness score (unpaved dirt roads & asphalt in typical range)
    val_in_range = (val >= config.min_value_threshold) & (val <= config.max_value_threshold)
    val_prob = np.where(val_in_range, 1.0, 0.2).astype(np.float32)

    # Texture smoothness score (low variance -> higher probability)
    texture_prob = np.clip(1.0 - (variance / max(1.0, config.max_local_variance)), 0.0, 1.0)

    # Building clearance score
    dist_prob = np.clip(dist_from_bldg / max(1.0, config.min_building_distance_px * 2.0), 0.0, 1.0)

    # Composite Probability Map
    prob_map = (
        0.30 * veg_prob +
        0.25 * sat_prob +
        0.15 * val_prob +
        0.20 * texture_prob +
        0.10 * dist_prob
    )

    # Hard zeroing inside building exclusion zones
    prob_map[bldg_exclusion] = 0.0
    prob_map = np.clip(prob_map, 0.0, 1.0)

    # Optional gentle spatial smoothing
    if config.gaussian_blur_sigma > 0:
        prob_map = cv2.GaussianBlur(prob_map, (0, 0), sigmaX=config.gaussian_blur_sigma)
        prob_map[bldg_exclusion] = 0.0

    # 4. Thresholding to create initial binary mask
    raw_mask = (prob_map >= config.confidence_threshold).astype(np.uint8)

    # 5. Morphological Cleanup (Opening + Closing)
    open_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.morph_opening_size, config.morph_opening_size)
    )
    closed_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.morph_closing_size, config.morph_closing_size)
    )
    morphed = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_k)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, closed_k)

    # Re-apply strict building exclusion
    morphed[bldg_exclusion] = 0

    # 6. Connected Components & Geometric Filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morphed, connectivity=8
    )

    final_mask = np.zeros_like(morphed, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px < config.min_road_component_area_px:
            continue

        comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(comp_w, comp_h) / max(1.0, min(comp_w, comp_h))

        # Check elongation or continuous network connectivity
        # Corridors spanning multiple blocks will have large bounding dimensions or aspect ratio
        if aspect_ratio >= config.min_elongation_ratio or area_px >= 1000:
            final_mask[labels == label_id] = 1

    return final_mask, prob_map.astype(np.float32)
