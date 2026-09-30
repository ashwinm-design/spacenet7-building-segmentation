"""
src/road_dl/evaluation.py

Comprehensive evaluation framework for deep-learning road extraction.
Computes:
1. Standard Pixel-Level Metrics: IoU, Dice, Precision, Recall.
2. Road Topological & Connectivity Metrics (Step 6):
   - Connected component count and fragmentation percentage
   - Largest connected component ratio
   - Skeleton continuity and branch/endpoint topology
   - Endpoint density (dead-ends per kilometer)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import convolve


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Computes morphological skeleton using pure OpenCV erosion/dilation."""
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    size = np.size(img)

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if (size - cv2.countNonZero(img)) == size:
            break
    return (skel > 0).astype(np.uint8)


def compute_road_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    """
    Computes standard binary segmentation metrics.
    Args:
        pred_mask: Binary numpy array (0 or 1).
        target_mask: Ground-truth binary numpy array (0 or 1).
    """
    p = (pred_mask > 0).astype(bool)
    t = (target_mask > 0).astype(bool)

    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def compute_connectivity_metrics(mask: np.ndarray, gsd_m: float = 0.5) -> Dict[str, Any]:
    """
    Evaluates topological graph and connectivity characteristics of road corridors.
    
    Args:
        mask: 2D binary uint8 road mask (1 = road, 0 = non-road).
        gsd_m: Ground sampling distance in meters per pixel.
        
    Returns:
        Structured dictionary of connectivity and network continuity metrics.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    total_road_px = int(np.sum(bin_mask))

    if total_road_px == 0:
        return {
            "component_count": 0,
            "largest_component_px": 0,
            "largest_component_ratio": 0.0,
            "fragmented_component_count": 0,
            "fragmented_road_pct": 0.0,
            "skeleton_length_m": 0.0,
            "endpoint_count": 0,
            "junction_count": 0,
            "endpoint_density_per_km": 0.0,
            "connectivity_score": 0.0,
        }

    # 1. Connected Components Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # Exclude background (label 0)
    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    comp_count = len(comp_areas)

    largest_comp_px = int(np.max(comp_areas)) if comp_count > 0 else 0
    largest_comp_ratio = round(float(largest_comp_px / max(1, total_road_px)), 4)

    # Fragmented components: small isolated fragments (< 100 pixels)
    frag_mask = comp_areas < 100
    frag_count = int(np.sum(frag_mask))
    frag_area_px = int(np.sum(comp_areas[frag_mask]))
    frag_pct = round(100.0 * frag_area_px / max(1, total_road_px), 2)

    # 2. Skeleton & Topological Graph Analysis
    skel = _morphological_skeleton(bin_mask)
    skel_px = int(np.sum(skel))
    skel_length_m = round(float(skel_px * gsd_m), 2)

    # 3x3 neighbor convolution to identify endpoints and junctions
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    neighbor_count = convolve(skel, kernel, mode="constant", cval=0) * skel

    # Endpoints: 1 neighbor; Junctions: >= 3 neighbors
    endpoints = np.logical_and(skel, neighbor_count == 1)
    junctions = np.logical_and(skel, neighbor_count >= 3)

    endpoint_count = int(np.sum(endpoints))
    junction_count = int(np.sum(junctions))

    # Dead-ends per km
    skel_km = max(1e-4, skel_length_m / 1000.0)
    endpoint_density = round(float(endpoint_count / skel_km), 2)

    # Connectivity Score: penalizes excessive fragmentation and dead-ends
    # Higher score = continuous, well-connected arterial network
    connectivity_score = round(float(largest_comp_ratio * (1.0 - (frag_pct / 100.0))), 4)

    return {
        "component_count": comp_count,
        "largest_component_px": largest_comp_px,
        "largest_component_ratio": largest_comp_ratio,
        "fragmented_component_count": frag_count,
        "fragmented_road_pct": frag_pct,
        "skeleton_length_m": skel_length_m,
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
        "endpoint_density_per_km": endpoint_density,
        "connectivity_score": connectivity_score,
    }
