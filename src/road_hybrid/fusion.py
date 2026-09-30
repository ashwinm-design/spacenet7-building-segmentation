"""
src/road_hybrid/fusion.py

Core fusion algorithms and connectivity refinement for SIH26012 Phase 13D:
1. Probability normalization
2. Strategy A: Weighted Probability Fusion (75/25)
3. Strategy B: Balanced Probability Fusion (50/50)
4. Strategy C: Deep-Learning Assisted Recovery (Classical anchor + DL narrow pathway recovery)
5. Strategy D: Logical Union with Structural Filtering
6. Building exclusion with configurable safety buffers (0m, 0.2m, 0.5m, 1.0m)
7. Topological gap bridging and connectivity refinement (0.5m, 1.0m, 2.0m, 3.0m)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, convolve


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


@dataclass
class HybridRoadResult:
    """Encapsulates output deliverables of a road fusion configuration."""
    strategy_name: str
    mask: np.ndarray             # Binary uint8 (0 or 1), shape (H, W)
    probability_map: np.ndarray  # Float32 in [0, 1], shape (H, W)
    params: Dict[str, Any] = field(default_factory=dict)
    corridor_area_m2: float = 0.0
    pixel_count: int = 0


def normalize_probability_maps(
    cv_prob: np.ndarray,
    dl_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes classical CV likelihood and deep-learning sigmoid probabilities into [0.0, 1.0].
    Ensures both fields share identical float32 numeric range for fair evidence combination.
    """
    if cv_prob.dtype == np.uint8:
        p_cv = cv_prob.astype(np.float32) / 255.0
    else:
        p_cv = cv_prob.astype(np.float32)
        if p_cv.max() > 1.0:
            p_cv = p_cv / 255.0

    if dl_prob.dtype == np.uint8:
        p_dl = dl_prob.astype(np.float32) / 255.0
    else:
        p_dl = dl_prob.astype(np.float32)
        if p_dl.max() > 1.0:
            p_dl = p_dl / 255.0

    p_cv = np.clip(p_cv, 0.0, 1.0)
    p_dl = np.clip(p_dl, 0.0, 1.0)
    return p_cv, p_dl


def apply_building_exclusion(
    road_mask: np.ndarray,
    building_mask: np.ndarray,
    buffer_m: float = 0.0,
    resolution_m: float = 0.20,
) -> np.ndarray:
    """
    Step 5: Enforces hard architectural building exclusion.
    road_mask[building_mask == 1] = 0.
    Optionally applies a safety setback buffer around building footprints.
    """
    cleaned_mask = road_mask.copy().astype(np.uint8)
    if building_mask is None or np.sum(building_mask) == 0:
        return cleaned_mask

    if buffer_m > 0.0:
        buffer_px = int(round(buffer_m / resolution_m))
        if buffer_px > 0:
            kernel_size = 2 * buffer_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_bldg = cv2.dilate(building_mask.astype(np.uint8), kernel)
        else:
            dilated_bldg = building_mask.astype(np.uint8)
    else:
        dilated_bldg = building_mask.astype(np.uint8)

    cleaned_mask[dilated_bldg > 0] = 0
    return cleaned_mask


def fuse_roads_weighted(
    p_cv: np.ndarray,
    p_dl: np.ndarray,
    w_cv: float = 0.75,
    w_dl: float = 0.25,
    threshold: float = 0.40,
    building_mask: Optional[np.ndarray] = None,
    buffer_m: float = 0.20,
    resolution_m: float = 0.20,
) -> HybridRoadResult:
    """
    Strategy A & B: Linear weighted combination of evidence fields.
    P_hybrid = w_cv * P_cv + w_dl * P_dl.
    """
    p_hybrid = (w_cv * p_cv) + (w_dl * p_dl)
    p_hybrid = np.clip(p_hybrid, 0.0, 1.0)

    raw_mask = (p_hybrid >= threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Apply building exclusion
    final_mask = apply_building_exclusion(clean_mask, building_mask, buffer_m=buffer_m, resolution_m=resolution_m)
    px_count = int(np.sum(final_mask))
    area_m2 = px_count * (resolution_m ** 2)

    name = f"hybrid_weighted_{int(w_cv*100)}_{int(w_dl*100)}"
    return HybridRoadResult(
        strategy_name=name,
        mask=final_mask,
        probability_map=p_hybrid,
        params={"w_cv": w_cv, "w_dl": w_dl, "threshold": threshold, "buffer_m": buffer_m},
        corridor_area_m2=round(area_m2, 2),
        pixel_count=px_count,
    )


def fuse_roads_dl_recovery(
    cv_mask: np.ndarray,
    p_cv: np.ndarray,
    p_dl: np.ndarray,
    dl_threshold: float = 0.40,
    max_recovery_dist_m: float = 15.0,
    building_mask: Optional[np.ndarray] = None,
    buffer_m: float = 0.20,
    resolution_m: float = 0.20,
) -> HybridRoadResult:
    """
    Strategy C: Deep-Learning Assisted Narrow Corridor Recovery.
    Uses classical CV as the authoritative structural trunk, and leverages DL
    to recover connected narrow alleyways and pathways within a constrained distance
    to existing verified corridors.
    """
    # Distance from verified classical corridor
    inv_cv = (cv_mask == 0).astype(np.uint8)
    dist_px = distance_transform_edt(inv_cv)
    dist_m = dist_px * resolution_m

    # DL candidate pathways: high DL confidence and within proximity envelope
    dl_candidates = (p_dl >= dl_threshold) & (dist_m <= max_recovery_dist_m)

    # Filter isolated DL noise: candidate components must connect to or be close to CV trunk
    combined_raw = np.logical_or(cv_mask > 0, dl_candidates).astype(np.uint8)

    # Morphological refinement
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_clean = cv2.morphologyEx(combined_raw, cv2.MORPH_CLOSE, k_close)

    # Building exclusion
    final_mask = apply_building_exclusion(combined_clean, building_mask, buffer_m=buffer_m, resolution_m=resolution_m)

    # Hybrid probability composite
    p_hybrid = np.maximum(p_cv, p_dl * (dist_m <= max_recovery_dist_m).astype(np.float32))

    px_count = int(np.sum(final_mask))
    area_m2 = px_count * (resolution_m ** 2)

    return HybridRoadResult(
        strategy_name="hybrid_dl_recovery",
        mask=final_mask,
        probability_map=p_hybrid,
        params={"dl_threshold": dl_threshold, "max_recovery_dist_m": max_recovery_dist_m, "buffer_m": buffer_m},
        corridor_area_m2=round(area_m2, 2),
        pixel_count=px_count,
    )


def fuse_roads_union_filtered(
    cv_mask: np.ndarray,
    dl_mask: np.ndarray,
    p_cv: np.ndarray,
    p_dl: np.ndarray,
    min_area_px: int = 150,     # ~6 m² minimum connected component
    building_mask: Optional[np.ndarray] = None,
    buffer_m: float = 0.20,
    resolution_m: float = 0.20,
) -> HybridRoadResult:
    """
    Strategy D: Logical Union with Structural Filtering.
    (CV_mask OR DL_mask) followed by hard building exclusion, morphological opening/closing,
    and connected-component area thresholding.
    """
    union_raw = np.logical_or(cv_mask > 0, dl_mask > 0).astype(np.uint8)

    # Building exclusion
    union_no_bldg = apply_building_exclusion(union_raw, building_mask, buffer_m=buffer_m, resolution_m=resolution_m)

    # Morphological opening to sever tenuous single-pixel bridges, closing to seal pinholes
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(union_no_bldg, cv2.MORPH_OPEN, k_open)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, k_close)

    # Component filtering: remove micro-debris < min_area_px
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    final_mask = np.zeros_like(morphed, dtype=np.uint8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            final_mask[labels == lbl] = 1

    p_hybrid = np.maximum(p_cv, p_dl)
    px_count = int(np.sum(final_mask))
    area_m2 = px_count * (resolution_m ** 2)

    return HybridRoadResult(
        strategy_name="hybrid_union_filtered",
        mask=final_mask,
        probability_map=p_hybrid,
        params={"min_area_px": min_area_px, "buffer_m": buffer_m},
        corridor_area_m2=round(area_m2, 2),
        pixel_count=px_count,
    )


def refine_road_connectivity(
    mask: np.ndarray,
    p_hybrid: np.ndarray,
    building_mask: Optional[np.ndarray] = None,
    gap_threshold_m: float = 1.0,
    resolution_m: float = 0.20,
    min_evidence_threshold: float = 0.15,
    min_component_area_m2: float = 15.0,
) -> np.ndarray:
    """
    Step 6: Graph/Topological connectivity refinement.
    Identifies endpoints of skeleton components and conservatively bridges small gaps
    <= gap_threshold_m if and only if:
    1. The bridging trajectory does NOT intersect any building footprint.
    2. The underlying road evidence along the bridge is non-zero (>= min_evidence_threshold),
       preventing phantom links across broad vacant land or courtyards.
    3. Prunes detached micro-components below min_component_area_m2.
    """
    refined_mask = mask.copy().astype(np.uint8)
    gap_px = int(round(gap_threshold_m / resolution_m))

    if gap_px <= 0:
        return refined_mask

    # Extract skeleton
    skel = _morphological_skeleton(refined_mask > 0)

    # Find endpoints (pixels with exactly 1 neighbor in 3x3)
    k_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve(skel, k_neighbors, mode="constant", cval=0) * skel
    endpoints = np.argwhere(neighbor_count == 1)

    if len(endpoints) < 2:
        return refined_mask

    # Bounded pairwise search for endpoints within gap_px
    bridged_count = 0
    bridge_overlay = np.zeros_like(refined_mask, dtype=np.uint8)

    for i in range(len(endpoints)):
        r1, c1 = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            r2, c2 = endpoints[j]
            dist = np.hypot(r2 - r1, c2 - c1)
            if 1 < dist <= gap_px:
                # Sample line between (r1, c1) and (r2, c2)
                num_pts = int(dist * 2) + 2
                rs = np.linspace(r1, r2, num_pts).astype(int)
                cs = np.linspace(c1, c2, num_pts).astype(int)

                # Check building intersection
                if building_mask is not None:
                    if np.any(building_mask[rs, cs] > 0):
                        continue  # Do not cross buildings

                # Check underlying hybrid road evidence
                line_evidence = p_hybrid[rs, cs]
                if np.mean(line_evidence) < min_evidence_threshold:
                    continue  # Do not bridge over confirmed open terrain

                # Draw conservative 2-pixel connection stroke
                cv2.line(bridge_overlay, (int(c1), int(r1)), (int(c2), int(r2)), 1, thickness=2)
                bridged_count += 1

    refined_mask = np.logical_or(refined_mask > 0, bridge_overlay > 0).astype(np.uint8)

    # Re-enforce building exclusion
    if building_mask is not None:
        refined_mask[building_mask > 0] = 0

    # Prune detached micro-components below minimum area
    min_px = int(round(min_component_area_m2 / (resolution_m ** 2)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
    clean_refined = np.zeros_like(refined_mask, dtype=np.uint8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            clean_refined[labels == lbl] = 1

    return clean_refined


def run_fusion_strategy_experiments(
    cv_mask: np.ndarray,
    cv_prob: np.ndarray,
    dl_mask: np.ndarray,
    dl_prob: np.ndarray,
    building_mask: np.ndarray,
    resolution_m: float = 0.20,
) -> Dict[str, HybridRoadResult]:
    """
    Executes all required fusion strategies across multiple parameters:
    - Strategy A: Weighted 75/25
    - Strategy B: Balanced 50/50
    - Strategy C: DL-assisted Recovery
    - Strategy D: Union + Structural Filtering
    - Building Buffer sensitivity tests (0m, 0.2m, 0.5m, 1.0m)
    - Connectivity Gap thresholds (0.5m, 1.0m, 2.0m, 3.0m)
    """
    p_cv, p_dl = normalize_probability_maps(cv_prob, dl_prob)
    results: Dict[str, HybridRoadResult] = {}

    # 1. Strategy A: Weighted 75/25
    results["hybrid_weighted_75_25"] = fuse_roads_weighted(
        p_cv, p_dl, w_cv=0.75, w_dl=0.25, threshold=0.40, building_mask=building_mask, buffer_m=0.20
    )

    # 2. Strategy B: Balanced 50/50
    results["hybrid_balanced_50_50"] = fuse_roads_weighted(
        p_cv, p_dl, w_cv=0.50, w_dl=0.50, threshold=0.40, building_mask=building_mask, buffer_m=0.20
    )

    # 3. Strategy C: DL-Assisted Recovery (default 15m radius)
    results["hybrid_dl_recovery"] = fuse_roads_dl_recovery(
        cv_mask, p_cv, p_dl, dl_threshold=0.40, max_recovery_dist_m=15.0, building_mask=building_mask, buffer_m=0.20
    )

    # 4. Strategy D: Union + Structural Filtering
    results["hybrid_union_filtered"] = fuse_roads_union_filtered(
        cv_mask, dl_mask, p_cv, p_dl, min_area_px=150, building_mask=building_mask, buffer_m=0.20
    )

    # 5. Building Setback Buffer Variations on DL-Assisted Recovery
    for buf in [0.0, 0.5, 1.0]:
        tag = f"hybrid_dl_recovery_buf_{int(buf*100)}cm"
        results[tag] = fuse_roads_dl_recovery(
            cv_mask, p_cv, p_dl, dl_threshold=0.40, max_recovery_dist_m=15.0, building_mask=building_mask, buffer_m=buf
        )

    # 6. Connectivity Refinement on Top Strategy (DL-Assisted Recovery) with Gap Variations
    base_res = results["hybrid_dl_recovery"]
    for gap in [0.5, 1.0, 2.0, 3.0]:
        refined_m = refine_road_connectivity(
            base_res.mask,
            base_res.probability_map,
            building_mask=building_mask,
            gap_threshold_m=gap,
            resolution_m=resolution_m,
        )
        tag = f"hybrid_dl_recovery_gap_{int(gap*10)}dm"
        px_count = int(np.sum(refined_m))
        results[tag] = HybridRoadResult(
            strategy_name=tag,
            mask=refined_m,
            probability_map=base_res.probability_map,
            params={"base_strategy": "dl_recovery", "gap_threshold_m": gap},
            corridor_area_m2=round(px_count * (resolution_m ** 2), 2),
            pixel_count=px_count,
        )

    return results
