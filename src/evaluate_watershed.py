"""
Phase 4: SpaceNet 7 Watershed Instance Separation Pipeline.

Applies Euclidean distance transform + peak local maxima detection +
marker-controlled watershed to decouple merged building instances from U-Net predictions.
Evaluates across a parameter grid:
- Probability thresholds: [0.40, 0.50, 0.60]
- Distance transform marker peak min-distance: [3, 5, 8]
- Minimum polygon area: [5, 10, 15] pixels
Compares against the baseline threshold 0.60 results.
"""

import os
import sys
import time
import csv
import json
import argparse
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

import rasterio
from rasterio import features
import shapely
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BuildingDataset
from src.model import get_model


def extract_clean_polygons(
    binary_mask: np.ndarray,
    min_area: float = 15.0,
    simplify_tol: float = 0.5
) -> Tuple[List[Polygon], Dict[str, int]]:
    """Extract clean Shapely polygons from binary mask with hole preservation."""
    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    shapes_gen = features.shapes(mask_uint8, mask=(mask_uint8 == 1))

    polygons = []
    stats = {
        "raw_count": 0,
        "valid_count": 0,
        "invalid_repaired": 0,
        "tiny_removed": 0,
        "has_holes": 0,
        "fragmented": 0
    }

    for geom, val in shapes_gen:
        if val != 1:
            continue
        stats["raw_count"] += 1
        poly = shape(geom)

        if not poly.is_valid:
            stats["invalid_repaired"] += 1
            if hasattr(shapely, "make_valid"):
                poly = shapely.make_valid(poly)
            else:
                poly = poly.buffer(0)

        if poly.is_empty:
            continue

        if poly.geom_type == "Polygon":
            poly_list = [poly]
        elif poly.geom_type == "MultiPolygon":
            poly_list = list(poly.geoms)
            stats["fragmented"] += len(poly_list) - 1
        elif poly.geom_type == "GeometryCollection":
            poly_list = [g for g in poly.geoms if g.geom_type == "Polygon"]
        else:
            continue

        for p in poly_list:
            if p.is_empty:
                continue
            if p.area < min_area:
                stats["tiny_removed"] += 1
                continue
            if simplify_tol > 0:
                p_simp = p.simplify(simplify_tol, preserve_topology=True)
                if not p_simp.is_empty and p_simp.is_valid:
                    p = p_simp
            if hasattr(p, "interiors") and len(list(p.interiors)) > 0:
                stats["has_holes"] += 1
            stats["valid_count"] += 1
            polygons.append(p)

    return polygons, stats


def apply_watershed_separation(
    binary_mask: np.ndarray,
    min_distance: int = 5,
    min_peak_val: float = 1.5
) -> np.ndarray:
    """
    Applies distance transform and marker-controlled watershed
    to split fused building instance blobs.
    Returns: instance label map (0 = background, 1..N = instance IDs).
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    if np.sum(mask_u8) == 0:
        return np.zeros_like(mask_u8, dtype=np.int32)

    # 1. Euclidean distance transform
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)

    # 2. Local peak detection
    filter_size = max(3, 2 * min_distance + 1)
    max_filt = maximum_filter(dist, size=filter_size)
    peaks = (dist == max_filt) & (dist >= min_peak_val) & (mask_u8 == 1)

    num_markers, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if num_markers <= 1:
        # Fallback to connected components if no peaks
        _, fallback = cv2.connectedComponents(mask_u8)
        return fallback.astype(np.int32)

    # 3. Marker-controlled watershed
    # Invert distance for basin simulation
    dist_norm = dist / (dist.max() + 1e-6)
    inv_dist = (255 - np.clip(dist_norm * 255, 0, 255)).astype(np.uint8)
    inv_dist_3ch = cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR)

    markers_cv = markers.copy().astype(np.int32)
    # Background seed = 1, building seeds = 2..N+1
    markers_cv[markers_cv > 0] += 1
    # Unknown regions = 0
    markers_cv[(mask_u8 == 1) & (markers_cv == 0)] = 0
    # Definite background = 1
    markers_cv[mask_u8 == 0] = 1

    cv2.watershed(inv_dist_3ch, markers_cv)

    # Extract separated instance labels (values > 1 inside mask)
    instances = np.zeros_like(mask_u8, dtype=np.int32)
    valid_region = (markers_cv > 1) & (mask_u8 == 1)
    instances[valid_region] = markers_cv[valid_region] - 1

    return instances


def extract_watershed_polygons(
    instance_map: np.ndarray,
    min_area: float = 15.0,
    simplify_tol: float = 0.5
) -> Tuple[List[Polygon], Dict[str, int]]:
    """
    Extract polygons from individual watershed instance labels.
    Ensures boundaries between adjacent watershed instances are preserved as separate polygons.
    """
    unique_ids = [uid for uid in np.unique(instance_map) if uid > 0]
    polygons = []
    stats = {
        "raw_count": len(unique_ids),
        "valid_count": 0,
        "invalid_repaired": 0,
        "tiny_removed": 0,
        "has_holes": 0,
        "fragmented": 0
    }

    for uid in unique_ids:
        single_inst_mask = (instance_map == uid).astype(np.uint8)
        poly_list, sub_stats = extract_clean_polygons(single_inst_mask, min_area=min_area, simplify_tol=simplify_tol)
        polygons.extend(poly_list)
        stats["valid_count"] += sub_stats["valid_count"]
        stats["invalid_repaired"] += sub_stats["invalid_repaired"]
        stats["tiny_removed"] += sub_stats["tiny_removed"]
        stats["has_holes"] += sub_stats["has_holes"]
        stats["fragmented"] += sub_stats["fragmented"]

    return polygons, stats


def match_polygons_spatial(
    gt_polys: List[Polygon],
    pred_polys: List[Polygon],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """Bipartite spatial matching between Ground-Truth and Predicted polygons."""
    m, n = len(gt_polys), len(pred_polys)

    if m == 0 and n == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_gt_indices": [], "matched_pred_indices": [], "mean_matched_iou": 1.0}
    if m == 0:
        return {"tp": 0, "fp": n, "fn": 0, "precision": 0.0, "recall": 1.0, "f1": 0.0, "matched_gt_indices": [], "matched_pred_indices": [], "mean_matched_iou": 0.0}
    if n == 0:
        return {"tp": 0, "fp": 0, "fn": m, "precision": 1.0, "recall": 0.0, "f1": 0.0, "matched_gt_indices": [], "matched_pred_indices": [], "mean_matched_iou": 0.0}

    candidates = []
    for g_idx, g in enumerate(gt_polys):
        g_bound = g.bounds
        for p_idx, p in enumerate(pred_polys):
            p_bound = p.bounds
            if (g_bound[0] > p_bound[2] or g_bound[2] < p_bound[0] or
                g_bound[1] > p_bound[3] or g_bound[3] < p_bound[1]):
                continue
            try:
                inter = g.intersection(p).area
                if inter <= 0:
                    continue
                union = g.area + p.area - inter
                iou = inter / union if union > 0 else 0
                if iou >= iou_threshold:
                    candidates.append((iou, g_idx, p_idx))
            except Exception:
                continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    for iou, g_idx, p_idx in candidates:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            matched_ious.append(iou)

    tp = len(matched_gt)
    fp = n - tp
    fn = m - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "matched_gt_indices": list(matched_gt), "matched_pred_indices": list(matched_pred),
        "mean_matched_iou": round(mean_iou, 4)
    }


def compute_area_metrics(gt_polys: List[Polygon], pred_polys: List[Polygon]) -> Dict[str, float]:
    """Compute aggregate area intersection, precision, recall, and IoU."""
    total_gt_area = sum(p.area for p in gt_polys)
    total_pred_area = sum(p.area for p in pred_polys)

    if total_gt_area == 0 and total_pred_area == 0:
        return {"area_iou": 1.0, "area_precision": 1.0, "area_recall": 1.0, "gt_area": 0.0, "pred_area": 0.0}
    if total_gt_area == 0:
        return {"area_iou": 0.0, "area_precision": 0.0, "area_recall": 1.0, "gt_area": 0.0, "pred_area": total_pred_area}
    if total_pred_area == 0:
        return {"area_iou": 0.0, "area_precision": 1.0, "area_recall": 0.0, "gt_area": total_gt_area, "pred_area": 0.0}

    gt_u = unary_union(gt_polys) if len(gt_polys) > 1 else gt_polys[0]
    pred_u = unary_union(pred_polys) if len(pred_polys) > 1 else pred_polys[0]

    try:
        inter_area = gt_u.intersection(pred_u).area
    except Exception:
        inter_area = 0.0

    union_area = gt_u.area + pred_u.area - inter_area
    area_iou = inter_area / union_area if union_area > 0 else 0.0
    area_prec = inter_area / pred_u.area if pred_u.area > 0 else 0.0
    area_rec = inter_area / gt_u.area if gt_u.area > 0 else 0.0

    return {
        "area_iou": round(area_iou, 4),
        "area_precision": round(area_prec, 4),
        "area_recall": round(area_rec, 4),
        "gt_area": round(total_gt_area, 1),
        "pred_area": round(total_pred_area, 1)
    }


def render_vector_view(
    canvas_shape: Tuple[int, int],
    polygons: List[Polygon],
    fill_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Tuple[int, int, int] = (40, 255, 120),
    border_thickness: int = 1
) -> np.ndarray:
    """Render Shapely polygons onto an RGB canvas."""
    h, w = canvas_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for p in polygons:
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        cv2.fillPoly(canvas, [ext], fill_color)
        for interior in p.interiors:
            hole = np.round(np.array(interior.coords)).astype(np.int32)
            cv2.fillPoly(canvas, [hole], (0, 0, 0))
        cv2.polylines(canvas, [ext], isClosed=True, color=border_color, thickness=border_thickness, lineType=cv2.LINE_AA)
        for interior in p.interiors:
            hole = np.round(np.array(interior.coords)).astype(np.int32)
            cv2.polylines(canvas, [hole], isClosed=True, color=border_color, thickness=border_thickness, lineType=cv2.LINE_AA)

    return canvas


def render_comparison_overlay(
    raw_rgb: np.ndarray,
    gt_polys: List[Polygon],
    pred_polys: List[Polygon],
    match_result: Dict[str, Any]
) -> np.ndarray:
    """Render spatial match overlay on raw RGB imagery."""
    overlay = raw_rgb.copy()
    color_layer = np.zeros_like(raw_rgb)
    matched_p_idx = set(match_result["matched_pred_indices"])
    matched_g_idx = set(match_result["matched_gt_indices"])

    for p_idx, p in enumerate(pred_polys):
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        if p_idx in matched_p_idx:
            cv2.fillPoly(color_layer, [ext], (30, 220, 100))  # Green TP
        else:
            cv2.fillPoly(color_layer, [ext], (255, 60, 60))   # Red FP

    overlay = cv2.addWeighted(overlay, 0.70, color_layer, 0.30, 0)

    for p_idx, p in enumerate(pred_polys):
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        col = (40, 255, 120) if p_idx in matched_p_idx else (255, 70, 70)
        cv2.polylines(overlay, [ext], True, col, 1, lineType=cv2.LINE_AA)

    for g_idx, g in enumerate(gt_polys):
        if g_idx not in matched_g_idx:
            ext = np.round(np.array(g.exterior.coords)).astype(np.int32)
            cv2.polylines(overlay, [ext], True, (255, 230, 50), 1, lineType=cv2.LINE_AA)  # Yellow FN

    return overlay


def run_watershed_experiment(
    model_checkpoint: str = "results/experiments/baseline_full/best_model.pth",
    test_splits_file: str = "data/processed_spacenet7/splits/test.txt",
    output_dir: str = "results/experiments/watershed",
    thresholds: List[float] = [0.40, 0.50, 0.60],
    min_distances: List[int] = [3, 5, 8],
    min_areas: List[float] = [5.0, 10.0, 15.0]
):
    """Run full Phase 4 watershed instance separation experiment."""
    os.makedirs(output_dir, exist_ok=True)
    poly_export_dir = os.path.join(output_dir, "polygons")
    vis_export_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(poly_export_dir, exist_ok=True)
    os.makedirs(vis_export_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 75)
    print("PHASE 4: SPACE NET 7 WATERSHED INSTANCE SEPARATION EXPERIMENT")
    print("=" * 75)
    print(f"Model Checkpoint:           {model_checkpoint}")
    print(f"Test Split File:            {test_splits_file}")
    print(f"Thresholds:                 {thresholds}")
    print(f"Distance Min Distances:     {min_distances} px")
    print(f"Min Polygon Areas:          {min_areas} px")
    print("-" * 75)

    # 1. Load Model
    model = get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=1, pretrained=True)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Test Pairs & Compute Probability Maps
    with open(test_splits_file, "r", encoding="utf-8") as f:
        test_lines = [l.strip().split(",") for l in f.readlines() if l.strip()]

    print(f"Loaded {len(test_lines)} test image-mask pairs.")
    print("Generating U-Net probability maps and ground-truth polygons...")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    test_samples = []
    t0 = time.time()

    with torch.no_grad():
        for img_p, mask_p in test_lines:
            name = os.path.basename(img_p)
            raw_rgb = Image.open(img_p).convert("RGB")
            raw_gt = Image.open(mask_p).convert("L")

            img_arr = np.array(raw_rgb, dtype=np.float32) / 255.0
            norm_img = (img_arr - mean) / std
            tensor_x = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            logits = model(tensor_x)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            gt_mask = (np.array(raw_gt) > 127).astype(np.uint8)

            gt_polys, gt_stats = extract_clean_polygons(gt_mask, min_area=15.0, simplify_tol=0.5)

            test_samples.append({
                "sample_name": name,
                "base_id": os.path.splitext(name)[0],
                "img_path": img_p,
                "mask_path": mask_p,
                "raw_rgb": np.array(raw_rgb),
                "gt_mask": gt_mask,
                "prob_map": prob,
                "gt_polys": gt_polys,
                "gt_poly_count": len(gt_polys)
            })

    total_gt_polygons = sum(s["gt_poly_count"] for s in test_samples)
    print(f"Inference complete in {time.time() - t0:.2f}s across {len(test_samples)} scenes (Total GT Polys: {total_gt_polygons:,})")
    print("-" * 75)

    # 3. Precompute Watershed Instance Maps for (thresh, min_dist)
    print("Precomputing distance-transform watershed instance partitions...")
    watershed_cache = {}  # (thresh, min_dist) -> list of instance_maps for each sample
    baseline_cache = {}   # thresh -> list of baseline binary masks

    t_ws = time.time()
    for thresh in thresholds:
        baseline_cache[thresh] = [(s["prob_map"] > thresh).astype(np.uint8) for s in test_samples]
        for m_dist in min_distances:
            key = (thresh, m_dist)
            watershed_cache[key] = []
            for b_mask in baseline_cache[thresh]:
                inst_map = apply_watershed_separation(b_mask, min_distance=m_dist, min_peak_val=1.5)
                watershed_cache[key].append(inst_map)

    print(f"Precomputed {len(watershed_cache)} watershed combinations in {time.time() - t_ws:.2f}s")
    print("-" * 75)

    # 4. Grid Sweep Evaluation
    grid_results = []
    config_sample_evals = {}

    print(f"Running grid evaluation across {len(thresholds) * len(min_distances) * len(min_areas)} configurations...")

    # Also compute baseline at thresh=0.60, min_area=15 for explicit comparison
    baseline_row = None

    for thresh in thresholds:
        for m_dist in min_distances:
            inst_maps = watershed_cache[(thresh, m_dist)]
            for m_area in min_areas:
                cfg_id = f"t{thresh:.2f}_d{m_dist}_a{int(m_area)}"
                cfg_t0 = time.time()

                total_pred_polys = 0
                all_match_05 = []
                all_match_25 = []
                all_area_m = []
                eval_records = []

                for s_idx, s in enumerate(test_samples):
                    inst_map = inst_maps[s_idx]
                    pred_polys, p_stats = extract_watershed_polygons(inst_map, min_area=m_area, simplify_tol=0.5)
                    total_pred_polys += len(pred_polys)

                    m05 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.50)
                    m25 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.25)
                    area_m = compute_area_metrics(s["gt_polys"], pred_polys)

                    all_match_05.append(m05)
                    all_match_25.append(m25)
                    all_area_m.append(area_m)

                    eval_records.append({
                        "sample_name": s["sample_name"],
                        "base_id": s["base_id"],
                        "pred_polys": pred_polys,
                        "match_05": m05,
                        "match_25": m25,
                        "area_m": area_m
                    })

                config_sample_evals[cfg_id] = eval_records

                # Aggregates
                tp05 = sum(m["tp"] for m in all_match_05)
                fp05 = sum(m["fp"] for m in all_match_05)
                fn05 = sum(m["fn"] for m in all_match_05)
                prec05 = tp05 / (tp05 + fp05) if (tp05 + fp05) > 0 else 0.0
                rec05 = tp05 / (tp05 + fn05) if (tp05 + fn05) > 0 else 0.0
                f1_05 = (2.0 * prec05 * rec05) / (prec05 + rec05) if (prec05 + rec05) > 0 else 0.0

                tp25 = sum(m["tp"] for m in all_match_25)
                fp25 = sum(m["fp"] for m in all_match_25)
                fn25 = sum(m["fn"] for m in all_match_25)
                prec25 = tp25 / (tp25 + fp25) if (tp25 + fp25) > 0 else 0.0
                rec25 = tp25 / (tp25 + fn25) if (tp25 + fn25) > 0 else 0.0
                f1_25 = (2.0 * prec25 * rec25) / (prec25 + rec25) if (prec25 + rec25) > 0 else 0.0

                mean_area_iou = float(np.mean([a["area_iou"] for a in all_area_m]))
                mean_area_prec = float(np.mean([a["area_precision"] for a in all_area_m]))
                mean_area_rec = float(np.mean([a["area_recall"] for a in all_area_m]))
                matched_ious = [m["mean_matched_iou"] for m in all_match_05 if m["tp"] > 0]
                mean_poly_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

                cfg_elapsed = time.time() - cfg_t0

                row = {
                    "method": "Watershed",
                    "config_id": cfg_id,
                    "threshold": thresh,
                    "min_distance": m_dist,
                    "min_area": m_area,
                    "pred_polygon_count": total_pred_polys,
                    "gt_polygon_count": total_gt_polygons,
                    "matched_count_iou50": tp05,
                    "unmatched_pred_fp": fp05,
                    "missed_gt_fn": fn05,
                    "area_iou": round(mean_area_iou, 4),
                    "area_precision": round(mean_area_prec, 4),
                    "area_recall": round(mean_area_rec, 4),
                    "polygon_iou_matched": round(mean_poly_iou, 4),
                    "obj_f1_iou50": round(f1_05, 4),
                    "obj_precision_iou50": round(prec05, 4),
                    "obj_recall_iou50": round(rec05, 4),
                    "obj_f1_iou25": round(f1_25, 4),
                    "obj_precision_iou25": round(prec25, 4),
                    "obj_recall_iou25": round(rec25, 4),
                    "eval_time_s": round(cfg_elapsed, 2)
                }
                grid_results.append(row)

                print(f"  [{cfg_id}] Pred Polys: {total_pred_polys:4d} | "
                      f"Obj F1 (IoU>=0.50): {f1_05:.4f} (Prec: {prec05:.4f}, Rec: {rec05:.4f}) | "
                      f"Obj F1 (IoU>=0.25): {f1_25:.4f} (Prec: {prec25:.4f}, Rec: {rec25:.4f}) | "
                      f"Area IoU: {mean_area_iou:.4f}")

    # Add baseline row (Thresh 0.60, min_area 15, no watershed)
    base_thresh = 0.60
    base_polys_total = 0
    base_m05 = []
    base_m25 = []
    base_area_m = []
    baseline_eval_records = []

    for s in test_samples:
        b_mask = (s["prob_map"] > base_thresh).astype(np.uint8)
        b_polys, _ = extract_clean_polygons(b_mask, min_area=15.0, simplify_tol=0.5)
        base_polys_total += len(b_polys)
        bm05 = match_polygons_spatial(s["gt_polys"], b_polys, iou_threshold=0.50)
        bm25 = match_polygons_spatial(s["gt_polys"], b_polys, iou_threshold=0.25)
        b_area = compute_area_metrics(s["gt_polys"], b_polys)
        base_m05.append(bm05)
        base_m25.append(bm25)
        base_area_m.append(b_area)
        baseline_eval_records.append({
            "sample_name": s["sample_name"],
            "base_id": s["base_id"],
            "pred_polys": b_polys,
            "match_05": bm05,
            "match_25": bm25
        })

    btp05 = sum(m["tp"] for m in base_m05)
    bfp05 = sum(m["fp"] for m in base_m05)
    bfn05 = sum(m["fn"] for m in base_m05)
    bprec05 = btp05 / (btp05 + bfp05) if (btp05 + bfp05) > 0 else 0.0
    brec05 = btp05 / (btp05 + bfn05) if (btp05 + bfn05) > 0 else 0.0
    bf1_05 = (2.0 * bprec05 * brec05) / (bprec05 + brec05) if (bprec05 + brec05) > 0 else 0.0

    btp25 = sum(m["tp"] for m in base_m25)
    bfp25 = sum(m["fp"] for m in base_m25)
    bfn25 = sum(m["fn"] for m in base_m25)
    bprec25 = btp25 / (btp25 + bfp25) if (btp25 + bfp25) > 0 else 0.0
    brec25 = btp25 / (btp25 + bfn25) if (btp25 + bfn25) > 0 else 0.0
    bf1_25 = (2.0 * bprec25 * brec25) / (bprec25 + brec25) if (bprec25 + brec25) > 0 else 0.0

    b_mean_area_iou = float(np.mean([a["area_iou"] for a in base_area_m]))
    b_mean_area_prec = float(np.mean([a["area_precision"] for a in base_area_m]))
    b_mean_area_rec = float(np.mean([a["area_recall"] for a in base_area_m]))
    b_matched_ious = [m["mean_matched_iou"] for m in base_m05 if m["tp"] > 0]
    b_mean_poly_iou = float(np.mean(b_matched_ious)) if b_matched_ious else 0.0

    baseline_row = {
        "method": "Baseline (No Watershed)",
        "config_id": "baseline_t0.60_a15",
        "threshold": 0.60,
        "min_distance": 0,
        "min_area": 15.0,
        "pred_polygon_count": base_polys_total,
        "gt_polygon_count": total_gt_polygons,
        "matched_count_iou50": btp05,
        "unmatched_pred_fp": bfp05,
        "missed_gt_fn": bfn05,
        "area_iou": round(b_mean_area_iou, 4),
        "area_precision": round(b_mean_area_prec, 4),
        "area_recall": round(b_mean_area_rec, 4),
        "polygon_iou_matched": round(b_mean_poly_iou, 4),
        "obj_f1_iou50": round(bf1_05, 4),
        "obj_precision_iou50": round(bprec05, 4),
        "obj_recall_iou50": round(brec05, 4),
        "obj_f1_iou25": round(bf1_25, 4),
        "obj_precision_iou25": round(bprec25, 4),
        "obj_recall_iou25": round(brec25, 4),
        "eval_time_s": 0.0
    }

    # Prepend baseline row
    all_comparison_rows = [baseline_row] + grid_results

    # 5. Save Comparison CSV
    comp_csv_path = os.path.join(output_dir, "comparison.csv")
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_comparison_rows)

    print(f"\nSaved comprehensive comparison to: {comp_csv_path}")

    # 6. Determine Best Watershed Configuration
    # We rank based on Object F1 (IoU >= 0.50) as primary metric, breaking ties with IoU >= 0.25 F1
    best_watershed = max(grid_results, key=lambda x: (x["obj_f1_iou50"], x["obj_f1_iou25"]))
    best_cfg_id = best_watershed["config_id"]

    print("=" * 75)
    print("COMPARISON: BASELINE vs BEST WATERSHED")
    print("=" * 75)
    print(f"Metric                           Baseline (Thresh 0.60)    Best Watershed ({best_cfg_id})    Change")
    print("-" * 75)
    print(f"Predicted Polygon Count:         {baseline_row['pred_polygon_count']:<24} {best_watershed['pred_polygon_count']:<24} {best_watershed['pred_polygon_count'] - baseline_row['pred_polygon_count']:+d}")
    print(f"Matched Buildings (IoU>=0.50):   {baseline_row['matched_count_iou50']:<24} {best_watershed['matched_count_iou50']:<24} {best_watershed['matched_count_iou50'] - baseline_row['matched_count_iou50']:+d}")
    print(f"Object Recall (IoU>=0.50):       {baseline_row['obj_recall_iou50']*100:5.2f}%                  {best_watershed['obj_recall_iou50']*100:5.2f}%                  {(best_watershed['obj_recall_iou50'] - baseline_row['obj_recall_iou50'])*100:+5.2f}%")
    print(f"Object Precision (IoU>=0.50):    {baseline_row['obj_precision_iou50']*100:5.2f}%                  {best_watershed['obj_precision_iou50']*100:5.2f}%                  {(best_watershed['obj_precision_iou50'] - baseline_row['obj_precision_iou50'])*100:+5.2f}%")
    print(f"Object F1 Score (IoU>=0.50):     {baseline_row['obj_f1_iou50']:<24} {best_watershed['obj_f1_iou50']:<24} {best_watershed['obj_f1_iou50'] - baseline_row['obj_f1_iou50']:+.4f}")
    print(f"Matched Buildings (IoU>=0.25):   {baseline_row['matched_count_iou25'] if 'matched_count_iou25' in baseline_row else btp25:<24} {best_watershed['obj_recall_iou25']*total_gt_polygons:<24.0f} {best_watershed['obj_recall_iou25']*total_gt_polygons - btp25:+3.0f}")
    print(f"Object Recall (IoU>=0.25):       {baseline_row['obj_recall_iou25']*100:5.2f}%                  {best_watershed['obj_recall_iou25']*100:5.2f}%                  {(best_watershed['obj_recall_iou25'] - baseline_row['obj_recall_iou25'])*100:+5.2f}%")
    print(f"Object Precision (IoU>=0.25):    {baseline_row['obj_precision_iou25']*100:5.2f}%                  {best_watershed['obj_precision_iou25']*100:5.2f}%                  {(best_watershed['obj_precision_iou25'] - baseline_row['obj_precision_iou25'])*100:+5.2f}%")
    print(f"Object F1 Score (IoU>=0.25):     {baseline_row['obj_f1_iou25']:<24} {best_watershed['obj_f1_iou25']:<24} {best_watershed['obj_f1_iou25'] - baseline_row['obj_f1_iou25']:+.4f}")
    print(f"Area IoU:                        {baseline_row['area_iou']:<24} {best_watershed['area_iou']:<24} {best_watershed['area_iou'] - baseline_row['area_iou']:+.4f}")
    print(f"Matched Polygon IoU:             {baseline_row['polygon_iou_matched']:<24} {best_watershed['polygon_iou_matched']:<24} {best_watershed['polygon_iou_matched'] - baseline_row['polygon_iou_matched']:+.4f}")
    print("-" * 75)

    # 7. Export GeoJSON for Best Watershed Configuration
    best_evals = config_sample_evals[best_cfg_id]
    thresh_poly_dir = os.path.join(poly_export_dir, best_cfg_id)
    os.makedirs(thresh_poly_dir, exist_ok=True)

    for item in best_evals:
        base_id = item["base_id"]
        features_list = []
        for p_idx, p in enumerate(item["pred_polys"], 1):
            features_list.append({
                "type": "Feature",
                "geometry": mapping(p),
                "properties": {
                    "feature_id": p_idx,
                    "sample_name": item["sample_name"],
                    "config": best_cfg_id,
                    "area_pixels": round(p.area, 2),
                    "perimeter_pixels": round(p.length, 2),
                    "is_valid": p.is_valid
                }
            })
        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features_list
        }
        with open(os.path.join(thresh_poly_dir, f"{base_id}_watershed_buildings.geojson"), "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)

    # 8. Visualizations for At Least 12 Test Samples
    print("Generating comprehensive visual panels for 12+ diverse test samples...")

    # Build lookup maps
    ws_eval_map = {item["sample_name"]: item for item in best_evals}
    base_eval_map = {item["sample_name"]: item for item in baseline_eval_records}

    # Sort test samples by building density
    sorted_by_cov = sorted(test_samples, key=lambda x: np.sum(x["gt_mask"]) / (256 * 256))

    selected_vis = []
    selected_names = set()

    def add_case(s, cat):
        if s["sample_name"] not in selected_names:
            selected_vis.append((s, cat))
            selected_names.add(s["sample_name"])

    # 1. Empty tile
    for s in sorted_by_cov:
        if np.sum(s["gt_mask"]) == 0:
            add_case(s, "Empty Background Tile")
            break

    # 2. Sparse homesteads
    for s in sorted_by_cov:
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if 0.5 <= cov <= 2.5:
            add_case(s, "Sparse Rural Homesteads")
            if len(selected_vis) >= 3:
                break

    # 3. Dense adjacent / touching buildings
    for s in reversed(sorted_by_cov):
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if cov >= 10.0:
            add_case(s, "Dense Adjacent / Touching Buildings")
            if len([c for c in selected_vis if "Dense" in c[1]]) >= 4:
                break

    # 4. Suburban residential
    for s in sorted_by_cov:
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if 3.0 <= cov <= 8.0:
            add_case(s, "Suburban Residential Parcels")
            if len([c for c in selected_vis if "Suburban" in c[1]]) >= 4:
                break

    # Fill to 14
    for s in test_samples:
        if len(selected_vis) >= 14:
            break
        add_case(s, "Representative Test Tile")

    # Render each test sample
    for s, cat in selected_vis:
        name = s["sample_name"]
        base_id = s["base_id"]

        orig_rgb = s["raw_rgb"]
        gt_polys = s["gt_polys"]
        prob = s["prob_map"]
        bin_mask = (prob > best_watershed["threshold"]).astype(np.uint8) * 255

        ws_polys = ws_eval_map[name]["pred_polys"]
        base_polys = base_eval_map[name]["pred_polys"]
        match_ws = ws_eval_map[name]["match_05"]

        # View 1: Original Image
        Image.fromarray(orig_rgb).save(os.path.join(vis_export_dir, f"{base_id}_1_original.png"))

        # View 2: Probability Map (Magma colormap)
        prob_u8 = np.clip(prob * 255, 0, 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_MAGMA)
        prob_rgb = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(prob_rgb).save(os.path.join(vis_export_dir, f"{base_id}_2_prob_map.png"))

        # View 3: Binary Mask
        bin_rgb = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)
        Image.fromarray(bin_rgb).save(os.path.join(vis_export_dir, f"{base_id}_3_binary_mask.png"))

        # View 4: Baseline Polygons (merged)
        base_rendered = render_vector_view((256, 256), base_polys, fill_color=(255, 255, 255), border_color=(255, 80, 80))
        Image.fromarray(base_rendered).save(os.path.join(vis_export_dir, f"{base_id}_4_baseline_polygons.png"))

        # View 5: Watershed Polygons (separated)
        ws_rendered = render_vector_view((256, 256), ws_polys, fill_color=(255, 255, 255), border_color=(40, 200, 255))
        Image.fromarray(ws_rendered).save(os.path.join(vis_export_dir, f"{base_id}_5_watershed_polygons.png"))

        # View 6: Ground Truth Polygons
        gt_rendered = render_vector_view((256, 256), gt_polys, fill_color=(255, 255, 255), border_color=(40, 255, 120))
        Image.fromarray(gt_rendered).save(os.path.join(vis_export_dir, f"{base_id}_6_gt_polygons.png"))

        # View 7: GT vs Watershed Match Overlay
        match_overlay = render_comparison_overlay(orig_rgb, gt_polys, ws_polys, match_ws)
        Image.fromarray(match_overlay).save(os.path.join(vis_export_dir, f"{base_id}_7_match_overlay.png"))

        # View 8: Direct Zoomed Split Comparison
        # Layout: BASELINE (merged) vs WATERSHED (separated) side by side
        comp_ws_base = np.zeros((256 + 32, 256 * 2, 3), dtype=np.uint8)
        comp_ws_base[:32, :] = [20, 24, 32]
        comp_ws_base[32:, 0:256] = base_rendered
        comp_ws_base[32:, 256:512] = ws_rendered

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comp_ws_base, f"BASELINE: Merged ({len(base_polys)} polys)", (10, 21), font, 0.40, (255, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(comp_ws_base, f"WATERSHED: Split ({len(ws_polys)} polys)", (266, 21), font, 0.40, (40, 220, 255), 1, cv2.LINE_AA)
        cv2.line(comp_ws_base, (256, 0), (256, 288), (60, 70, 85), 1)
        Image.fromarray(comp_ws_base).save(os.path.join(vis_export_dir, f"{base_id}_baseline_vs_watershed.png"))

        # View 9: Full 7-Panel Composite
        # Layout: Aerial | Prob | Mask | Baseline Polys | Watershed Polys | GT Polys | Match Overlay
        header_h = 32
        comp_w = 256 * 7
        comp_h = 256 + header_h
        composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
        composite[:header_h, :] = [20, 24, 32]

        panels = [
            (orig_rgb, "1. Aerial Image", (220, 225, 235)),
            (prob_rgb, "2. Prob Map", (240, 180, 80)),
            (bin_rgb, "3. Binary Mask", (200, 200, 200)),
            (base_rendered, f"4. Baseline ({len(base_polys)})", (255, 90, 90)),
            (ws_rendered, f"5. Watershed ({len(ws_polys)})", (40, 220, 255)),
            (gt_rendered, f"6. GT ({len(gt_polys)})", (40, 255, 120)),
            (match_overlay, f"7. Match F1: {match_ws['f1']}", (255, 220, 80))
        ]

        for p_idx, (p_img, p_title, p_col) in enumerate(panels):
            x_offset = p_idx * 256
            composite[header_h:, x_offset:x_offset+256] = p_img
            cv2.putText(composite, p_title, (x_offset + 10, 21), font, 0.38, p_col, 1, cv2.LINE_AA)
            if p_idx > 0:
                cv2.line(composite, (x_offset, 0), (x_offset, comp_h), (50, 58, 70), 1)

        Image.fromarray(composite).save(os.path.join(vis_export_dir, f"{base_id}_comparison_full.png"))

    print(f"Generated visual panels for {len(selected_vis)} cases in: {vis_export_dir}")

    # 9. Save Best Config JSON and Comprehensive Report JSON
    best_config_dict = {
        "method": "Marker-Controlled Watershed on Euclidean Distance Transform",
        "best_config_id": best_cfg_id,
        "probability_threshold": best_watershed["threshold"],
        "min_distance_peak": best_watershed["min_distance"],
        "min_polygon_area_pixels": best_watershed["min_area"],
        "metrics": best_watershed
    }
    with open(os.path.join(output_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config_dict, f, indent=2)

    report_dict = {
        "experiment": "Phase 4 Watershed Instance Separation",
        "baseline_threshold_0.60": baseline_row,
        "best_watershed_configuration": best_watershed,
        "comparison_delta": {
            "polygon_count_delta": best_watershed["pred_polygon_count"] - baseline_row["pred_polygon_count"],
            "matched_buildings_delta_iou50": best_watershed["matched_count_iou50"] - baseline_row["matched_count_iou50"],
            "object_recall_delta_iou50": round(best_watershed["obj_recall_iou50"] - baseline_row["obj_recall_iou50"], 4),
            "object_precision_delta_iou50": round(best_watershed["obj_precision_iou50"] - baseline_row["obj_precision_iou50"], 4),
            "object_f1_delta_iou50": round(best_watershed["obj_f1_iou50"] - baseline_row["obj_f1_iou50"], 4),
            "object_recall_delta_iou25": round(best_watershed["obj_recall_iou25"] - baseline_row["obj_recall_iou25"], 4),
            "object_precision_delta_iou25": round(best_watershed["obj_precision_iou25"] - baseline_row["obj_precision_iou25"], 4),
            "object_f1_delta_iou25": round(best_watershed["obj_f1_iou25"] - baseline_row["obj_f1_iou25"], 4),
            "area_iou_delta": round(best_watershed["area_iou"] - baseline_row["area_iou"], 4)
        },
        "all_configurations_tested": len(grid_results),
        "comparison_csv_path": comp_csv_path,
        "polygons_dir": thresh_poly_dir,
        "visualizations_dir": vis_export_dir
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("=" * 75)
    print("PHASE 4 WATERSHED EXPERIMENT COMPLETE!")
    print(f"Report saved to: {os.path.join(output_dir, 'report.json')}")
    print("=" * 75)

    return report_dict


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 7 Watershed Instance Separation")
    parser.add_argument("--checkpoint", default="results/experiments/baseline_full/best_model.pth")
    parser.add_argument("--test-splits", default="data/processed_spacenet7/splits/test.txt")
    parser.add_argument("--output-dir", default="results/experiments/watershed")
    args = parser.parse_args()

    run_watershed_experiment(
        model_checkpoint=args.checkpoint,
        test_splits_file=args.test_splits,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
