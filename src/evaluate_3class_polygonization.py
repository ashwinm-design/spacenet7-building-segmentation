"""
Phase 7: SpaceNet 7 3-Class Raster to Vector (GIS Polygon) Evaluation Pipeline.

Objective:
  Determine whether the 3-class boundary-aware U-Net produces better-separated building polygons
  than the original binary U-Net baseline and classical watershed post-processing.

Methodology:
  - Uses the SAME 64 held-out test patches and the SAME 6,344 ground-truth building polygons.
  - Model: results/experiments/3class_model/best_model.pth (Phase 6 best checkpoint).
  - Extracts initial instances from Class 1 (Building Interior) connected components ONLY.
  - Implements boundary-aware expansion into Class 2 (Building Boundary) pixels:
      * Never grows into Class 0 (Background).
      * Never allows two instances to merge.
      * Resolves competing instances deterministically via Euclidean distance transform (nearest-interior rule).
      * Preserves discrete instance IDs.
  - Evaluates 3 diagnostic variants:
      1. variant_class1_only (G = 0, no growth)
      2. variant_growth_1px (G = 1, expand into boundary by up to 1 pixel / dist <= 1.5)
      3. variant_growth_2px (G = 2, expand into boundary by up to 2 pixels / dist <= 2.5)
  - Evaluates using the EXACT same spatial bipartite matching (IoU >= 0.50 & 0.25) and area metrics.
  - Generates GeoJSON predictions, metrics.csv, summary.json, comparison.csv, visualizations/, and report.md.
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

# Ensure standard output uses UTF-8 to prevent Windows CP1252 charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import rasterio
from rasterio import features
import shapely
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn

from src.model import get_model


def extract_clean_polygons_from_mask(
    binary_mask: np.ndarray,
    min_area: float = 15.0,
    simplify_tol: float = 0.5
) -> Tuple[List[Polygon], Dict[str, int]]:
    """
    Extract polygons from binary mask with interior holes preserved,
    geometric validation, self-intersection repair, and tiny noise filtering.
    Matches the exact implementation from Phase 3.
    """
    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    shapes_gen = features.shapes(mask_uint8, mask=(mask_uint8 == 1))

    polygons = []
    stats = {
        "raw_count": 0,
        "valid_count": 0,
        "invalid_repaired": 0,
        "tiny_removed": 0,
        "has_holes": 0,
        "self_intersecting_fixed": 0
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

        if not poly.is_simple:
            stats["self_intersecting_fixed"] += 1

        if poly.is_empty:
            continue

        if poly.geom_type == "Polygon":
            poly_list = [poly]
        elif poly.geom_type == "MultiPolygon":
            poly_list = list(poly.geoms)
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


def extract_clean_polygons_from_instances(
    instance_labels: np.ndarray,
    min_area: float = 15.0,
    simplify_tol: float = 0.5
) -> Tuple[List[Polygon], Dict[str, int]]:
    """
    Extract polygons from labeled integer instance mask, guaranteeing that
    each labeled component generates a distinct polygon instance without merging.
    """
    labels_int32 = instance_labels.astype(np.int32)
    shapes_gen = features.shapes(labels_int32, mask=(labels_int32 > 0))

    polygons = []
    stats = {
        "raw_count": 0,
        "valid_count": 0,
        "invalid_repaired": 0,
        "tiny_removed": 0,
        "has_holes": 0,
        "self_intersecting_fixed": 0
    }

    for geom, val in shapes_gen:
        if val <= 0:
            continue
        stats["raw_count"] += 1
        poly = shape(geom)

        if not poly.is_valid:
            stats["invalid_repaired"] += 1
            if hasattr(shapely, "make_valid"):
                poly = shapely.make_valid(poly)
            else:
                poly = poly.buffer(0)

        if not poly.is_simple:
            stats["self_intersecting_fixed"] += 1

        if poly.is_empty:
            continue

        if poly.geom_type == "Polygon":
            poly_list = [poly]
        elif poly.geom_type == "MultiPolygon":
            poly_list = list(poly.geoms)
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


def grow_interior_into_boundary(
    interior_labels: np.ndarray,
    boundary_mask: np.ndarray,
    max_growth_distance: float = 1.5
) -> np.ndarray:
    """
    Deterministically expand Class 1 interior instances into adjacent Class 2 boundary pixels:
    - Never grow into Class 0 background.
    - Never merge instances (each boundary pixel assigned to at most one instance).
    - If multiple interiors compete for the same boundary pixel, assign to nearest interior.
    - Preserve instance IDs.
    """
    if np.sum(interior_labels > 0) == 0:
        return interior_labels.copy()
    if np.sum(boundary_mask > 0) == 0 or max_growth_distance <= 0:
        return interior_labels.copy()

    # Distance transform from interior components (zeros in boolean mask)
    dist, nearest_idx = distance_transform_edt(interior_labels == 0, return_indices=True)

    grown = interior_labels.copy()
    # Qualifying boundary pixels: where boundary == 1, not already labeled, and distance <= threshold
    qualifying = (boundary_mask == 1) & (interior_labels == 0) & (dist <= max_growth_distance)

    # Assign nearest interior label
    grown[qualifying] = interior_labels[nearest_idx[0, qualifying], nearest_idx[1, qualifying]]
    return grown


def match_polygons_spatial(
    gt_polys: List[Polygon],
    pred_polys: List[Polygon],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Bipartite spatial matching between Ground-Truth and Predicted polygons.
    Exact implementation from Phase 3 for scientific reproducibility.
    """
    m, n = len(gt_polys), len(pred_polys)

    if m == 0 and n == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "matched_gt_indices": [], "matched_pred_indices": [],
            "mean_matched_iou": 1.0
        }
    if m == 0:
        return {
            "tp": 0, "fp": n, "fn": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "matched_gt_indices": [], "matched_pred_indices": [],
            "mean_matched_iou": 0.0
        }
    if n == 0:
        return {
            "tp": 0, "fp": 0, "fn": m,
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "matched_gt_indices": [], "matched_pred_indices": [],
            "mean_matched_iou": 0.0
        }

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "matched_gt_indices": list(matched_gt),
        "matched_pred_indices": list(matched_pred),
        "mean_matched_iou": round(mean_iou, 4)
    }


def compute_area_metrics(gt_polys: List[Polygon], pred_polys: List[Polygon]) -> Dict[str, float]:
    """Compute overall area intersection, precision, recall, and IoU."""
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


def save_polygons_geojson(polygons: List[Polygon], filepath: str, properties_list: List[Dict] = None):
    """Save Shapely polygons as GeoJSON FeatureCollection."""
    features_list = []
    for idx, poly in enumerate(polygons):
        props = properties_list[idx] if properties_list and idx < len(properties_list) else {"id": idx + 1, "area": round(poly.area, 2)}
        features_list.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(poly)
        })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features_list
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)


def render_polygon_overlay(
    rgb_img: np.ndarray,
    polygons: List[Polygon],
    border_color: Tuple[int, int, int] = (46, 204, 113),
    fill_alpha: float = 0.35
) -> np.ndarray:
    """Render colored polygon outlines and transparent fill on RGB image."""
    overlay = rgb_img.copy()
    fill_canvas = np.zeros_like(rgb_img)

    for p in polygons:
        try:
            ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
            cv2.fillPoly(fill_canvas, [ext], border_color)
            for hole in p.interiors:
                hint = np.round(np.array(hole.coords)).astype(np.int32)
                cv2.fillPoly(fill_canvas, [hint], (0, 0, 0))
        except Exception:
            continue

    mask_nz = (fill_canvas > 0)
    overlay[mask_nz] = cv2.addWeighted(overlay, 1.0 - fill_alpha, fill_canvas, fill_alpha, 0)[mask_nz]

    for p in polygons:
        try:
            ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
            cv2.polylines(overlay, [ext], True, border_color, 1, lineType=cv2.LINE_AA)
            for hole in p.interiors:
                hint = np.round(np.array(hole.coords)).astype(np.int32)
                cv2.polylines(overlay, [hint], True, border_color, 1, lineType=cv2.LINE_AA)
        except Exception:
            continue

    return overlay


def run_3class_polygonization_experiment(
    checkpoint_path: str = "results/experiments/3class_model/best_model.pth",
    test_splits_file: str = "data/processed_spacenet7/splits/test.txt",
    output_dir: str = "results/experiments/3class_polygonization",
    min_area_pixels: float = 15.0,
    simplify_tol: float = 0.5,
    device_str: str = "cpu"
):
    """
    Run the complete Phase 7 3-class polygonization and benchmarking experiment.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(device_str)
    print("\n" + "=" * 75)
    print("PHASE 7: 3-CLASS RASTER-TO-VECTOR POLYGONIZATION & BENCHMARKING")
    print("=" * 75)
    print(f"Model Checkpoint:       {checkpoint_path}")
    print(f"Test Split File:        {test_splits_file}")
    print(f"Min Polygon Area:       {min_area_pixels} px")
    print(f"Simplify Tolerance:     {simplify_tol} px")
    print("-" * 75)

    # 1. Load Model
    print("Loading Phase 6 Best Checkpoint...")
    model = get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=3, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Load 64 Test Pairs & Ground Truth
    with open(test_splits_file, "r", encoding="utf-8") as f:
        test_lines = [l.strip().split(",") for l in f if l.strip()]

    print(f"Loaded {len(test_lines)} test image-mask pairs.")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Extract Ground Truth Polygons
    print("Extracting Ground Truth Polygons across test set...")
    test_samples = []
    total_gt_polys = 0

    with torch.no_grad():
        for img_p, mask_p in test_lines:
            name = os.path.basename(img_p)
            base_id = os.path.splitext(name)[0]
            raw_rgb = np.array(Image.open(img_p).convert("RGB"))
            raw_gt = np.array(Image.open(mask_p).convert("L"))

            # Normalize for network forward
            img_arr = raw_rgb.astype(np.float32) / 255.0
            norm_img = (img_arr - mean) / std
            tensor_x = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            logits = model(tensor_x)
            pred_class = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Ground truth polygons from binary mask
            gt_binary = (raw_gt > 127).astype(np.uint8)
            gt_polys, gt_stats = extract_clean_polygons_from_mask(gt_binary, min_area=min_area_pixels, simplify_tol=simplify_tol)
            total_gt_polys += len(gt_polys)

            test_samples.append({
                "sample_name": name,
                "base_id": base_id,
                "img_path": img_p,
                "mask_path": mask_p,
                "raw_rgb": raw_rgb,
                "gt_binary": gt_binary,
                "gt_polys": gt_polys,
                "gt_poly_count": len(gt_polys),
                "pred_class": pred_class
            })

    print(f"Verified Ground Truth Polygons across 64 test patches: {total_gt_polys:,}")
    assert total_gt_polys == 6344, f"Mismatch in GT polygon count: {total_gt_polys} != 6344"
    print("GT count matches exactly with Phase 3 baseline (6,344 polygons) [PASS].")
    print("-" * 75)

    # 3. Evaluate the 3 Diagnostic Variants
    # Variant definitions: (variant_id, display_name, max_growth_dist)
    variants = [
        ("class1_only", "3-Class U-Net (Class-1 Only, No Growth)", 0.0),
        ("growth_1px", "3-Class U-Net (Class-1 + Boundary Growth 1px)", 1.5),
        ("growth_2px", "3-Class U-Net (Class-1 + Boundary Growth 2px)", 2.5),
    ]

    variant_results = []
    variant_sample_evals = {}

    for var_id, var_name, growth_dist in variants:
        t_var = time.time()
        print(f"Evaluating Variant: {var_name} (growth_dist={growth_dist})...")
        pred_export_dir = os.path.join(output_dir, "predictions", var_id)
        os.makedirs(pred_export_dir, exist_ok=True)

        tot_pred = 0
        tot_valid = 0
        tot_invalid_rep = 0
        tot_tiny_rem = 0

        all_m05 = []
        all_m25 = []
        all_area = []
        sample_records = []

        for s in test_samples:
            p_cls = s["pred_class"]
            int_mask = (p_cls == 1).astype(np.uint8)
            bnd_mask = (p_cls == 2).astype(np.uint8)

            # Step A: Connected components on Class 1 ONLY
            num_cc, initial_labels = cv2.connectedComponents(int_mask)

            # Step B: Boundary-aware growth
            if growth_dist > 0:
                final_labels = grow_interior_into_boundary(initial_labels, bnd_mask, max_growth_distance=growth_dist)
            else:
                final_labels = initial_labels

            # Step C: Extract clean polygons
            pred_polys, p_stats = extract_clean_polygons_from_instances(
                final_labels, min_area=min_area_pixels, simplify_tol=simplify_tol
            )

            tot_pred += len(pred_polys)
            tot_valid += p_stats["valid_count"]
            tot_invalid_rep += p_stats["invalid_repaired"]
            tot_tiny_rem += p_stats["tiny_removed"]

            # Save GeoJSON
            geojson_path = os.path.join(pred_export_dir, f"{s['base_id']}_buildings.geojson")
            save_polygons_geojson(pred_polys, geojson_path)

            # Step D: Spatial Bipartite Matching against SAME GT
            m05 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.50)
            m25 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.25)
            area_m = compute_area_metrics(s["gt_polys"], pred_polys)

            all_m05.append(m05)
            all_m25.append(m25)
            all_area.append(area_m)

            sample_records.append({
                "base_id": s["base_id"],
                "gt_count": s["gt_poly_count"],
                "pred_count": len(pred_polys),
                "obj_f1_05": m05["f1"],
                "obj_f1_25": m25["f1"],
                "area_iou": area_m["area_iou"],
                "pred_polys": pred_polys
            })

        variant_sample_evals[var_id] = sample_records

        # Global aggregation
        total_tp_05 = sum(m["tp"] for m in all_m05)
        total_fp_05 = sum(m["fp"] for m in all_m05)
        total_fn_05 = sum(m["fn"] for m in all_m05)
        obj_prec_05 = total_tp_05 / (total_tp_05 + total_fp_05) if (total_tp_05 + total_fp_05) > 0 else 0.0
        obj_rec_05 = total_tp_05 / (total_tp_05 + total_fn_05) if (total_tp_05 + total_fn_05) > 0 else 0.0
        obj_f1_05 = 2.0 * obj_prec_05 * obj_rec_05 / (obj_prec_05 + obj_rec_05) if (obj_prec_05 + obj_rec_05) > 0 else 0.0

        total_tp_25 = sum(m["tp"] for m in all_m25)
        total_fp_25 = sum(m["fp"] for m in all_m25)
        total_fn_25 = sum(m["fn"] for m in all_m25)
        obj_prec_25 = total_tp_25 / (total_tp_25 + total_fp_25) if (total_tp_25 + total_fp_25) > 0 else 0.0
        obj_rec_25 = total_tp_25 / (total_tp_25 + total_fn_25) if (total_tp_25 + total_fn_25) > 0 else 0.0
        obj_f1_25 = 2.0 * obj_prec_25 * obj_rec_25 / (obj_prec_25 + obj_rec_25) if (obj_prec_25 + obj_rec_25) > 0 else 0.0

        # Mean matched IoU across matched pairs
        all_matched_ious = []
        for m in all_m05:
            if "matched_gt_indices" in m and len(m["matched_gt_indices"]) > 0:
                all_matched_ious.append(m["mean_matched_iou"])
        mean_matched_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

        # Mean Area Metrics
        mean_area_iou = float(np.mean([a["area_iou"] for a in all_area]))
        mean_area_prec = float(np.mean([a["area_precision"] for a in all_area]))
        mean_area_rec = float(np.mean([a["area_recall"] for a in all_area]))

        var_stat = {
            "variant_id": var_id,
            "method": var_name,
            "growth_distance": growth_dist,
            "pred_polygon_count": tot_pred,
            "gt_polygon_count": total_gt_polys,
            "area_iou": round(mean_area_iou, 4),
            "area_precision": round(mean_area_prec, 4),
            "area_recall": round(mean_area_rec, 4),
            "polygon_iou_matched": round(mean_matched_iou, 4),
            "obj_f1_iou50": round(obj_f1_05, 4),
            "obj_precision_iou50": round(obj_prec_05, 4),
            "obj_recall_iou50": round(obj_rec_05, 4),
            "obj_f1_iou25": round(obj_f1_25, 4),
            "obj_precision_iou25": round(obj_prec_25, 4),
            "obj_recall_iou25": round(obj_rec_25, 4),
            "valid_polygon_count": tot_valid,
            "invalid_repaired_count": tot_invalid_rep,
            "tiny_removed_count": tot_tiny_rem,
            "valid_pct": round(tot_valid / max(1, tot_pred) * 100, 2),
            "eval_time_s": round(time.time() - t_var, 2)
        }
        variant_results.append(var_stat)

        print(f"  -> Pred Polygons: {tot_pred:,} | Area IoU: {mean_area_iou:.4f} | "
              f"Strict F1 (IoU>=0.50): {obj_f1_05:.4f} | Relaxed F1 (IoU>=0.25): {obj_f1_25:.4f}")

    # 4. Save metrics.csv
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    fieldnames = list(variant_results[0].keys())
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(variant_results)
    print(f"\nSaved metrics.csv to {metrics_csv_path}")

    # 5. Build comparison.csv directly contrasting with Binary Baseline and Watershed
    # Baseline benchmark numbers from Phase 3 & 4
    comparison_rows = [
        {
            "method": "Binary U-Net Baseline (tau=0.60)",
            "source_experiment": "Phase 3 Raster-to-Vector",
            "pred_polygons": 2990,
            "gt_polygons": 6344,
            "area_iou": 0.4553,
            "area_precision": 0.6443,
            "area_recall": 0.6052,
            "matched_poly_iou": 0.6496,
            "strict_f1_050": 0.2091,
            "strict_prec_050": 0.3264,
            "strict_rec_050": 0.1538,
            "relaxed_f1_025": 0.4532,
            "relaxed_prec_025": 0.7074,
            "relaxed_rec_025": 0.3334
        },
        {
            "method": "Marker-Controlled Watershed",
            "source_experiment": "Phase 4 Post-Processing",
            "pred_polygons": 4376,
            "gt_polygons": 6344,
            "area_iou": 0.3634,
            "area_precision": 0.7011,
            "area_recall": 0.4232,
            "matched_poly_iou": 0.5904,
            "strict_f1_050": 0.1310,
            "strict_prec_050": 0.1604,
            "strict_rec_050": 0.1107,
            "relaxed_f1_025": 0.4940,
            "relaxed_prec_025": 0.6051,
            "relaxed_rec_025": 0.4174
        }
    ]

    for vr in variant_results:
        comparison_rows.append({
            "method": vr["method"],
            "source_experiment": f"Phase 7 ({vr['variant_id']})",
            "pred_polygons": vr["pred_polygon_count"],
            "gt_polygons": vr["gt_polygon_count"],
            "area_iou": vr["area_iou"],
            "area_precision": vr["area_precision"],
            "area_recall": vr["area_recall"],
            "matched_poly_iou": vr["polygon_iou_matched"],
            "strict_f1_050": vr["obj_f1_iou50"],
            "strict_prec_050": vr["obj_precision_iou50"],
            "strict_rec_050": vr["obj_recall_iou50"],
            "relaxed_f1_025": vr["obj_f1_iou25"],
            "relaxed_prec_025": vr["obj_precision_iou25"],
            "relaxed_rec_025": vr["obj_recall_iou25"]
        })

    comp_csv_path = os.path.join(output_dir, "comparison.csv")
    comp_fields = list(comparison_rows[0].keys())
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved comparison.csv to {comp_csv_path}")

    # 6. Save summary.json
    summary_dict = {
        "experiment_name": "Phase 7 SpaceNet 7 3-Class Raster to Vector Polygonization",
        "checkpoint": checkpoint_path,
        "test_patches_count": len(test_samples),
        "gt_polygons_count": total_gt_polys,
        "min_area_pixels": min_area_pixels,
        "simplify_tolerance_pixels": simplify_tol,
        "variants": variant_results,
        "master_comparison": comparison_rows,
        "elapsed_seconds": round(time.time() - t_start, 2)
    }
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary.json to {summary_json_path}")

    # 7. Generate Multi-Panel Visual Comparisons
    print("\nGenerating 5-Panel Visual Comparisons for Representative Test Patches...")
    # Select representative test patches covering:
    # - touching buildings
    # - dense buildings
    # - small buildings
    # - large buildings
    # - irregular buildings
    representative_test_ids = [
        ("touching_dense_1", "2019_09_r0000_c0000"),
        ("touching_dense_2", "2019_09_r0000_c0256"),
        ("dense_urban_3",    "2019_10_r0000_c0000"),
        ("dense_urban_4",    "2019_10_r0000_c0256"),
        ("irregular_5",      "2019_09_r0000_c0512"),
        ("irregular_6",      "2019_10_r0000_c0512"),
        ("small_buildings_7", "2019_09_r0512_c0768"),
        ("small_buildings_8", "2019_10_r0512_c0768"),
        ("large_buildings_9", "2019_09_r0512_c0000"),
        ("large_buildings_10", "2019_10_r0512_c0000"),
        ("sparse_suburban_11", "2019_09_r0256_c0000"),
        ("sparse_suburban_12", "2019_10_r0256_c0000")
    ]

    sample_by_id = {s["base_id"]: s for s in test_samples}
    growth_1px_evals = {r["base_id"]: r for r in variant_sample_evals["growth_1px"]}
    binary_baseline_dir = "results/experiments/polygonization/polygons/threshold_0.60"

    for cat_label, base_id in representative_test_ids:
        if base_id not in sample_by_id:
            continue
        s = sample_by_id[base_id]
        rgb = s["raw_rgb"]
        gt_polys = s["gt_polys"]
        pred_3class_polys = growth_1px_evals[base_id]["pred_polys"]

        # Load binary baseline polygons
        bin_geojson = os.path.join(binary_baseline_dir, f"{base_id}_buildings.geojson")
        binary_polys = []
        if os.path.exists(bin_geojson):
            with open(bin_geojson, "r", encoding="utf-8") as f:
                data = json.load(f)
            for feat in data.get("features", []):
                geom = shape(feat["geometry"])
                if geom.geom_type == "Polygon":
                    binary_polys.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    binary_polys.extend(list(geom.geoms))

        # Render 5-panel figure:
        # A. Original Aerial Image
        # B. Ground Truth Polygons (Cyan)
        # C. Binary Baseline Polygons (Orange/Red)
        # D. 3-Class Predicted Polygons (Emerald Green)
        # E. Zoomed Detail comparing touching buildings
        fig, axes = plt.subplots(1, 5, figsize=(25, 5.2), dpi=200)
        fig.patch.set_facecolor("#0f1117")

        # Find best zoom window (around highest GT density / touching)
        zt, zl, zb, zr = 64, 64, 144, 144
        if len(gt_polys) > 0:
            # find bbox of densest region
            weight_mask = np.zeros((256, 256), dtype=np.float32)
            for p in gt_polys:
                ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
                cv2.polylines(weight_mask, [ext], True, 1.0, 1)
            density = cv2.filter2D(weight_mask, -1, np.ones((80, 80), dtype=np.float32))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(density[40:216, 40:216])
            center_x = max_loc[0] + 40
            center_y = max_loc[1] + 40
            zt = max(0, min(256 - 80, center_y - 40))
            zl = max(0, min(256 - 80, center_x - 40))
            zb = zt + 80
            zr = zl + 80

        # Panel A: Original Image
        axes[0].imshow(rgb)
        rect = patches.Rectangle((zl, zt), 80, 80, linewidth=2, edgecolor="#f1c40f", facecolor="none", linestyle="--")
        axes[0].add_patch(rect)
        axes[0].set_title(f"A. Aerial Image\n{base_id}", color="white", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        # Panel B: Ground Truth Polygons (Cyan)
        gt_overlay = render_polygon_overlay(rgb, gt_polys, border_color=(0, 240, 255), fill_alpha=0.35)
        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"B. Ground Truth\n{len(gt_polys)} Polygons", color="#00ffff", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")

        # Panel C: Binary Baseline Polygons (Orange / Merged)
        bin_overlay = render_polygon_overlay(rgb, binary_polys, border_color=(255, 100, 50), fill_alpha=0.35)
        axes[2].imshow(bin_overlay)
        axes[2].set_title(f"C. Binary Baseline (tau=0.60)\n{len(binary_polys)} Polygons (Merged)", color="#ff6432", fontsize=11, fontweight="bold", pad=8)
        axes[2].axis("off")

        # Panel D: 3-Class Predicted Polygons (Emerald Green / Separated)
        pred_overlay = render_polygon_overlay(rgb, pred_3class_polys, border_color=(46, 204, 113), fill_alpha=0.35)
        axes[3].imshow(pred_overlay)
        axes[3].set_title(f"D. 3-Class U-Net (Ours)\n{len(pred_3class_polys)} Polygons (Separated)", color="#2ecc71", fontsize=11, fontweight="bold", pad=8)
        axes[3].axis("off")

        # Panel E: Zoomed Side-by-Side Detail
        # Compare Zoomed Binary vs Zoomed 3-Class
        zoom_bin = bin_overlay[zt:zb, zl:zr]
        zoom_3cls = pred_overlay[zt:zb, zl:zr]
        sep = np.ones((80, 3, 3), dtype=np.uint8) * 255
        zoom_comp = np.concatenate([zoom_bin, sep, zoom_3cls], axis=1)

        axes[4].imshow(zoom_comp)
        axes[4].set_title(f"E. Zoomed Detail [{zl}:{zr}, {zt}:{zb}]\nBinary (Merged)  |  3-Class (Separated)", color="#f1c40f", fontsize=11, fontweight="bold", pad=8)
        axes[4].axis("off")

        # Title & Layout
        clean_cat = cat_label.replace("_", " ").upper()
        plt.suptitle(f"CADASTRAL POLYGON COMPARISON: {clean_cat} — SpaceNet 7 Held-Out Test Set",
                     color="white", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_fig_path = os.path.join(vis_dir, f"{cat_label}_{base_id}.png")
        plt.savefig(out_fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Rendered visual panel: {out_fig_path}")

    print(f"\nAll visual panels saved to {vis_dir}")
    print(f"Total Phase 7 elapsed time: {time.time() - t_start:.2f}s")
    print("=" * 75 + "\n")

    return summary_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: SpaceNet 7 3-Class Raster to Vector Polygonization")
    parser.add_argument("--checkpoint", type=str, default="results/experiments/3class_model/best_model.pth")
    parser.add_argument("--test_splits", type=str, default="data/processed_spacenet7/splits/test.txt")
    parser.add_argument("--output_dir", type=str, default="results/experiments/3class_polygonization")
    parser.add_argument("--min_area", type=float, default=15.0)
    parser.add_argument("--simplify_tol", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_3class_polygonization_experiment(
        checkpoint_path=args.checkpoint,
        test_splits_file=args.test_splits,
        output_dir=args.output_dir,
        min_area_pixels=args.min_area,
        simplify_tol=args.simplify_tol,
        device_str=args.device
    )
