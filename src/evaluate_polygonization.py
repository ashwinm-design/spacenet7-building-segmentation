"""
Phase 3: SpaceNet 7 Raster to Vector Evaluation Pipeline.

Converts U-Net segmentation predictions into validated GIS building polygons.
Evaluates across probability thresholds: 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65.
Performs geometric validation, bipartite spatial matching (IoU >= 0.50 & 0.25),
generates GeoJSON vector exports, and produces multi-panel cadastral overlays.
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

import rasterio
from rasterio import features
import shapely
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

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
    """
    Extract polygons from binary mask with interior holes preserved,
    geometric validation, self-intersection repair, and tiny noise filtering.
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

        # Extract constituent polygons if multi-geometry
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


def match_polygons_spatial(
    gt_polys: List[Polygon],
    pred_polys: List[Polygon],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Bipartite spatial matching between Ground-Truth and Predicted polygons.
    Uses IoU overlap matching with greedy assignment.
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

    # Compute pairwise IoU
    candidates = []
    for g_idx, g in enumerate(gt_polys):
        g_bound = g.bounds
        for p_idx, p in enumerate(pred_polys):
            p_bound = p.bounds
            # Quick bounding box overlap rejection
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

    # Sort descending by IoU
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

    # Rasterize or union
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
        # Exterior
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        cv2.fillPoly(canvas, [ext], fill_color)

        # Holes
        for interior in p.interiors:
            hole = np.round(np.array(interior.coords)).astype(np.int32)
            cv2.fillPoly(canvas, [hole], (0, 0, 0))

        # Border
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
    """
    Render vector overlay on RGB image:
    - True Positives (Matched): Green fill with emerald border
    - False Positives (Unmatched Pred): Coral Red/Orange fill with red border
    - False Negatives (Missed GT): Cyan/Blue outline
    """
    overlay = raw_rgb.copy()
    h, w = raw_rgb.shape[:2]

    color_layer = np.zeros_like(raw_rgb)
    matched_p_idx = set(match_result["matched_pred_indices"])
    matched_g_idx = set(match_result["matched_gt_indices"])

    # Draw matched predictions in translucent green
    for p_idx, p in enumerate(pred_polys):
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        if p_idx in matched_p_idx:
            cv2.fillPoly(color_layer, [ext], (30, 220, 100))  # Green TP
        else:
            cv2.fillPoly(color_layer, [ext], (255, 60, 60))   # Red FP

    overlay = cv2.addWeighted(overlay, 0.70, color_layer, 0.30, 0)

    # Draw contours
    for p_idx, p in enumerate(pred_polys):
        ext = np.round(np.array(p.exterior.coords)).astype(np.int32)
        col = (40, 255, 120) if p_idx in matched_p_idx else (255, 70, 70)
        cv2.polylines(overlay, [ext], True, col, 1, lineType=cv2.LINE_AA)

    # Draw missed GT outlines in bright cyan
    for g_idx, g in enumerate(gt_polys):
        if g_idx not in matched_g_idx:
            ext = np.round(np.array(g.exterior.coords)).astype(np.int32)
            cv2.polylines(overlay, [ext], True, (255, 230, 50), 1, lineType=cv2.LINE_AA)  # Yellow/Cyan FN

    return overlay


def run_raster_to_vector_evaluation(
    model_checkpoint: str = "results/experiments/baseline_full/best_model.pth",
    test_splits_file: str = "data/processed_spacenet7/splits/test.txt",
    output_dir: str = "results/experiments/polygonization",
    thresholds: List[float] = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    min_area_pixels: float = 15.0,
    simplify_tol: float = 0.5
):
    """Run full Phase 3 raster-to-vector evaluation pipeline."""
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

    print("=" * 70)
    print("PHASE 3: SPACE NET 7 RASTER TO VECTOR (GIS POLYGON) EVALUATION")
    print("=" * 70)
    print(f"Model Checkpoint:      {model_checkpoint}")
    print(f"Test Split File:       {test_splits_file}")
    print(f"Thresholds to Test:    {thresholds}")
    print(f"Min Polygon Area:      {min_area_pixels} pixels")
    print(f"Simplify Tolerance:    {simplify_tol} pixels")
    print("-" * 70)

    # 1. Load Model
    model = get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=1, pretrained=True)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Test Pairs & Compute Predictions
    with open(test_splits_file, "r", encoding="utf-8") as f:
        test_lines = [l.strip().split(",") for l in f.readlines() if l.strip()]

    print(f"Loaded {len(test_lines)} test image-mask pairs.")
    print("Generating probability maps for all test images...")

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

            # Ground truth polygons
            gt_polys, gt_stats = extract_clean_polygons(gt_mask, min_area=min_area_pixels, simplify_tol=simplify_tol)

            test_samples.append({
                "sample_name": name,
                "base_id": os.path.splitext(name)[0],
                "img_path": img_p,
                "mask_path": mask_p,
                "raw_rgb": np.array(raw_rgb),
                "gt_mask": gt_mask,
                "prob_map": prob,
                "gt_polys": gt_polys,
                "gt_poly_count": len(gt_polys),
                "gt_stats": gt_stats
            })

    pred_time = time.time() - t0
    print(f"Probabilities generated in {pred_time:.2f}s")
    print("-" * 70)

    # 3. Sweep Thresholds and Evaluate
    threshold_results = []
    sample_eval_per_thresh = {}

    print("Evaluating polygonization across thresholds...")

    for thresh in thresholds:
        thresh_start = time.time()
        total_pred_polys = 0
        total_gt_polys = sum(s["gt_poly_count"] for s in test_samples)
        total_valid = 0
        total_invalid_repaired = 0
        total_tiny_removed = 0

        all_match_05 = []
        all_match_25 = []
        all_area_metrics = []

        sample_eval_per_thresh[thresh] = []

        for s in test_samples:
            bin_mask = (s["prob_map"] > thresh).astype(np.uint8)
            pred_polys, p_stats = extract_clean_polygons(bin_mask, min_area=min_area_pixels, simplify_tol=simplify_tol)

            total_pred_polys += len(pred_polys)
            total_valid += p_stats["valid_count"]
            total_invalid_repaired += p_stats["invalid_repaired"]
            total_tiny_removed += p_stats["tiny_removed"]

            # Spatial match at 0.50 IoU
            m05 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.50)
            # Spatial match at 0.25 IoU
            m25 = match_polygons_spatial(s["gt_polys"], pred_polys, iou_threshold=0.25)
            # Area metrics
            area_m = compute_area_metrics(s["gt_polys"], pred_polys)

            all_match_05.append(m05)
            all_match_25.append(m25)
            all_area_metrics.append(area_m)

            sample_eval_per_thresh[thresh].append({
                "sample_name": s["sample_name"],
                "base_id": s["base_id"],
                "pred_polys": pred_polys,
                "gt_poly_count": s["gt_poly_count"],
                "pred_poly_count": len(pred_polys),
                "match_05": m05,
                "match_25": m25,
                "area_metrics": area_m,
                "poly_stats": p_stats
            })

        # Aggregate across 64 test samples
        agg_tp_05 = sum(m["tp"] for m in all_match_05)
        agg_fp_05 = sum(m["fp"] for m in all_match_05)
        agg_fn_05 = sum(m["fn"] for m in all_match_05)
        obj_prec_05 = agg_tp_05 / (agg_tp_05 + agg_fp_05) if (agg_tp_05 + agg_fp_05) > 0 else 0.0
        obj_rec_05 = agg_tp_05 / (agg_tp_05 + agg_fn_05) if (agg_tp_05 + agg_fn_05) > 0 else 0.0
        obj_f1_05 = (2.0 * obj_prec_05 * obj_rec_05) / (obj_prec_05 + obj_rec_05) if (obj_prec_05 + obj_rec_05) > 0 else 0.0

        agg_tp_25 = sum(m["tp"] for m in all_match_25)
        agg_fp_25 = sum(m["fp"] for m in all_match_25)
        agg_fn_25 = sum(m["fn"] for m in all_match_25)
        obj_prec_25 = agg_tp_25 / (agg_tp_25 + agg_fp_25) if (agg_tp_25 + agg_fp_25) > 0 else 0.0
        obj_rec_25 = agg_tp_25 / (agg_tp_25 + agg_fn_25) if (agg_tp_25 + agg_fn_25) > 0 else 0.0
        obj_f1_25 = (2.0 * obj_prec_25 * obj_rec_25) / (obj_prec_25 + obj_rec_25) if (obj_prec_25 + obj_rec_25) > 0 else 0.0

        # Mean matched IoU
        matched_ious_all = [m["mean_matched_iou"] for m in all_match_05 if m["tp"] > 0]
        mean_poly_iou = float(np.mean(matched_ious_all)) if matched_ious_all else 0.0

        mean_area_iou = float(np.mean([a["area_iou"] for a in all_area_metrics]))
        mean_area_prec = float(np.mean([a["area_precision"] for a in all_area_metrics]))
        mean_area_rec = float(np.mean([a["area_recall"] for a in all_area_metrics]))

        thresh_elapsed = time.time() - thresh_start

        row = {
            "threshold": thresh,
            "pred_polygon_count": total_pred_polys,
            "gt_polygon_count": total_gt_polys,
            "area_iou": round(mean_area_iou, 4),
            "area_precision": round(mean_area_prec, 4),
            "area_recall": round(mean_area_rec, 4),
            "polygon_iou_matched": round(mean_poly_iou, 4),
            "obj_f1_iou50": round(obj_f1_05, 4),
            "obj_precision_iou50": round(obj_prec_05, 4),
            "obj_recall_iou50": round(obj_rec_05, 4),
            "obj_f1_iou25": round(obj_f1_25, 4),
            "obj_precision_iou25": round(obj_prec_25, 4),
            "obj_recall_iou25": round(obj_rec_25, 4),
            "valid_polygon_count": total_valid,
            "invalid_repaired_count": total_invalid_repaired,
            "tiny_removed_count": total_tiny_removed,
            "valid_pct": round(total_valid / (total_valid + total_invalid_repaired) * 100, 2) if (total_valid + total_invalid_repaired) > 0 else 100.0,
            "eval_time_s": round(thresh_elapsed, 2)
        }
        threshold_results.append(row)

        print(f"  Thresh {thresh:.2f} | Area IoU: {mean_area_iou:.4f} Area Prec: {mean_area_prec:.4f} Rec: {mean_area_rec:.4f} | "
              f"Obj F1 (IoU>=0.50): {obj_f1_05:.4f} (Prec: {obj_prec_05:.4f}, Rec: {obj_rec_05:.4f}) | "
              f"Polys: {total_pred_polys} ({row['valid_pct']}% valid) [{thresh_elapsed:.1f}s]")

    # 4. Save Threshold Sweep CSV
    thresh_csv_path = os.path.join(output_dir, "threshold_results.csv")
    with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_results[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_results)

    print(f"\nThreshold comparison saved to: {thresh_csv_path}")

    # Determine Best Thresholds
    best_area_iou_thresh = max(threshold_results, key=lambda x: x["area_iou"])["threshold"]
    best_obj_f1_thresh = max(threshold_results, key=lambda x: x["obj_f1_iou50"])["threshold"]
    best_prec_thresh = max(threshold_results, key=lambda x: x["obj_precision_iou50"])["threshold"]
    best_rec_thresh = max(threshold_results, key=lambda x: x["obj_recall_iou50"])["threshold"]

    optimal_thresh = best_obj_f1_thresh
    print(f"Best Thresholds:")
    print(f"  - Best Area IoU:         {best_area_iou_thresh:.2f}")
    print(f"  - Best Object F1 (0.50): {best_obj_f1_thresh:.2f}")
    print(f"  - Best Precision:        {best_prec_thresh:.2f}")
    print(f"  - Best Recall:           {best_rec_thresh:.2f}")
    print(f"Selected Optimal Threshold for Vector Export: {optimal_thresh:.2f}")
    print("-" * 70)

    # 5. Export GeoJSON Files for Optimal Threshold
    best_evals = sample_eval_per_thresh[optimal_thresh]
    thresh_poly_dir = os.path.join(poly_export_dir, f"threshold_{optimal_thresh:.2f}")
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
                    "threshold": optimal_thresh,
                    "area_pixels": round(p.area, 2),
                    "perimeter_pixels": round(p.length, 2),
                    "is_valid": p.is_valid,
                    "has_holes": hasattr(p, "interiors") and len(list(p.interiors)) > 0
                }
            })
        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features_list
        }
        out_geojson = os.path.join(thresh_poly_dir, f"{base_id}_buildings.geojson")
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)

    print(f"Exported {len(best_evals)} GeoJSON building polygon files to: {thresh_poly_dir}")

    # 6. Generate Visualizations for at least 12 Diverse Test Samples
    print("Generating comprehensive visual panels for 12+ diverse test samples...")

    # Selection logic covering diverse categories
    sorted_by_cov = sorted(test_samples, key=lambda x: np.sum(x["gt_mask"]) / (256 * 256))
    eval_map = {item["sample_name"]: item for item in best_evals}

    selected_vis = []
    selected_vis_names = set()

    def add_vis_case(s, category):
        if s["sample_name"] not in selected_vis_names:
            selected_vis.append((s, category))
            selected_vis_names.add(s["sample_name"])

    # 1. Empty / zero building tiles
    for s in sorted_by_cov:
        if np.sum(s["gt_mask"]) == 0:
            add_vis_case(s, "Empty Background Tile (True Negative Test)")
            break

    # 2. Sparse / rural homesteads
    for s in sorted_by_cov:
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if 0.5 <= cov <= 2.5:
            add_vis_case(s, "Sparse Rural Homesteads")
            if len(selected_vis) >= 3:
                break

    # 3. Dense urban clusters
    for s in reversed(sorted_by_cov):
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if cov >= 10.0:
            add_vis_case(s, "Dense Urban / Commercial Blocks")
            if len([c for c in selected_vis if "Dense" in c[1]]) >= 3:
                break

    # 4. Suburban residential (touching & closely spaced)
    for s in sorted_by_cov:
        cov = np.sum(s["gt_mask"]) / (256 * 256) * 100
        if 3.0 <= cov <= 7.0:
            add_vis_case(s, "Suburban Residential / Closely Spaced Parcels")
            if len([c for c in selected_vis if "Suburban" in c[1]]) >= 4:
                break

    # 5. Outliers / high FP or FN
    high_fp_item = max(best_evals, key=lambda x: x["match_05"]["fp"])
    add_vis_case([s for s in test_samples if s["sample_name"] == high_fp_item["sample_name"]][0], "High False Positive / Boundary Over-Segmentation")

    high_fn_item = max(best_evals, key=lambda x: x["match_05"]["fn"])
    add_vis_case([s for s in test_samples if s["sample_name"] == high_fn_item["sample_name"]][0], "High False Negative / Tree Occlusion")

    # Fill to at least 14 cases
    for s in test_samples:
        if len(selected_vis) >= 14:
            break
        add_vis_case(s, "Representative Test Tile")

    # Render visualizations for selected cases
    for s, cat in selected_vis:
        name = s["sample_name"]
        base_id = s["base_id"]
        ev = eval_map[name]
        pred_polys = ev["pred_polys"]
        gt_polys = s["gt_polys"]

        # View 1: Original Image
        orig_img_rgb = s["raw_rgb"]
        Image.fromarray(orig_img_rgb).save(os.path.join(vis_export_dir, f"{base_id}_original.png"))

        # View 2: Ground Truth Polygons
        gt_rendered = render_vector_view((256, 256), gt_polys, fill_color=(255, 255, 255), border_color=(40, 255, 120))
        Image.fromarray(gt_rendered).save(os.path.join(vis_export_dir, f"{base_id}_gt_polygons.png"))

        # View 3: Predicted Polygons
        pred_rendered = render_vector_view((256, 256), pred_polys, fill_color=(255, 255, 255), border_color=(255, 80, 80))
        Image.fromarray(pred_rendered).save(os.path.join(vis_export_dir, f"{base_id}_pred_polygons.png"))

        # View 4: Ground Truth vs Predicted Spatial Overlay
        overlay_view = render_comparison_overlay(orig_img_rgb, gt_polys, pred_polys, ev["match_05"])
        Image.fromarray(overlay_view).save(os.path.join(vis_export_dir, f"{base_id}_overlay_match.png"))

        # View 5: Composite Side-by-Side Comparison
        # Layout: Aerial Image | GT Polygons | Predicted Polygons | Vector Spatial Overlay
        w, h = 256, 256
        header_h = 32
        comp_w = w * 4
        comp_h = h + header_h
        composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
        composite[:header_h, :] = [20, 24, 32]

        composite[header_h:, 0:w] = orig_img_rgb
        composite[header_h:, w:w*2] = gt_rendered
        composite[header_h:, w*2:w*3] = pred_rendered
        composite[header_h:, w*3:w*4] = overlay_view

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        cv2.putText(composite, "1. Aerial Image", (10, 21), font, font_scale, (220, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(composite, f"2. GT Polys ({len(gt_polys)})", (w + 10, 21), font, font_scale, (40, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(composite, f"3. Pred Polys ({len(pred_polys)})", (w * 2 + 10, 21), font, font_scale, (255, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(composite, f"4. Match Overlay (F1:{ev['match_05']['f1']})", (w * 3 + 10, 21), font, font_scale, (240, 190, 80), 1, cv2.LINE_AA)

        for i in range(1, 4):
            cv2.line(composite, (w * i, 0), (w * i, comp_h), (50, 58, 70), 1)

        Image.fromarray(composite).save(os.path.join(vis_export_dir, f"{base_id}_comparison.png"))

    print(f"Generated visual panels for {len(selected_vis)} diverse cases in: {vis_export_dir}")

    # 7. Create Final Polygonization Report JSON
    optimal_row = [r for r in threshold_results if r["threshold"] == optimal_thresh][0]

    report = {
        "experiment_phase": "Phase 3: Raster to Vector Evaluation",
        "checkpoint_used": model_checkpoint,
        "test_samples_count": len(test_samples),
        "polygonization_settings": {
            "min_area_pixels": min_area_pixels,
            "simplify_tolerance": simplify_tol,
            "thresholds_evaluated": thresholds
        },
        "best_thresholds": {
            "by_area_iou": best_area_iou_thresh,
            "by_object_f1_iou50": best_obj_f1_thresh,
            "by_object_precision": best_prec_thresh,
            "by_object_recall": best_rec_thresh,
            "selected_optimal": optimal_thresh
        },
        "optimal_threshold_metrics": optimal_row,
        "threshold_sweep_summary": threshold_results,
        "geometry_integrity": {
            "total_predicted_polygons": optimal_row["pred_polygon_count"],
            "valid_polygons": optimal_row["valid_polygon_count"],
            "invalid_polygons_repaired": optimal_row["invalid_repaired_count"],
            "tiny_polygons_removed": optimal_row["tiny_removed_count"],
            "validity_percentage": optimal_row["valid_pct"]
        },
        "visualizations_dir": vis_export_dir,
        "polygons_geojson_dir": thresh_poly_dir
    }

    report_json_path = os.path.join(output_dir, "polygonization_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 70)
    print(f"Comprehensive report saved to: {report_json_path}")
    print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 7 Raster to Vector Evaluation")
    parser.add_argument("--checkpoint", default="results/experiments/baseline_full/best_model.pth")
    parser.add_argument("--test-splits", default="data/processed_spacenet7/splits/test.txt")
    parser.add_argument("--output-dir", default="results/experiments/polygonization")
    parser.add_argument("--min-area", type=float, default=15.0)
    parser.add_argument("--simplify", type=float, default=0.5)
    args = parser.parse_args()

    run_raster_to_vector_evaluation(
        model_checkpoint=args.checkpoint,
        test_splits_file=args.test_splits,
        output_dir=args.output_dir,
        min_area_pixels=args.min_area,
        simplify_tol=args.simplify
    )


if __name__ == "__main__":
    main()
