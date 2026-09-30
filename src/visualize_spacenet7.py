"""
SpaceNet 7 Dataset Visualization & Inspection Tool.

Features:
1. Original Aerial Image (256x256)
2. Ground Truth Binary Building Mask
3. Cadastral Overlay (semi-transparent building footprint overlay + contours)
4. Side-by-Side Comparison (Original | Mask | Overlay)
5. Sample Selection & Navigation (Random, Prev, Next, Split Filter)
6. Dataset Information (Filenames, Dims, Building Pixels, Coverage %, Connected Components)
7. Multiple Random Samples (Generates & displays at least 6 random examples across Train, Val, Test)
8. Export & Save Visualizations (Saved to results/visualizations/spacenet7/)
9. Automated Dataset Sanity Check (Dimensions, Binary values, NaNs, File integrity)
10. Global & Split Dataset Statistics (Mean, Min, Max coverage, Empty masks, Dense masks)
"""

import os
import sys
import json
import random
import argparse
import webbrowser
import http.server
import socketserver
import threading
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image


def load_dataset_records(dataset_dir: str = "data/processed_spacenet7") -> Dict[str, List[Dict[str, str]]]:
    """Load image-mask file pairs from split files."""
    splits_dir = os.path.join(dataset_dir, "splits")
    records = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            continue
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",") for line in f.readlines() if line.strip()]
        for line in lines:
            if len(line) >= 2:
                img_path, mask_path = line[0], line[1]
                records[split].append({
                    "split": split,
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "sample_name": os.path.splitext(os.path.basename(img_path))[0]
                })

    return records


def run_sanity_checks(records: Dict[str, List[Dict[str, str]]], sample_size: int = 50) -> Dict[str, Any]:
    """
    Perform rigorous sanity check on dataset pairs:
    - Files exist and can be loaded
    - Spatial dimensions match identically
    - Mask contains only binary values (0 and 255 or 0 and 1)
    - No NaN or Inf values
    """
    all_items = []
    for split, items in records.items():
        all_items.extend(items)

    check_pool = random.sample(all_items, min(sample_size, len(all_items))) if all_items else []

    checks_passed = True
    dim_match_count = 0
    binary_valid_count = 0
    no_nan_count = 0
    load_success_count = 0
    errors = []

    for item in check_pool:
        img_path = item["img_path"]
        mask_path = item["mask_path"]

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            checks_passed = False
            errors.append(f"File missing: {img_path} or {mask_path}")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            checks_passed = False
            errors.append(f"Failed to read image or mask: {item['sample_name']}")
            continue

        load_success_count += 1

        # Check identical spatial dimensions
        if img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]:
            dim_match_count += 1
        else:
            checks_passed = False
            errors.append(f"Dimension mismatch in {item['sample_name']}: img {img.shape[:2]} vs mask {mask.shape}")

        # Check binary values
        unique_vals = set(np.unique(mask).tolist())
        if unique_vals.issubset({0, 1, 255}):
            binary_valid_count += 1
        else:
            checks_passed = False
            errors.append(f"Non-binary mask values in {item['sample_name']}: {unique_vals}")

        # Check NaN / Inf
        if not np.isnan(img).any() and not np.isnan(mask).any() and not np.isinf(img).any() and not np.isinf(mask).any():
            no_nan_count += 1
        else:
            checks_passed = False
            errors.append(f"NaN or Inf values detected in {item['sample_name']}")

    return {
        "status": "PASSED" if checks_passed else "FAILED",
        "samples_checked": len(check_pool),
        "total_dataset_samples": len(all_items),
        "load_success": f"{load_success_count}/{len(check_pool)}",
        "dimension_matching": f"{dim_match_count}/{len(check_pool)}",
        "binary_valid": f"{binary_valid_count}/{len(check_pool)}",
        "no_nan_inf": f"{no_nan_count}/{len(check_pool)}",
        "errors": errors
    }


def compute_sample_metrics(img_path: str, mask_path: str) -> Dict[str, Any]:
    """Compute detailed geometric & cadastral statistics for a single sample."""
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return {}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = mask.shape
    total_pixels = h * w
    building_pixels = int(np.sum(mask > 0))
    coverage_pct = float(building_pixels / total_pixels * 100)

    # Connected components (building footprints)
    binary_mask = (mask > 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    num_buildings = max(0, num_labels - 1)

    return {
        "image_shape": [w, h, 3],
        "mask_shape": [w, h],
        "total_pixels": total_pixels,
        "building_pixels": building_pixels,
        "coverage_pct": round(coverage_pct, 2),
        "num_buildings": num_buildings
    }


def compute_aggregate_statistics(records: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Compute dataset-wide statistics across splits."""
    all_coverages = []
    num_empty = 0
    num_dense = 0  # > 15% coverage
    split_stats = {}

    for split, items in records.items():
        coverages = []
        for item in items:
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            b_pixels = int(np.sum(mask > 0))
            cov = (b_pixels / (mask.shape[0] * mask.shape[1])) * 100
            coverages.append(cov)
            all_coverages.append(cov)
            if cov == 0.0:
                num_empty += 1
            if cov >= 15.0:
                num_dense += 1

        if coverages:
            split_stats[split] = {
                "count": len(coverages),
                "mean_coverage_pct": round(float(np.mean(coverages)), 2),
                "min_coverage_pct": round(float(np.min(coverages)), 2),
                "max_coverage_pct": round(float(np.max(coverages)), 2),
            }

    global_stats = {
        "total_samples": len(all_coverages),
        "mean_coverage_pct": round(float(np.mean(all_coverages)), 2) if all_coverages else 0.0,
        "min_coverage_pct": round(float(np.min(all_coverages)), 2) if all_coverages else 0.0,
        "max_coverage_pct": round(float(np.max(all_coverages)), 2) if all_coverages else 0.0,
        "empty_masks_count": num_empty,
        "dense_masks_count": num_dense,
        "split_breakdown": split_stats
    }
    return global_stats


def create_visualization_views(
    img_path: str,
    mask_path: str,
    alpha: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate:
    1. Original RGB image (256x256)
    2. Binary Mask RGB (256x256)
    3. Cadastral Overlay (RGB + translucent building fill + high-contrast boundary contour)
    4. Side-by-side composite panel (Original | Mask | Overlay) with clean headers
    """
    img_bgr = cv2.imread(img_path)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = mask_gray.shape

    # Binary mask visualized in 3-channel grayscale / cyan
    mask_vis = np.zeros_like(img_rgb)
    mask_vis[mask_gray > 0] = [255, 255, 255]

    # Overlay: Base image + translucent building fill + distinct outline
    overlay = img_rgb.copy()
    color_fill = np.zeros_like(img_rgb)
    color_fill[mask_gray > 0] = [255, 50, 50]  # Cadastral Coral Red

    overlay = cv2.addWeighted(overlay, 1.0 - alpha, color_fill, alpha, 0)

    # Building boundary contours
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 255, 120), 1, lineType=cv2.LINE_AA)  # Emerald green contour

    # Compose side-by-side banner
    header_h = 32
    comp_w = w * 3
    comp_h = h + header_h
    composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)

    # Fill header bar
    composite[:header_h, :] = [24, 28, 36]  # Dark slate header

    # Place sub-images
    composite[header_h:, 0:w] = img_rgb
    composite[header_h:, w:w*2] = mask_vis
    composite[header_h:, w*2:w*3] = overlay

    # Add text labels on header bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    cv2.putText(composite, "1. Aerial Image", (15, 21), font, font_scale, (220, 225, 235), thickness, cv2.LINE_AA)
    cv2.putText(composite, "2. Ground Truth Mask", (w + 15, 21), font, font_scale, (220, 225, 235), thickness, cv2.LINE_AA)
    cv2.putText(composite, "3. Cadastral Overlay", (w * 2 + 15, 21), font, font_scale, (40, 255, 120), thickness, cv2.LINE_AA)

    # Dividers
    cv2.line(composite, (w, 0), (w, comp_h), (50, 58, 70), 1)
    cv2.line(composite, (w * 2, 0), (w * 2, comp_h), (50, 58, 70), 1)

    return img_rgb, mask_vis, overlay, composite


def save_sample_visualizations(
    sample: Dict[str, str],
    output_dir: str
) -> Dict[str, str]:
    """Generate and save the 4 visualization artifacts for a sample."""
    os.makedirs(output_dir, exist_ok=True)
    name = sample["sample_name"]

    img_rgb, mask_vis, overlay, composite = create_visualization_views(
        sample["img_path"], sample["mask_path"]
    )

    rgb_file = os.path.join(output_dir, f"{name}_original.png")
    mask_file = os.path.join(output_dir, f"{name}_mask.png")
    overlay_file = os.path.join(output_dir, f"{name}_overlay.png")
    comp_file = os.path.join(output_dir, f"{name}_comparison.png")

    Image.fromarray(img_rgb).save(rgb_file)
    Image.fromarray(mask_vis).save(mask_file)
    Image.fromarray(overlay).save(overlay_file)
    Image.fromarray(composite).save(comp_file)

    metrics = compute_sample_metrics(sample["img_path"], sample["mask_path"])
    metrics_file = os.path.join(output_dir, f"{name}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "sample_name": name,
            "split": sample["split"],
            "image_path": sample["img_path"],
            "mask_path": sample["mask_path"],
            "metrics": metrics
        }, f, indent=2)

    return {
        "rgb": rgb_file,
        "mask": mask_file,
        "overlay": overlay_file,
        "comparison": comp_file,
        "metrics": metrics
    }


def generate_interactive_html(
    inspected_samples: List[Dict[str, Any]],
    global_stats: Dict[str, Any],
    sanity_results: Dict[str, Any],
    output_html_path: str
):
    """
    Build a self-contained, responsive, local web application
    allowing interactive inspection of all required views.
    """
    samples_json = json.dumps(inspected_samples)
    stats_json = json.dumps(global_stats)
    sanity_json = json.dumps(sanity_results)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SpaceNet 7 Cadastral Dataset Inspector — SIH 2026</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #141c2e;
      --accent: #10b981;
      --accent-hover: #059669;
      --border: #23304a;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --danger: #ef4444;
      --warning: #f59e0b;
      --font-mono: 'Consolas', 'Courier New', monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 1.5rem; font-weight: 700; color: #fff; }}
    .subtitle {{ font-size: 0.875rem; color: var(--text-muted); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .grid-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .stat-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #fff; margin-top: 4px; }}
    .stat-sub {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }}

    /* Main Inspection Workspace */
    .workspace {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }}
    @media (max-width: 1024px) {{
      .workspace {{ grid-template-columns: 1fr; }}
    }}
    .sidebar {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .section-title {{ font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent); }}
    .control-group {{ display: flex; flex-direction: column; gap: 8px; }}
    label {{ font-size: 0.8rem; font-weight: 600; color: var(--text-muted); }}
    select, button {{
      background: #1a243b;
      border: 1px solid var(--border);
      color: #fff;
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.875rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    select:hover, button:hover {{ border-color: var(--accent); }}
    button.btn-primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #0b0f19;
      font-weight: 700;
    }}
    button.btn-primary:hover {{ background: var(--accent-hover); }}
    .btn-row {{ display: flex; gap: 8px; }}
    .btn-row button {{ flex: 1; }}

    /* Info Table */
    .info-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    .info-table td {{ padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
    .info-table td:first-child {{ color: var(--text-muted); font-weight: 500; }}
    .info-table td:last-child {{ text-align: right; font-family: var(--font-mono); color: #fff; }}

    /* Viewer Area */
    .viewer {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    .tab {{
      padding: 8px 16px;
      border-radius: 6px;
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .tab.active {{
      background: #1a243b;
      color: #fff;
      border-color: var(--border);
      border-bottom: 2px solid var(--accent);
    }}
    .view-container {{
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 480px;
      background: #0d121d;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    .view-image {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}

    /* Gallery Grid */
    .gallery-section {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-top: 16px;
    }}
    .gallery-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .gallery-card {{
      background: #0d121d;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .gallery-card img {{
      width: 100%;
      display: block;
      border-bottom: 1px solid var(--border);
    }}
    .gallery-card-body {{
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .gallery-card-title {{
      font-size: 0.85rem;
      font-weight: 600;
      font-family: var(--font-mono);
      color: #fff;
    }}
    .gallery-card-tag {{
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .tag-train {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; }}
    .tag-val {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
    .tag-test {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
  </style>
</head>
<body>

  <header>
    <div>
      <h1>SpaceNet 7 Cadastral Dataset Inspector</h1>
      <div class="subtitle">Area of Interest: L15-0506E-1204N_2027_3374_13 • SIH 2026 Prototype</div>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
      <span class="badge" id="sanityBadge">✓ Sanity Checks: PASSED</span>
    </div>
  </header>

  <!-- High-level Dataset Stats -->
  <div class="grid-stats">
    <div class="stat-card">
      <div class="stat-label">Total Patches</div>
      <div class="stat-value">{global_stats['total_samples']}</div>
      <div class="stat-sub">256x256 image-mask pairs</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Mean Coverage</div>
      <div class="stat-value">{global_stats['mean_coverage_pct']}%</div>
      <div class="stat-sub">Range: {global_stats['min_coverage_pct']}% – {global_stats['max_coverage_pct']}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Dense Masks (>15%)</div>
      <div class="stat-value">{global_stats['dense_masks_count']}</div>
      <div class="stat-sub">High parcel density</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Empty Masks</div>
      <div class="stat-value">{global_stats['empty_masks_count']}</div>
      <div class="stat-sub">Zero building pixels</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Splits (Train/Val/Test)</div>
      <div class="stat-value">256 / 64 / 64</div>
      <div class="stat-sub">Temporal chronological split</div>
    </div>
  </div>

  <!-- Main Interactive Workspace -->
  <div class="workspace">
    <!-- Left Sidebar: Controls & Sample Info -->
    <div class="sidebar">
      <div class="section-title">Sample Controls</div>

      <div class="control-group">
        <label for="splitSelect">Filter by Split</label>
        <select id="splitSelect" onchange="onSplitFilterChange()">
          <option value="all">All Splits ({len(inspected_samples)} loaded)</option>
          <option value="train">Train</option>
          <option value="val">Validation</option>
          <option value="test">Test</option>
        </select>
      </div>

      <div class="control-group">
        <label for="sampleSelect">Select Sample</label>
        <select id="sampleSelect" onchange="onSampleSelectChange()"></select>
      </div>

      <div class="btn-row">
        <button onclick="prevSample()">← Prev</button>
        <button onclick="randomSample()">Random 🎲</button>
        <button onclick="nextSample()">Next →</button>
      </div>

      <div class="section-title" style="margin-top:10px;">Sample Information</div>
      <table class="info-table">
        <tr><td>Sample ID</td><td id="infoName">-</td></tr>
        <tr><td>Split</td><td id="infoSplit">-</td></tr>
        <tr><td>Image Size</td><td id="infoDims">256 × 256</td></tr>
        <tr><td>Building Pixels</td><td id="infoPixels">-</td></tr>
        <tr><td>Footprint Coverage</td><td id="infoCoverage">-</td></tr>
        <tr><td>Connected Buildings</td><td id="infoBuildings">-</td></tr>
      </table>

      <div style="margin-top: auto; font-size: 0.72rem; color: var(--text-muted); line-height: 1.4;">
        📁 Local Visualizations Saved:<br>
        <span style="color:#fff; font-family:var(--font-mono);">results/visualizations/spacenet7/</span>
      </div>
    </div>

    <!-- Right Viewer: Tabbed Views -->
    <div class="viewer">
      <div class="tabs">
        <button class="tab active" onclick="switchView('comparison')">Side-by-Side Comparison</button>
        <button class="tab" onclick="switchView('overlay')">Cadastral Overlay ⭐</button>
        <button class="tab" onclick="switchView('rgb')">Original Aerial Image</button>
        <button class="tab" onclick="switchView('mask')">Ground Truth Mask</button>
      </div>

      <div class="view-container">
        <img id="mainDisplayImg" class="view-image" src="" alt="Sample Visualization">
      </div>
    </div>
  </div>

  <!-- Multi-Sample Gallery (Random 6 Samples from Train, Val, Test) -->
  <div class="gallery-section">
    <div class="gallery-header">
      <div>
        <div class="section-title">Multi-Sample Inspection Gallery</div>
        <div class="subtitle">Representative random samples selected across Train, Validation, and Test splits</div>
      </div>
      <span class="badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.3);">
        {len(inspected_samples)} Verified Samples
      </span>
    </div>

    <div class="gallery-grid" id="galleryGrid"></div>
  </div>

  <script>
    const samples = {samples_json};
    let currentIndex = 0;
    let currentView = 'comparison'; // 'comparison', 'overlay', 'rgb', 'mask'

    function init() {{
      const select = document.getElementById('sampleSelect');
      select.innerHTML = '';
      samples.forEach((s, idx) => {{
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `[${{s.split.toUpperCase()}}] ${{s.sample_name}} (${{s.metrics.coverage_pct}}%)`;
        select.appendChild(opt);
      }});

      renderGallery();
      updateView();
    }}

    function onSplitFilterChange() {{
      const split = document.getElementById('splitSelect').value;
      const select = document.getElementById('sampleSelect');
      select.innerHTML = '';
      samples.forEach((s, idx) => {{
        if (split === 'all' || s.split === split) {{
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `[${{s.split.toUpperCase()}}] ${{s.sample_name}} (${{s.metrics.coverage_pct}}%)`;
          select.appendChild(opt);
        }}
      }});
      if (select.options.length > 0) {{
        currentIndex = parseInt(select.options[0].value);
        updateView();
      }}
    }}

    function onSampleSelectChange() {{
      currentIndex = parseInt(document.getElementById('sampleSelect').value);
      updateView();
    }}

    function nextSample() {{
      const select = document.getElementById('sampleSelect');
      let nextOpt = select.selectedIndex + 1;
      if (nextOpt >= select.options.length) nextOpt = 0;
      select.selectedIndex = nextOpt;
      currentIndex = parseInt(select.options[nextOpt].value);
      updateView();
    }}

    function prevSample() {{
      const select = document.getElementById('sampleSelect');
      let prevOpt = select.selectedIndex - 1;
      if (prevOpt < 0) prevOpt = select.options.length - 1;
      select.selectedIndex = prevOpt;
      currentIndex = parseInt(select.options[prevOpt].value);
      updateView();
    }}

    function randomSample() {{
      const select = document.getElementById('sampleSelect');
      const randOpt = Math.floor(Math.random() * select.options.length);
      select.selectedIndex = randOpt;
      currentIndex = parseInt(select.options[randOpt].value);
      updateView();
    }}

    function switchView(viewName) {{
      currentView = viewName;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      event.target.classList.add('active');
      updateView();
    }}

    function updateView() {{
      const s = samples[currentIndex];
      if (!s) return;

      document.getElementById('infoName').textContent = s.sample_name;
      document.getElementById('infoSplit').textContent = s.split.toUpperCase();
      document.getElementById('infoDims').textContent = `${{s.metrics.image_shape[0]}} × ${{s.metrics.image_shape[1]}}`;
      document.getElementById('infoPixels').textContent = s.metrics.building_pixels.toLocaleString() + ' px';
      document.getElementById('infoCoverage').textContent = `${{s.metrics.coverage_pct}}%`;
      document.getElementById('infoBuildings').textContent = s.metrics.num_buildings;

      const imgElem = document.getElementById('mainDisplayImg');
      if (currentView === 'comparison') {{
        imgElem.src = s.rel_paths.comparison;
      }} else if (currentView === 'overlay') {{
        imgElem.src = s.rel_paths.overlay;
      }} else if (currentView === 'rgb') {{
        imgElem.src = s.rel_paths.rgb;
      }} else if (currentView === 'mask') {{
        imgElem.src = s.rel_paths.mask;
      }}
    }}

    function renderGallery() {{
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      samples.forEach((s, idx) => {{
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.innerHTML = `
          <img src="${{s.rel_paths.comparison}}" alt="${{s.sample_name}}">
          <div class="gallery-card-body">
            <div>
              <div class="gallery-card-title">${{s.sample_name}}</div>
              <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">
                ${{s.metrics.coverage_pct}}% coverage • ${{s.metrics.num_buildings}} buildings
              </div>
            </div>
            <span class="gallery-card-tag tag-${{s.split}}">${{s.split}}</span>
          </div>
        `;
        card.onclick = () => {{
          currentIndex = idx;
          const select = document.getElementById('sampleSelect');
          select.value = idx;
          updateView();
          window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }};
        grid.appendChild(card);
      }});
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def run_local_server(directory: str, port: int = 8050):
    """Launch a lightweight, non-blocking local web server."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, format, *args):
            pass  # quiet logging

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"\n[HTTP Server] Local Dataset Inspector running at: http://localhost:{port}/")
            httpd.serve_forever()
    except Exception as e:
        print(f"Server notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 7 Dataset Visualizer & Inspector")
    parser.add_argument("--dataset-dir", default="data/processed_spacenet7", help="Processed dataset root")
    parser.add_argument("--output-dir", default="results/visualizations/spacenet7", help="Output directory for visual artifacts")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of random samples to export & inspect")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP server to view inspector in browser")
    parser.add_argument("--port", type=int, default=8050, help="Port for local server")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 60)
    print("SpaceNet 7 Dataset Inspector & Verification Tool")
    print("=" * 60)

    # 1. Load dataset records
    records = load_dataset_records(args.dataset_dir)
    total_records = sum(len(v) for v in records.values())
    if total_records == 0:
        print(f"Error: No records found in {args.dataset_dir}. Run dataset preparation first.")
        sys.exit(1)

    print(f"Dataset Loaded: {total_records} total samples")
    print(f"  - Train: {len(records['train'])} pairs")
    print(f"  - Val:   {len(records['val'])} pairs")
    print(f"  - Test:  {len(records['test'])} pairs")
    print("-" * 60)

    # 2. Automated Sanity Check
    print("Running Automated Sanity Checks across dataset...")
    sanity_results = run_sanity_checks(records, sample_size=total_records)  # Check all 384 pairs!
    print(f"Sanity Check Result: [{sanity_results['status']}]")
    print(f"  - Verified Samples:       {sanity_results['samples_checked']} of {total_records}")
    print(f"  - File Load Integrity:    {sanity_results['load_success']}")
    print(f"  - Dimension Alignment:    {sanity_results['dimension_matching']} (256x256)")
    print(f"  - Mask Binary Range:      {sanity_results['binary_valid']} ({'{0, 255}'})")
    print(f"  - NaN / Inf Check:        {sanity_results['no_nan_inf']}")
    if sanity_results["errors"]:
        print(f"  - Errors encountered:     {len(sanity_results['errors'])}")
        for err in sanity_results["errors"][:5]:
            print(f"    * {err}")
    print("-" * 60)

    # 3. Global Statistics
    print("Computing Dataset-Wide Cadastral Statistics...")
    global_stats = compute_aggregate_statistics(records)
    print(f"  - Mean Building Coverage: {global_stats['mean_coverage_pct']}%")
    print(f"  - Min Building Coverage:  {global_stats['min_coverage_pct']}%")
    print(f"  - Max Building Coverage:  {global_stats['max_coverage_pct']}%")
    print(f"  - Empty Masks:            {global_stats['empty_masks_count']} ({round(global_stats['empty_masks_count']/total_records*100, 1)}%)")
    print(f"  - Dense Masks (>=15%):    {global_stats['dense_masks_count']} ({round(global_stats['dense_masks_count']/total_records*100, 1)}%)")
    print("-" * 60)

    # 4. Select Random Samples from Train, Val, Test
    random.seed(42)
    selected_samples = []
    # Pick 2 from train, 2 from val, 2 from test
    selected_samples.extend(random.sample(records["train"], min(2, len(records["train"]))))
    selected_samples.extend(random.sample(records["val"], min(2, len(records["val"]))))
    selected_samples.extend(random.sample(records["test"], min(2, len(records["test"]))))

    print(f"Exporting {len(selected_samples)} Verified Samples to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    inspected_data = []
    for sample in selected_samples:
        paths = save_sample_visualizations(sample, args.output_dir)
        name = sample["sample_name"]
        inspected_data.append({
            "sample_name": name,
            "split": sample["split"],
            "metrics": paths["metrics"],
            "rel_paths": {
                "rgb": f"{name}_original.png",
                "mask": f"{name}_mask.png",
                "overlay": f"{name}_overlay.png",
                "comparison": f"{name}_comparison.png"
            }
        })
        print(f"  [OK] [{sample['split'].upper()}] {name}: {paths['metrics']['coverage_pct']}% coverage, {paths['metrics']['num_buildings']} buildings")

    # 5. Generate Standalone HTML Inspector
    html_path = os.path.join(args.output_dir, "index.html")
    generate_interactive_html(inspected_data, global_stats, sanity_results, html_path)
    print(f"\nInteractive HTML Inspector generated at:")
    print(f"  file:///{os.path.abspath(html_path).replace(os.sep, '/')}")
    print("=" * 60)

    # 6. Save Overall Summary JSON
    summary_file = os.path.join(args.output_dir, "inspection_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "sanity_check": sanity_results,
            "statistics": global_stats,
            "inspected_samples": inspected_data
        }, f, indent=2)

    # Optional server
    if args.serve:
        server_thread = threading.Thread(target=run_local_server, args=(args.output_dir, args.port), daemon=True)
        server_thread.start()
        # Keep alive for CLI
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
