"""
src/road_dl/visualization.py

Presentation-grade visual validation figures for Phase 13C Deep-Learning Road Extraction.
Produces 4 figures at 300 DPI:
  01_classical_vs_dl.png: Side-by-side comparison of classical CV vs DL road mask
  02_classical_mask.png: Classical road corridor mask and probability field
  03_dl_mask.png: Deep-learning road probability field and vector corridors
  04_overlay_comparison.png: Master GIS multi-layer overlay (Buildings + Classical Roads + DL Roads)
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd


def plot_road_dl_comparisons(
    image_rgb: np.ndarray,
    classical_mask: np.ndarray,
    dl_mask: np.ndarray,
    dl_prob: np.ndarray,
    classical_roads_gdf: gpd.GeoDataFrame,
    dl_roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> List[Path]:
    """
    Renders 4 publication figures for Phase 13C evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]] if bounds else [0, w, h, 0]

    saved_paths = []

    # ─────────────────────────────────────────────────────────────
    # Figure 1: 01_classical_vs_dl.png (Side-by-Side Masks)
    # ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)

    # Left: Classical CV Mask
    ax1.imshow(image_rgb, extent=extent)
    m1_rgba = np.zeros((h, w, 4), dtype=np.float32)
    m1_rgba[classical_mask > 0] = [0.0, 0.85, 1.0, 0.65]  # Cyan
    ax1.imshow(m1_rgba, extent=extent)
    ax1.set_xlim(bounds[0], bounds[2]) if bounds else None
    ax1.set_ylim(bounds[1], bounds[3]) if bounds else None
    ax1.set_title(
        f"A. Classical CV Baseline Road Extraction\n"
        f"Area: {classical_roads_gdf.geometry.area.sum():,.1f} m² | Components: {len(classical_roads_gdf)}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.axis("off")
    ax1.legend(
        handles=[Patch(facecolor="#00D9FF", edgecolor="white", label="Classical CV Corridor Mask")],
        loc="lower right",
        framealpha=0.9,
    )

    # Right: Deep Learning Mask
    ax2.imshow(image_rgb, extent=extent)
    m2_rgba = np.zeros((h, w, 4), dtype=np.float32)
    m2_rgba[dl_mask > 0] = [0.46, 1.0, 0.01, 0.65]  # Lime green
    ax2.imshow(m2_rgba, extent=extent)
    ax2.set_xlim(bounds[0], bounds[2]) if bounds else None
    ax2.set_ylim(bounds[1], bounds[3]) if bounds else None
    ax2.set_title(
        f"B. Deep-Learning Road Extraction (U-Net ResNet18)\n"
        f"Area: {dl_roads_gdf.geometry.area.sum():,.1f} m² | Components: {len(dl_roads_gdf)}",
        fontsize=13,
        fontweight="bold",
    )
    ax2.axis("off")
    ax2.legend(
        handles=[Patch(facecolor="#76FF03", edgecolor="white", label="Deep-Learning Road Mask")],
        loc="lower right",
        framealpha=0.9,
    )

    plt.suptitle("SIH26012 — Physical Road Corridor Extraction: Classical CV vs Deep Learning", fontsize=16, fontweight="bold", y=0.96)
    plt.tight_layout()
    p1 = output_dir / "01_classical_vs_dl.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_paths.append(p1)

    # ─────────────────────────────────────────────────────────────
    # Figure 2: 02_classical_mask.png (Classical Master)
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 11), dpi=300)
    ax.imshow(image_rgb, extent=extent)
    if len(classical_roads_gdf) > 0:
        classical_roads_gdf.plot(ax=ax, facecolor="#00e5ff", edgecolor="#0066cc", linewidth=1.8, alpha=0.5, zorder=2)
    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    ax.set_title("SIH26012 Phase 13B — Classical CV Physical Road Corridors", fontsize=13, fontweight="bold", pad=12)
    ax.legend(
        handles=[Patch(facecolor="#00e5ff", edgecolor="#0066cc", label=f"Classical Roads (N={len(classical_roads_gdf)})")],
        loc="lower right",
        framealpha=0.9,
    )
    plt.tight_layout()
    p2 = output_dir / "02_classical_mask.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_paths.append(p2)

    # ─────────────────────────────────────────────────────────────
    # Figure 3: 03_dl_mask.png (Deep Learning Heatmap + Vectors)
    # ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)

    # Probability Heatmap
    im = ax1.imshow(dl_prob, cmap="inferno", vmin=0.0, vmax=1.0)
    ax1.set_title("A. Deep-Learning Continuous Probability P(road | x)", fontsize=13, fontweight="bold")
    ax1.axis("off")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Neural Network Confidence", fontsize=10)

    # DL Vector Corridors
    ax2.imshow(image_rgb, extent=extent)
    if len(dl_roads_gdf) > 0:
        dl_roads_gdf.plot(ax=ax2, facecolor="#76ff03", edgecolor="#33691e", linewidth=1.8, alpha=0.5, zorder=2)
    if bounds:
        ax2.set_xlim(bounds[0], bounds[2])
        ax2.set_ylim(bounds[1], bounds[3])
    ax2.set_title("B. Vectorized Deep-Learning Road Corridors", fontsize=13, fontweight="bold")
    ax2.axis("off")
    ax2.legend(
        handles=[Patch(facecolor="#76ff03", edgecolor="#33691e", label=f"DL Road Corridors (N={len(dl_roads_gdf)})")],
        loc="lower right",
        framealpha=0.9,
    )

    plt.suptitle("SIH26012 Phase 13C — Deep-Learning Road Probability Field & Vector Geometry", fontsize=16, fontweight="bold", y=0.96)
    plt.tight_layout()
    p3 = output_dir / "03_dl_mask.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_paths.append(p3)

    # ─────────────────────────────────────────────────────────────
    # Figure 4: 04_overlay_comparison.png (Combined Master GIS)
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.imshow(image_rgb, extent=extent)

    # Classical roads (cyan outline/fill)
    if len(classical_roads_gdf) > 0:
        classical_roads_gdf.plot(ax=ax, facecolor="#00d2ff", edgecolor="#0055ff", linewidth=1.8, alpha=0.35, zorder=2)

    # DL roads (lime fill/outline)
    if len(dl_roads_gdf) > 0:
        dl_roads_gdf.plot(ax=ax, facecolor="none", edgecolor="#76ff03", linewidth=2.4, linestyle="--", zorder=3)

    # Buildings (red/amber)
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        buildings_gdf.plot(ax=ax, facecolor="#ff9100", edgecolor="#d50000", linewidth=1.2, alpha=0.85, zorder=4)

    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel("UTM Easting (m, EPSG:32732)", fontsize=11, fontweight="bold")
        ax.set_ylabel("UTM Northing (m, EPSG:32732)", fontsize=11, fontweight="bold")

    ax.set_title(
        "SIH26012 Phase 13C — Multi-Modal Road Feature Comparison\n"
        "AI Buildings + Classical CV Corridors (Blue) + Deep-Learning Roads (Green Dashed)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    leg = [
        Patch(facecolor="#ff9100", edgecolor="#d50000", label=f"AI Building Footprints ({len(buildings_gdf)})"),
        Patch(facecolor="#00d2ff", edgecolor="#0055ff", alpha=0.5, label=f"Classical CV Roads ({len(classical_roads_gdf)})"),
        Patch(facecolor="none", edgecolor="#76ff03", label=f"Deep-Learning Roads ({len(dl_roads_gdf)})"),
    ]
    ax.legend(handles=leg, loc="lower right", framealpha=0.92, fontsize=10)
    plt.tight_layout()
    p4 = output_dir / "04_overlay_comparison.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_paths.append(p4)

    return saved_paths
