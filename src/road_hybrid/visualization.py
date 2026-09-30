"""
src/road_hybrid/visualization.py

Publication-quality 300 DPI visualization generators for SIH26012 Phase 13D:
- Probability histograms (Step 3)
- Figures 01 through 08 (Step 13)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import box
import cv2


def plot_probability_histograms(
    p_cv: np.ndarray,
    p_dl: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Step 3: Diagnostic histograms comparing normalized classical CV likelihood
    and deep-learning sigmoid probabilities.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p_cv_flat = p_cv.flatten()
    p_dl_flat = p_dl.flatten()

    # 1. Classical Probability Histogram
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.hist(p_cv_flat, bins=50, range=(0.0, 1.0), color="#00D2FF", edgecolor="#0066FF", alpha=0.85, log=True)
    ax.axvline(0.40, color="#FF0055", linestyle="--", linewidth=1.5, label="Threshold (0.40)")
    ax.set_title("Normalized Classical CV Road Probability Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Probability Value $P_{classical}(x) \\in [0, 1]$", fontsize=9)
    ax.set_ylabel("Pixel Frequency (Log Scale)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    cv_hist_path = output_dir / "classical_probability_histogram.png"
    fig.savefig(cv_hist_path, dpi=300)
    plt.close(fig)

    # 2. Deep-Learning Probability Histogram
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.hist(p_dl_flat, bins=50, range=(0.0, 1.0), color="#E040FB", edgecolor="#7B1FA2", alpha=0.85, log=True)
    ax.axvline(0.40, color="#FF0055", linestyle="--", linewidth=1.5, label="Threshold (0.40)")
    ax.set_title("Normalized Deep-Learning Road Probability Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Sigmoid Probability $P_{DL}(x) \\in [0, 1]$", fontsize=9)
    ax.set_ylabel("Pixel Frequency (Log Scale)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    dl_hist_path = output_dir / "dl_probability_histogram.png"
    fig.savefig(dl_hist_path, dpi=300)
    plt.close(fig)

    return cv_hist_path, dl_hist_path


def generate_all_phase13d_figures(
    image_rgb: np.ndarray,
    extent: List[float],
    buildings_gdf: gpd.GeoDataFrame,
    cv_roads_gdf: gpd.GeoDataFrame,
    dl_roads_gdf: gpd.GeoDataFrame,
    hybrid_roads_gdf: gpd.GeoDataFrame,
    hybrid_nodes_gdf: gpd.GeoDataFrame,
    hybrid_edges_gdf: gpd.GeoDataFrame,
    cv_parcels_gdf: gpd.GeoDataFrame,
    dl_parcels_gdf: gpd.GeoDataFrame,
    hybrid_parcels_gdf: gpd.GeoDataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Step 13: Generates all 8 high-resolution 300 DPI publication figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {}

    # Helper function for base extent
    minx, maxx, miny, maxy = extent

    # -------------------------------------------------------------
    # 01_classical.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=ax, color="#FF9100", linewidth=0.8, alpha=0.8, label="Buildings")
    if len(cv_roads_gdf) > 0:
        cv_roads_gdf.plot(ax=ax, facecolor="#00D2FF", edgecolor="#0066FF", alpha=0.55, linewidth=1.0)
    ax.set_title("Phase 13B: Classical CV Road Corridors (Baseline)", fontsize=12, fontweight="bold")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    p1 = output_dir / "01_classical.png"
    fig.savefig(p1, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["01_classical"] = p1

    # -------------------------------------------------------------
    # 02_deep_learning.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=ax, color="#FF9100", linewidth=0.8, alpha=0.8)
    if len(dl_roads_gdf) > 0:
        dl_roads_gdf.plot(ax=ax, facecolor="#E040FB", edgecolor="#AA00FF", alpha=0.55, linewidth=1.0)
    ax.set_title("Phase 13C: Deep-Learning Road Corridors (ResNet18 U-Net)", fontsize=12, fontweight="bold")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    p2 = output_dir / "02_deep_learning.png"
    fig.savefig(p2, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["02_deep_learning"] = p2

    # -------------------------------------------------------------
    # 03_hybrid.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=ax, color="#FF9100", linewidth=0.8, alpha=0.8)
    if len(hybrid_roads_gdf) > 0:
        hybrid_roads_gdf.plot(ax=ax, facecolor="#00E676", edgecolor="#00B0FF", alpha=0.55, linewidth=1.2)
    ax.set_title("Phase 13D: Hybrid Road & Access Corridors (Fusion)", fontsize=12, fontweight="bold")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    p3 = output_dir / "03_hybrid.png"
    fig.savefig(p3, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["03_hybrid"] = p3

    # -------------------------------------------------------------
    # 04_three_way_comparison.png (2x2 Master Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=300)
    # (a) Aerial + Buildings
    axes[0, 0].imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.plot(ax=axes[0, 0], facecolor="#FF9100", edgecolor="#D50000", alpha=0.6, linewidth=0.8)
    axes[0, 0].set_title("(a) Orthomosaic + AI Buildings (N=254)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (b) Classical CV Roads
    axes[0, 1].imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=axes[0, 1], color="#FF9100", linewidth=0.6, alpha=0.7)
    if len(cv_roads_gdf) > 0:
        cv_roads_gdf.plot(ax=axes[0, 1], facecolor="#00D2FF", edgecolor="#0066FF", alpha=0.6, linewidth=1.0)
    axes[0, 1].set_title("(b) Classical CV Baseline (Contiguous Trunk)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # (c) Deep Learning Roads
    axes[1, 0].imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=axes[1, 0], color="#FF9100", linewidth=0.6, alpha=0.7)
    if len(dl_roads_gdf) > 0:
        dl_roads_gdf.plot(ax=axes[1, 0], facecolor="#E040FB", edgecolor="#AA00FF", alpha=0.6, linewidth=1.0)
    axes[1, 0].set_title("(c) Deep Learning (Narrow Paths + Fragmentation)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # (d) Hybrid Fusion
    axes[1, 1].imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=axes[1, 1], color="#FF9100", linewidth=0.6, alpha=0.7)
    if len(hybrid_roads_gdf) > 0:
        hybrid_roads_gdf.plot(ax=axes[1, 1], facecolor="#00E676", edgecolor="#00B0FF", alpha=0.6, linewidth=1.2)
    axes[1, 1].set_title("(d) Phase 13D Hybrid Fusion (Unified Network)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    fig.tight_layout()
    p4 = output_dir / "04_three_way_comparison.png"
    fig.savefig(p4, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["04_three_way_comparison"] = p4

    # -------------------------------------------------------------
    # 05_hybrid_graph.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.imshow(image_rgb, extent=extent, origin="upper")
    buildings_gdf.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.5, alpha=0.5)

    if len(hybrid_roads_gdf) > 0:
        hybrid_roads_gdf.plot(ax=ax, facecolor="#00E676", alpha=0.20)

    # Plot edges colored by source_evidence
    evidence_colors = {
        "both": "#00E676",         # Green: mutually agreed
        "classical": "#00D2FF",    # Cyan: classical only
        "deep_learning": "#E040FB", # Magenta: DL recovered
        "hybrid": "#FFD600",
    }
    for ev_type, color in evidence_colors.items():
        subset = hybrid_edges_gdf[hybrid_edges_gdf["source_evidence"] == ev_type]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, linewidth=2.4, label=f"Edge: {ev_type.replace('_', ' ').title()} ({len(subset)})")

    # Plot nodes
    if len(hybrid_nodes_gdf) > 0:
        endpoints = hybrid_nodes_gdf[hybrid_nodes_gdf["node_type"] == "endpoint"]
        intersections = hybrid_nodes_gdf[hybrid_nodes_gdf["node_type"] == "intersection"]
        if len(endpoints) > 0:
            endpoints.plot(ax=ax, color="#FF5252", markersize=14, label=f"Dead-End ({len(endpoints)})")
        if len(intersections) > 0:
            intersections.plot(ax=ax, color="#FFD600", markersize=26, marker="s", label=f"Intersection ({len(intersections)})")

    ax.set_title("Phase 13D: Hybrid Centerline Graph & Source Attribution", fontsize=12, fontweight="bold")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.axis("off")
    p5 = output_dir / "05_hybrid_graph.png"
    fig.savefig(p5, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["05_hybrid_graph"] = p5

    # -------------------------------------------------------------
    # 06_parcel_impact.png (3-way parcel comparison)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    # (a) Classical parcels
    axes[0].imshow(image_rgb, extent=extent, origin="upper")
    cv_parcels_gdf.boundary.plot(ax=axes[0], color="#00D2FF", linewidth=0.9)
    buildings_gdf.plot(ax=axes[0], facecolor="#FF9100", alpha=0.5)
    axes[0].set_title(f"Classical Constrained (N={len(cv_parcels_gdf)})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # (b) DL parcels
    axes[1].imshow(image_rgb, extent=extent, origin="upper")
    dl_parcels_gdf.boundary.plot(ax=axes[1], color="#E040FB", linewidth=0.9)
    buildings_gdf.plot(ax=axes[1], facecolor="#FF9100", alpha=0.5)
    axes[1].set_title(f"DL Constrained (N={len(dl_parcels_gdf)})", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # (c) Hybrid parcels
    axes[2].imshow(image_rgb, extent=extent, origin="upper")
    hybrid_parcels_gdf.boundary.plot(ax=axes[2], color="#00FFA3", linewidth=1.1)
    buildings_gdf.plot(ax=axes[2], facecolor="#FF9100", alpha=0.5)
    axes[2].set_title(f"Hybrid Constrained (N={len(hybrid_parcels_gdf)})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("Impact of Road Constraints on Preliminary Parcel Candidates (Same 254 Buildings)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    p6 = output_dir / "06_parcel_impact.png"
    fig.savefig(p6, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["06_parcel_impact"] = p6

    # -------------------------------------------------------------
    # 07_narrow_pathway_recovery.png (Zoomed Subregion)
    # -------------------------------------------------------------
    # Zoom on a dense corridor area: central 25% of ROI
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w_crop = (maxx - minx) * 0.40
    h_crop = (maxy - miny) * 0.40
    zoom_box = [cx - w_crop / 2, cx + w_crop / 2, cy - h_crop / 2, cy + h_crop / 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    for ax in axes:
        ax.imshow(image_rgb, extent=extent, origin="upper")
        buildings_gdf.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.8, alpha=0.7)
        buildings_gdf.plot(ax=ax, facecolor="#FF9100", alpha=0.45)
        ax.set_xlim(zoom_box[0], zoom_box[1])
        ax.set_ylim(zoom_box[2], zoom_box[3])
        ax.axis("off")

    if len(cv_roads_gdf) > 0:
        cv_roads_gdf.plot(ax=axes[0], facecolor="#00D2FF", edgecolor="#0066FF", alpha=0.6, linewidth=1.2)
    axes[0].set_title("Classical CV: Misses Narrow Alleyways", fontsize=10, fontweight="bold")

    if len(dl_roads_gdf) > 0:
        dl_roads_gdf.plot(ax=axes[1], facecolor="#E040FB", edgecolor="#AA00FF", alpha=0.6, linewidth=1.2)
    axes[1].set_title("Deep Learning: Detects Path but Fragmented", fontsize=10, fontweight="bold")

    if len(hybrid_roads_gdf) > 0:
        hybrid_roads_gdf.plot(ax=axes[2], facecolor="#00E676", edgecolor="#00B0FF", alpha=0.6, linewidth=1.2)
    # Overlay DL-recovered edges
    dl_edges = hybrid_edges_gdf[hybrid_edges_gdf["source_evidence"] == "deep_learning"]
    if len(dl_edges) > 0:
        dl_edges.plot(ax=axes[2], color="#FF007F", linewidth=2.5, label="Recovered Alleyway")
        axes[2].legend(loc="lower right", fontsize=8)
    axes[2].set_title("Hybrid Fusion: Continuous Trunk + Recovered Narrow Pathway", fontsize=10, fontweight="bold")

    fig.suptitle("Close-up Inspection: Narrow Pathway Recovery in Dense Urban Settlement", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    p7 = output_dir / "07_narrow_pathway_recovery.png"
    fig.savefig(p7, bbox_inches="tight", dpi=300)
    plt.close(fig)
    fig_paths["07_narrow_pathway_recovery"] = p7

    return fig_paths
