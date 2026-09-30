"""
src/parcel_intelligence/visualization.py

Phase 14A — Parcel Intelligence Audit Visualization for SIH26012

Produces publication-quality 300 DPI figures:
  1. Confidence Distribution Histogram
  2. Band Distribution Bar Chart
  3. Confidence Component Radar/Spider Chart
  4. Flag Distribution Chart
  5. Spatial Confidence Heatmap
  6. Top-N / Bottom-N Case Comparison

All visualizations use Matplotlib with consistent styling.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import geopandas as gpd


# ─── Color Scheme ───
BAND_COLORS = {
    "HIGH": "#2ecc71",
    "MODERATE": "#f39c12",
    "LOW": "#e67e22",
    "VERY_LOW": "#e74c3c",
}

CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "parcel_confidence",
    ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71"],
    N=256,
)


def plot_confidence_distribution(
    scores: List[float],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Histogram of confidence score distribution with band annotations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    arr = np.array(scores)
    bins = np.arange(0, 1.05, 0.05)

    # Color bins by band
    n_vals, bin_edges, patches = ax.hist(arr, bins=bins, edgecolor="white", linewidth=0.5, alpha=0.9)

    for patch, left_edge in zip(patches, bin_edges[:-1]):
        mid = left_edge + 0.025
        if mid >= 0.75:
            patch.set_facecolor(BAND_COLORS["HIGH"])
        elif mid >= 0.50:
            patch.set_facecolor(BAND_COLORS["MODERATE"])
        elif mid >= 0.30:
            patch.set_facecolor(BAND_COLORS["LOW"])
        else:
            patch.set_facecolor(BAND_COLORS["VERY_LOW"])

    # Band threshold lines
    for threshold, label, color in [(0.75, "HIGH", "#2ecc71"), (0.50, "MODERATE", "#f39c12"), (0.30, "LOW", "#e67e22")]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.92, label, color=color, fontsize=8, fontweight="bold")

    # Statistics
    mean_val = float(arr.mean())
    median_val = float(np.median(arr))
    ax.axvline(mean_val, color="#3498db", linestyle="-", linewidth=2, alpha=0.8)
    ax.axvline(median_val, color="#9b59b6", linestyle="-", linewidth=2, alpha=0.8)

    ax.text(mean_val + 0.01, ax.get_ylim()[1] * 0.80, f"Mean={mean_val:.3f}", color="#3498db", fontsize=8)
    ax.text(median_val + 0.01, ax.get_ylim()[1] * 0.70, f"Median={median_val:.3f}", color="#9b59b6", fontsize=8)

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=12)
    ax.set_title("Phase 14A — Parcel Candidate Confidence Distribution", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=BAND_COLORS["HIGH"], label="HIGH (≥0.75)"),
        Patch(facecolor=BAND_COLORS["MODERATE"], label="MODERATE (≥0.50)"),
        Patch(facecolor=BAND_COLORS["LOW"], label="LOW (≥0.30)"),
        Patch(facecolor=BAND_COLORS["VERY_LOW"], label="VERY LOW (<0.30)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_band_distribution(
    band_counts: Dict[str, int],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Bar chart of confidence band distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bands = ["HIGH", "MODERATE", "LOW", "VERY_LOW"]
    counts = [band_counts.get(b, 0) for b in bands]
    colors = [BAND_COLORS[b] for b in bands]
    total = sum(counts)

    bars = ax.bar(bands, counts, color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    # Annotate bars
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / max(1, total)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Confidence Band", fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=12)
    ax.set_title("Phase 14A — Confidence Band Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_component_radar(
    mean_components: Dict[str, float],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Radar/spider chart showing mean sub-scores across confidence dimensions.
    """
    labels = list(mean_components.keys())
    values = list(mean_components.values())

    # Close the polygon
    values_closed = values + [values[0]]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles_closed, values_closed, 'o-', color="#3498db", linewidth=2, markersize=8)
    ax.fill(angles_closed, values_closed, color="#3498db", alpha=0.25)

    # Add threshold circles
    for threshold, color in [(0.75, "#2ecc71"), (0.50, "#f39c12"), (0.30, "#e67e22")]:
        circle_vals = [threshold] * (n + 1)
        ax.plot(angles_closed, circle_vals, '--', color=color, linewidth=1, alpha=0.5)

    ax.set_xticks(angles)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Phase 14A — Mean Confidence Component Scores", fontsize=14, fontweight="bold", pad=20)

    # Annotate values
    for angle, value, label in zip(angles, values, labels):
        ax.annotate(f"{value:.3f}", xy=(angle, value), xytext=(5, 5),
                    textcoords="offset points", fontsize=9, fontweight="bold", color="#2c3e50")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_flag_distribution(
    flag_dist: Dict[str, int],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Horizontal bar chart of flag type distribution.
    """
    if not flag_dist:
        # Empty chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No uncertainty flags raised", ha="center", va="center",
                fontsize=14, color="#7f8c8d", transform=ax.transAxes)
        ax.set_title("Phase 14A — Uncertainty Flag Distribution", fontsize=14, fontweight="bold")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(12, max(4, len(flag_dist) * 0.6)))

    sorted_flags = sorted(flag_dist.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_flags]
    counts = [f[1] for f in sorted_flags]

    # Color by severity
    severity_colors = {
        "NEEDS_TOPOLOGY_REVIEW": "#e74c3c",
        "NEEDS_BOUNDARY_REVIEW": "#e67e22",
        "WEAK_BUILDING_EVIDENCE": "#e67e22",
        "WEAK_ROAD_EVIDENCE": "#e67e22",
        "LOW_CONFIDENCE": "#e67e22",
        "HIGH_ELONGATION": "#f39c12",
        "OVERSIZED_PARCEL": "#f39c12",
        "UNDERSIZED_PARCEL": "#f39c12",
        "MULTI_SOURCE_CONFLICT": "#f39c12",
    }
    colors = [severity_colors.get(n, "#95a5a6") for n in names]

    bars = ax.barh(names, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Candidates Flagged", fontsize=12)
    ax.set_title("Phase 14A — Uncertainty Flag Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spatial_confidence_map(
    parcels_gdf: gpd.GeoDataFrame,
    confidence_scores: List[float],
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    road_gdf: Optional[gpd.GeoDataFrame] = None,
    output_path: str = "spatial_confidence.png",
    dpi: int = 300,
) -> str:
    """
    Spatial heatmap showing parcel candidates colored by confidence score.
    """
    fig, ax = plt.subplots(figsize=(14, 14))

    gdf = parcels_gdf.copy()
    gdf["confidence"] = confidence_scores

    # Plot parcels colored by confidence
    gdf.plot(
        ax=ax,
        column="confidence",
        cmap=CONFIDENCE_CMAP,
        vmin=0, vmax=1,
        edgecolor="#2c3e50",
        linewidth=0.5,
        alpha=0.75,
        legend=True,
        legend_kwds={
            "label": "Confidence Score",
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.05,
        },
    )

    # Overlay buildings
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        buildings_gdf.plot(ax=ax, facecolor="none", edgecolor="#2c3e50", linewidth=0.8, alpha=0.7)

    # Overlay roads
    if road_gdf is not None and len(road_gdf) > 0:
        road_gdf.plot(ax=ax, facecolor="#bdc3c7", edgecolor="#7f8c8d", linewidth=0.5, alpha=0.4)

    ax.set_title("Phase 14A — Spatial Confidence Map\n(AI-Generated Preliminary Parcel Candidates)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_bottom_comparison(
    top_bottom: Dict[str, Any],
    output_path: str,
    dpi: int = 300,
) -> str:
    """
    Side-by-side comparison of top-N and bottom-N candidates.
    """
    top_n = top_bottom.get("top_n", [])
    bottom_n = top_bottom.get("bottom_n", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(top_n) * 0.5)))

    # Top candidates
    if top_n:
        pids = [f"P{t['parcel_id']}" for t in top_n]
        confs = [t["confidence"] for t in top_n]
        colors = [BAND_COLORS.get(t["confidence_band"], "#95a5a6") for t in top_n]

        ax1.barh(pids, confs, color=colors, edgecolor="white", linewidth=0.8)
        for i, (pid, conf) in enumerate(zip(pids, confs)):
            ax1.text(conf + 0.01, i, f"{conf:.4f}", va="center", fontsize=9, fontweight="bold")
        ax1.set_xlim(0, 1)
        ax1.invert_yaxis()
        ax1.set_xlabel("Confidence", fontsize=11)
        ax1.set_title("Top Candidates (Highest Confidence)", fontsize=12, fontweight="bold", color="#27ae60")
        ax1.grid(axis="x", alpha=0.3)

    # Bottom candidates
    if bottom_n:
        pids = [f"P{b['parcel_id']}" for b in bottom_n]
        confs = [b["confidence"] for b in bottom_n]
        flag_counts = [b["flag_count"] for b in bottom_n]
        colors = [BAND_COLORS.get(b["confidence_band"], "#95a5a6") for b in bottom_n]

        ax2.barh(pids, confs, color=colors, edgecolor="white", linewidth=0.8)
        for i, (pid, conf, fc) in enumerate(zip(pids, confs, flag_counts)):
            ax2.text(conf + 0.01, i, f"{conf:.4f} ({fc} flags)", va="center", fontsize=9, fontweight="bold")
        ax2.set_xlim(0, 1)
        ax2.invert_yaxis()
        ax2.set_xlabel("Confidence", fontsize=11)
        ax2.set_title("Bottom Candidates (Lowest Confidence)", fontsize=12, fontweight="bold", color="#c0392b")
        ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Phase 14A — Top vs Bottom Parcel Candidates", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
