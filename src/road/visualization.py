"""
src/road/visualization.py

Presentation-grade visual validation figures for SIH26012 Road and Access Corridor Extraction.
Generates 4 high-contrast publication figures:
  01_input_image.png: Raw aerial imagery (Open Cities 0.2m GSD)
  02_road_mask.png: Extracted road corridor binary mask and probability field
  03_road_vector_overlay.png: High-contrast GIS road corridor vector boundaries
  04_buildings_roads_overlay.png: Integrated AI Building Footprints + Road Corridors
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
import shapely


def plot_road_visualizations(
    image: np.ndarray,
    road_mask: np.ndarray,
    prob_map: np.ndarray,
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    bounds: Optional[Tuple[float, float, float, float]] = None,
):
    """
    Renders 4 presentation-ready figures for Phase 13B evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]

    # Coordinate extent for spatial overlay if bounds provided
    # bounds: (minx, miny, maxx, maxy) in EPSG:32732
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]] if bounds else [0, w, h, 0]

    # ─────────────────────────────────────────────────────────────
    # Figure 1: 01_input_image.png
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    if bounds:
        ax.imshow(image, extent=extent)
        ax.set_xlabel("UTM Easting (m, EPSG:32732)", fontsize=11, fontweight="bold")
        ax.set_ylabel("UTM Northing (m, EPSG:32732)", fontsize=11, fontweight="bold")
    else:
        ax.imshow(image)
        ax.set_xlabel("Pixel Column", fontsize=11)
        ax.set_ylabel("Pixel Row", fontsize=11)

    ax.set_title(
        "SIH26012 Phase 13B — High-Resolution Aerial Input\n"
        f"Open Cities Aerial Imagery ({w}×{h} px | GSD ~0.20m | 204.8m × 204.8m)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    fig1_path = output_dir / "01_input_image.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # Figure 2: 02_road_mask.png (Dual Panel: Probability + Clean Mask)
    # ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)

    # Left: Continuous Probability
    im1 = ax1.imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    ax1.set_title("A. Continuous Road Probability Field P(road | x)", fontsize=12, fontweight="bold")
    ax1.axis("off")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Estimated Likelihood", fontsize=10)

    # Right: Clean Filtered Binary Mask
    ax2.imshow(image)
    road_mask_rgba = np.zeros((h, w, 4), dtype=np.float32)
    road_mask_rgba[road_mask > 0] = [0.0, 0.85, 1.0, 0.65]  # Bright Cyan
    ax2.imshow(road_mask_rgba)
    ax2.set_title("B. Morphologically Cleaned Access Corridor Mask", fontsize=12, fontweight="bold")
    ax2.axis("off")

    legend_elements = [
        Patch(facecolor="#00D9FF", edgecolor="white", alpha=0.8, label="Extracted Road/Access Corridor"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=10)

    plt.suptitle(
        "SIH26012 Phase 13B — Spectral & Texture-Driven Road Feature Extraction",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()
    fig2_path = output_dir / "02_road_mask.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # Figure 3: 03_road_vector_overlay.png
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 11), dpi=300)
    if bounds:
        ax.imshow(image, extent=extent)
    else:
        ax.imshow(image)

    # Overlay road vector polygons
    if len(roads_gdf) > 0:
        roads_gdf.plot(
            ax=ax,
            facecolor="#00e5ff",
            edgecolor="#0066cc",
            linewidth=1.8,
            alpha=0.45,
        )

    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel("UTM Easting (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("UTM Northing (m)", fontsize=11, fontweight="bold")

    total_rd_area = roads_gdf['area_m2'].sum() if len(roads_gdf) > 0 else 0.0
    legend_elements = [
        Patch(
            facecolor="#00e5ff",
            edgecolor="#0066cc",
            alpha=0.6,
            label=f"Physical Road Corridors (N={len(roads_gdf)} | {total_rd_area:,.1f} m²)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=10)
    ax.set_title(
        "SIH26012 Phase 13B — Vectorized Physical Road & Access Corridors\n"
        "(GIS Vector Polygons with Area, Width, and Continuity Attributes)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    fig3_path = output_dir / "03_road_vector_overlay.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # Figure 4: 04_buildings_roads_overlay.png (Master Combined Presentation)
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    if bounds:
        ax.imshow(image, extent=extent)
    else:
        ax.imshow(image)

    # 1. Plot Road corridors first (bottom vector layer)
    if len(roads_gdf) > 0:
        roads_gdf.plot(
            ax=ax,
            facecolor="#00d2ff",
            edgecolor="#0055ff",
            linewidth=2.0,
            alpha=0.45,
            zorder=2,
        )

    # 2. Plot AI Building Footprints (top vector layer)
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        buildings_gdf.plot(
            ax=ax,
            facecolor="#ff9100",
            edgecolor="#d50000",
            linewidth=1.2,
            alpha=0.75,
            zorder=3,
        )

    if bounds:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel("UTM Easting (m, EPSG:32732)", fontsize=11, fontweight="bold")
        ax.set_ylabel("UTM Northing (m, EPSG:32732)", fontsize=11, fontweight="bold")

    b_count = len(buildings_gdf) if buildings_gdf is not None else 0
    b_area = buildings_gdf.geometry.area.sum() if buildings_gdf is not None else 0.0

    legend_elements = [
        Patch(
            facecolor="#ff9100",
            edgecolor="#d50000",
            alpha=0.8,
            label=f"AI Building Footprints (N={b_count} | {b_area:,.1f} m²)",
        ),
        Patch(
            facecolor="#00d2ff",
            edgecolor="#0055ff",
            alpha=0.6,
            label=f"Physical Road / Access Corridors (N={len(roads_gdf)} | {total_rd_area:,.1f} m²)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, fontsize=11)
    ax.set_title(
        "SIH26012 Phase 13B — Multi-Layer Physical Feature Extraction\n"
        "AI Building Footprints + Vector Access Corridors (EPSG:32732)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    fig4_path = output_dir / "04_buildings_roads_overlay.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    return [fig1_path, fig2_path, fig3_path, fig4_path]
