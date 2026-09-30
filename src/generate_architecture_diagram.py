"""
src/generate_architecture_diagram.py

Renders a publication-grade, high-resolution architecture diagram
for the SIH26012 system and exports to results/experiments/phase16_hardening/final_architecture.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_PATH = Path("results/experiments/phase16_hardening/final_architecture.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set high DPI for crisp presentation display
fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
fig.patch.set_facecolor('#0f172a')  # Dark slate background
ax.set_facecolor('#0f172a')

# Title & Banner
ax.text(8, 9.5, "SIH26012: END-TO-END SYSTEM ARCHITECTURE", 
        fontsize=20, fontweight='bold', color='#38bdf8', ha='center', va='center', fontfamily='sans-serif')
ax.text(8, 9.15, "AI-Assisted Aerial Building Segmentation, Hybrid Road Fusion & Human-in-the-Loop Parcel Verification", 
        fontsize=11, color='#94a3b8', ha='center', va='center', fontfamily='sans-serif')

def draw_box(ax, x, y, w, h, title, subtitle, color, bg_color='#1e293b', border_color=None):
    if border_color is None:
        border_color = color
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                edgecolor=border_color, facecolor=bg_color, linewidth=2, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, title, fontsize=10, fontweight='bold', color=color, 
            ha='center', va='center', fontfamily='sans-serif', zorder=3)
    ax.text(x + w/2, y + h*0.30, subtitle, fontsize=8, color='#cbd5e1', 
            ha='center', va='center', fontfamily='sans-serif', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='#64748b'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=7, headlength=7, shrink=0.05),
                zorder=1)

# Layer 1: Data Ingestion
ax.text(1.2, 8.4, "DATA & INGESTION", fontsize=9, fontweight='bold', color='#38bdf8', fontfamily='sans-serif')
draw_box(ax, 0.4, 7.2, 2.8, 0.9, "Aerial / Drone Orthomosaic", "Open Cities GeoTIFF (EPSG:32732)", '#38bdf8')
draw_box(ax, 0.4, 5.8, 2.8, 0.9, "Validation & Preprocessing", "Rasterio, NaN checks, Tiled Grid", '#38bdf8')
draw_arrow(ax, 1.8, 7.2, 1.8, 6.7)

# Layer 2: Dual Feature Extraction (Buildings & Roads)
ax.text(5.5, 8.4, "AI & COMPUTER VISION MODELS", fontsize=9, fontweight='bold', color='#a855f7', fontfamily='sans-serif')
draw_box(ax, 3.8, 7.2, 3.4, 0.9, "Building AI Segmentation", "SpaceNet 7 3-Class ResNet34 U-Net", '#a855f7')
draw_box(ax, 3.8, 5.8, 3.4, 0.9, "Building Polygonizer", "Douglas-Peucker & Instance Attribution", '#a855f7')
draw_arrow(ax, 5.5, 7.2, 5.5, 6.7)

draw_box(ax, 7.8, 7.2, 3.4, 0.9, "Classical Road Extraction", "Morphological CV & Building Exclusion", '#a855f7')
draw_box(ax, 7.8, 5.8, 3.4, 0.9, "Deep Learning Road Model", "ResNet18 U-Net Binary Segmentation", '#a855f7')

# Inter-layer arrows
draw_arrow(ax, 3.2, 6.25, 3.8, 7.65)
draw_arrow(ax, 3.2, 6.25, 7.8, 7.65)
draw_arrow(ax, 3.2, 6.25, 7.8, 6.25)

# Layer 3: Spatial Fusion & Parcel Engine
ax.text(12.5, 8.4, "SPATIAL FUSION & PARCEL ENGINE", fontsize=9, fontweight='bold', color='#f59e0b', fontfamily='sans-serif')
draw_box(ax, 11.8, 7.2, 3.6, 0.9, "Phase 13D Hybrid Road Fusion", "Topological Graph (Nodes & Edges)", '#f59e0b')
draw_box(ax, 11.8, 5.8, 3.6, 0.9, "Preliminary Parcel Generator", "Road-Constrained Voronoi Partitioning", '#f59e0b')
draw_arrow(ax, 13.6, 7.2, 13.6, 6.7)

draw_arrow(ax, 11.2, 7.65, 11.8, 7.65)
draw_arrow(ax, 11.2, 6.25, 11.8, 7.2)
draw_arrow(ax, 7.2, 6.25, 11.8, 6.25)

# Layer 4: Parcel Intelligence & Quality
draw_box(ax, 11.8, 4.4, 3.6, 0.9, "Parcel Intelligence Engine", "4-Component Confidence & Flags", '#f59e0b')
draw_arrow(ax, 13.6, 5.8, 13.6, 5.3)

# Layer 5: WebGIS Verification & Human-in-the-Loop
ax.text(5.5, 3.8, "WEBGIS INTERACTIVE PLATFORM & AUDITING", fontsize=9, fontweight='bold', color='#10b981', fontfamily='sans-serif')
draw_box(ax, 2.0, 2.6, 3.6, 0.9, "FastAPI Async Backend", "REST Endpoints, Real-Time Polling", '#10b981')
draw_box(ax, 6.2, 2.6, 3.6, 0.9, "React + MapLibre WebGIS", "Multi-Layer Visualization & Inspection", '#10b981')
draw_box(ax, 10.4, 2.6, 3.6, 0.9, "Human-in-the-Loop Verification", "Accept, Needs Review, Vertex Editing", '#10b981')

draw_arrow(ax, 13.6, 4.4, 12.2, 3.5)
draw_arrow(ax, 5.6, 3.05, 6.2, 3.05)
draw_arrow(ax, 9.8, 3.05, 10.4, 3.05)

# Layer 6: Audited Deliverables & Exports
ax.text(8.0, 1.9, "STANDARDIZED GIS EXPORTS & AUDIT TRAILS", fontsize=9, fontweight='bold', color='#38bdf8', fontfamily='sans-serif', ha='center')
draw_box(ax, 0.8, 0.7, 3.2, 0.8, "Building Footprints", "GeoJSON + Statistics CSV", '#38bdf8')
draw_box(ax, 4.4, 0.7, 3.2, 0.8, "Hybrid Road Network", "GeoJSON Corridors & Graph", '#38bdf8')
draw_box(ax, 8.0, 0.7, 3.2, 0.8, "Reviewed Parcels", "GIS-Ready GeoJSON", '#38bdf8')
draw_box(ax, 11.6, 0.7, 3.6, 0.8, "Immutable Audit Trail", "parcel_reviews_audit.json", '#38bdf8')

draw_arrow(ax, 3.8, 2.6, 2.4, 1.5)
draw_arrow(ax, 8.0, 2.6, 6.0, 1.5)
draw_arrow(ax, 8.0, 2.6, 9.6, 1.5)
draw_arrow(ax, 12.2, 2.6, 13.4, 1.5)

# Disclaimer Footer
ax.text(8, 0.2, "LEGAL DISCLAIMER: AI-generated parcel boundaries are preliminary spatial candidates requiring professional cadastral survey verification.",
        fontsize=8, color='#f43f5e', ha='center', va='center', fontfamily='sans-serif', fontstyle='italic')

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_PATH, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()

print(f"Architecture diagram generated: {OUT_PATH}")
