# SpaceNet 7 Sample Dataset — Inspection Report

> **Project**: AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction (SIH 2026)  
> **Date**: 2026-09-28  
> **Dataset**: SpaceNet 7 Multi-Temporal Urban Development Training Sample

---

## 1. Dataset Root

```
c:\Users\Ashwinpc\OneDrive\Desktop\sih project\sample\L15-0506E-1204N_2027_3374_13\
```

The dataset represents a **single Area of Interest (AOI)** tile identified as `L15-0506E-1204N` at tile position `2027_3374` zoom `13`.

**Geographic location**: Approximately **30.19°N, -90.89°W** (Louisiana, USA — near New Orleans).

---

## 2. Directory Structure

```
sample/
└── L15-0506E-1204N_2027_3374_13/
    ├── images/                  # 24 GeoTIFF satellite images
    ├── images_masked/           # 24 GeoTIFF images with UDM masking applied
    ├── labels/                  # 48 GeoJSON files (24 Buildings + 24 UDM)
    ├── labels_match/            # 24 GeoJSON building labels (EPSG:3857 coords, with tracking IDs)
    ├── labels_match_pix/        # 24 GeoJSON building labels (pixel coordinates)
    └── UDM_masks/               # 2 UDM (Usable Data Mask) raster TIFFs
```

---

## 3. File Counts

| Directory         | File Count | File Type       | Total Size  |
|--------------------|-----------|-----------------|-------------|
| `images/`          | 24        | `.tif`          | ~96 MB      |
| `images_masked/`   | 24        | `.tif`          | ~96 MB      |
| `labels/`          | 48        | `.geojson`      | ~20 MB      |
| `labels_match/`    | 24        | `.geojson`      | ~26 MB      |
| `labels_match_pix/`| 24        | `.geojson`      | ~27 MB      |
| `UDM_masks/`       | 2         | `.tif`          | ~2 MB       |
| **Total**          | **146**   | `.tif` + `.geojson` | **~264 MB** |

---

## 4. Image Information

| Property                | Value                          |
|--------------------------|-------------------------------|
| **Format**               | GeoTIFF (`.tif`)             |
| **Dimensions**           | 1024 × 1024 pixels           |
| **Bands**                | 4 (likely R, G, B, NIR or R, G, B, Alpha) |
| **Bits per sample**      | Multi-value (likely 8-bit per band) |
| **Compression**          | None (tag 259 = 1)           |
| **Photometric**          | RGB (tag 262 = 2)            |
| **File size**            | ~4.0 MB each                 |
| **Source**               | Planet Labs monthly mosaics   |
| **Approximate GSD**      | ~4m (Planet Dove constellation) |

### Filename Pattern
```
global_monthly_{YEAR}_{MONTH}_mosaic_{TILE_ID}_{COL}_{ROW}_{ZOOM}.tif
```
- `global_monthly` — Planet global monthly basemap mosaic
- `{YEAR}_{MONTH}` — Temporal identifier (2018-01 through 2019-12)
- `mosaic` — Composite mosaic product
- `L15-0506E-1204N` — Quadkey tile reference (lat/lon encoded)
- `2027_3374_13` — Web Mercator tile column, row, zoom level 13

### Temporal Coverage
- **Date range**: January 2018 → December 2019
- **Total months**: 24 (complete monthly series)
- **Same geographic area** captured every month → supports change detection

---

## 5. Label Information

### 5.1 `labels/` — Original Building Footprint Labels (CRS84 / WGS84)

| Property              | Value |
|------------------------|-------|
| **Format**             | GeoJSON FeatureCollection |
| **CRS**                | `urn:ogc:def:crs:OGC:1.3:CRS84` (WGS84 lon/lat) |
| **Geometry type**      | Polygon |
| **Properties**         | `Id` (integer, all 0 — no unique building tracking) |
| **Features per file**  | 1,864 – 2,475 buildings (varies by month) |
| **Coordinate range**   | lon ≈ [-90.92, -90.88], lat ≈ [30.18, 30.21] |

Also contains 24 UDM (Usable Data Mask) GeoJSON files — mostly empty (217 bytes), except June 2018 and November 2019 which have mask polygons.

### 5.2 `labels_match/` — Building Labels with Tracking IDs (EPSG:3857)

| Property              | Value |
|------------------------|-------|
| **CRS**                | `urn:ogc:def:crs:EPSG::3857` (Web Mercator) |
| **Geometry type**      | Polygon |
| **Properties**         | `Id` (unique building ID for tracking across time), `area` (m²), `iou_score` |
| **Features per file**  | Same count as `labels/` |

These labels assign **unique building IDs** across months, enabling temporal change tracking. The `iou_score` property appears to default to 0 in this sample.

### 5.3 `labels_match_pix/` — Building Labels in Pixel Coordinates ⭐

| Property              | Value |
|------------------------|-------|
| **CRS**                | None (pixel coordinate space) |
| **Geometry type**      | Polygon |
| **Properties**         | `Id` (unique building ID), `area` (pixels²), `image_fname` (corresponding TIF filename) |
| **Features per file**  | Same count as `labels/` |
| **Coordinate range**   | x ∈ [-0.1, 1024.1], y ∈ [-1.0, 1019.0] |

> **⭐ THIS IS THE KEY DATASET for rasterization.** Polygons are already in pixel coordinates aligned to the 1024×1024 images. The `image_fname` property explicitly links each polygon to its source image.

---

## 6. Coordinate / Reference Information

| Label Directory   | CRS              | Coordinate Space    | Use Case |
|--------------------|------------------|---------------------|----------|
| `labels/`          | CRS84 (WGS84)   | Geographic (lon/lat) | GIS analysis, real-world georef |
| `labels_match/`    | EPSG:3857        | Projected (meters)   | Spatial analysis, area calculations |
| `labels_match_pix/`| None (pixel)     | Image pixel space    | **Direct rasterization to masks** |

---

## 7. Image-Label Relationship

### How images map to labels:

Each image file has a **1:1 correspondence** with label files across all three label directories. The relationship is encoded in the filename:

```
IMAGE:    global_monthly_2018_01_mosaic_L15-0506E-1204N_2027_3374_13.tif
LABEL:    global_monthly_2018_01_mosaic_L15-0506E-1204N_2027_3374_13_Buildings.geojson
                                                                     ^^^^^^^^^^^^^^^^^^^
                                                                     suffix added
```

**All 24 images** have corresponding labels in all three directories ✅

### Multi-temporal structure:
- All 24 images cover the **exact same geographic area**
- Each month has **different building annotations** (building count grows over time: 2,273 → 2,475)
- This reflects **real urban development** over the 2-year period

### For mask creation:
The `labels_match_pix/` directory is optimal:
1. Polygons are already in **pixel coordinates** (no reprojection needed)
2. Each GeoJSON explicitly references its source image via `image_fname`
3. Coordinates align with the 1024×1024 image grid

**To create a binary mask**: Read polygon coordinates from `labels_match_pix/`, fill polygons onto a 1024×1024 canvas.

---

## 8. Example Image-Label Pair

| Item | Path |
|------|------|
| **Image** | `sample/.../images/global_monthly_2018_01_mosaic_L15-0506E-1204N_2027_3374_13.tif` |
| **Label (pixel)** | `sample/.../labels_match_pix/global_monthly_2018_01_mosaic_L15-0506E-1204N_2027_3374_13_Buildings.geojson` |
| **Building count** | 2,273 polygons |
| **Pixel extent** | x=[-0.1, 1024.1], y=[-1.0, 1019.0] |
| **Image dims** | 1024 × 1024 × 4 bands |

> ⚠️ Minor edge case: Some polygon vertices extend slightly beyond image bounds (by ~1 pixel). This is normal for vector data and should be clipped during rasterization.

---

## 9. Potential Issues

| Issue | Severity | Mitigation |
|-------|----------|------------|
| **4-band images** (not 3-band RGB) | Medium | Use only first 3 bands (RGB), or investigate if band 4 is Alpha/NIR |
| **Polygons slightly exceed image bounds** | Low | Clip coordinates to [0, 1024] during rasterization |
| **All `Id` = 0 in `labels/`** | Info | Use `labels_match/` or `labels_match_pix/` for unique building IDs |
| **Only 1 AOI in sample** | Info | Full SpaceNet 7 dataset has ~100 AOIs; sample is sufficient for pipeline development |
| **UDM masks for only 2 months** | Low | June 2018 and November 2019 have cloud/quality masks; others appear cloud-free |
| **Building count drops in June 2018** (1,864 vs ~2,300) | Info | Likely due to cloud cover (UDM mask exists for that month) |
| **No `.shp`, `.csv`, or `.gpkg` files found** | Info | All labels are in GeoJSON format |
| **Existing `dataset.py` expects COCO format** | Medium | Need new rasterization logic for SpaceNet GeoJSON format |

---

## 10. Recommended Next Steps

1. **Write a rasterization script** that:
   - Reads one `labels_match_pix/` GeoJSON file
   - Fills building polygons onto a blank 1024×1024 binary mask
   - Saves as PNG (0 = background, 255 = building)
   
2. **Verify one example visually** by overlaying the mask on the RGB image

3. **Investigate band 4** — determine if it's Alpha or NIR using `rasterio` or by reading TIFF tags

4. **Extend to all 24 months** once the single-pair pipeline works

5. **Consider tiling** — the 1024×1024 images may need to be cut into 256×256 patches for U-Net training

---

## Summary Table

| Metric | Value |
|--------|-------|
| Dataset | SpaceNet 7 Training Sample |
| AOI | L15-0506E-1204N (Louisiana, USA) |
| Images | 24 GeoTIFF (1024×1024×4 bands) |
| Labels | 24 GeoJSON per label type × 3 types |
| Temporal span | Jan 2018 – Dec 2019 (monthly) |
| Buildings per image | 1,864 – 2,475 |
| Total dataset size | ~264 MB |
| Best label for masks | `labels_match_pix/` (pixel coords) |
