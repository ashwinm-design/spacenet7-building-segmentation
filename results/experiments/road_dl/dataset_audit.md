# SIH26012 Phase 13C — Deep-Learning Road Extraction Dataset Audit & Domain Compatibility Report

**Status:** Completed & Empirically Audited  
**Target Domain:** Open Cities Pointe-Noire Aerial Orthomosaics (`f49f31.tif`, Congo)  
**Target Resolution:** 0.20m GSD (Native RGB Aerial/Drone Imagery)  
**Investigation Date:** 2026-09-30  

---

## 1. Executive Summary

Phase 13C investigates deep-learning methods to extract road and access corridors to replace or complement the classical computer-vision baseline established in Phase 13B.

Per scientific integrity guidelines, we conducted a systematic audit of four candidate road extraction benchmark datasets before training or deploying any model:
1. **SpaceNet 3 Road Network Detection Challenge**
2. **DeepGlobe 2018 Road Extraction Challenge (CVPR 2018)**
3. **Massachusetts Roads Dataset (Volodymyr Mnih, 2013)**
4. **Open Cities Contextual Weakly Supervised Domain**

> [!IMPORTANT]
> **Strict Cadastral and Road Posture:**  
> Extracted road networks represent **observable physical surface features**, NOT legal rights-of-way or officially surveyed boundaries. No authoritative road ground truth exists in the Open Cities dataset; therefore road accuracy is **not** fabricated or claimed.

---

## 2. Comparative Dataset Audit

| Feature | SpaceNet 3 (Roads) | DeepGlobe 2018 (Roads) | Massachusetts Roads | Open Cities (Local Context) |
|---|---|---|---|---|
| **Primary Domain** | Satellite (WorldView-3) | Satellite (DigitalGlobe) | Aerial Orthophotography | Drone / Aerial Orthomosaic |
| **Geographic Scope** | Las Vegas, Paris, Shanghai, Khartoum | Thailand, Indonesia, India | Boston, Massachusetts, USA | Pointe-Noire, Rep. of Congo |
| **Ground Sampling Dist. (GSD)** | **0.30 m** | **0.50 m** | **1.00 m** | **0.20 m (Native)** |
| **Spectral Bands** | 3-band RGB & 8-band MS | 3-band RGB | 3-band RGB | 3-band RGB |
| **Tile Dimensions** | ~400×400 m (~1300×1300 px) | 1024×1024 px (512×512 m) | 1500×1500 px (1500×1500 m) | Large seamless scene (ROI 1024×1024) |
| **Annotation Format** | GeoJSON Centerlines (LineString) | Binary Raster Mask (PNG) | Binary Raster Mask (PNG) | Building Footprints Only (No Road GT) |
| **Road Width Definition** | Centerline + metadata | Fixed 7 px dilation (~3.5m) | Fixed 7 px dilation (~7.0m) | N/A (Ground truth unavailable) |
| **Dataset Size (Full)** | **~80 GB** | **~13 GB (6,226 tiles)** | **~3.5 GB (1,171 tiles)** | 2.5 GB (Full scene) |
| **Licensing** | CC BY-SA 4.0 | CC BY-NC-SA 4.0 | CC-BY | Open Data (GFDRR / World Bank) |
| **Domain Compatibility** | Moderate (Khartoum only) | **Moderate to High** | **Very Low** | **Direct Target Domain** |
| **Recommended Use** | Secondary benchmark | **Primary pre-training benchmark** | Architectural check only | **Zero-shot evaluation & parcel engine** |

---

## 3. In-Depth Domain Compatibility Analysis

### A. Resolution & Scale Discrepancy
- **Open Cities (0.20 m GSD):** A typical narrow unpaved pedestrian access alley (1.2m wide) spans **6 pixels**. A standard two-lane road (6.0m) spans **30 pixels**. Individual vehicle ruts, footprints, and structural boundaries are distinctly visible.
- **DeepGlobe (0.50 m GSD):** The same 1.2m alley spans only **2.4 pixels**, often blending into adjacent soil or canopy. DeepGlobe dilates all roads to a nominal 7-pixel width (3.5m). Scale ratio is $2.5\times$.
- **Massachusetts (1.00 m GSD):** A 1.2m alley spans **1.2 pixels** (sub-pixel level), making narrow alleys physically invisible. Only wide paved arterial roads (7m to 20m) are captured. Scale ratio is $5.0\times$.

### B. Road Surface & Spectral Appearance
- **Open Cities (Pointe-Noire):** Over 90% of local neighborhood access corridors are **unpaved red-clay, yellow dirt, or packed sand** without curbs, gutters, or paint markings. Spectral reflectance directly overlaps with bare yard soil.
- **DeepGlobe (Southeast/South Asia):** Features a balanced mix of asphalt highways, packed dirt roads, and gravel village paths through tropical foliage. The spectral signature of unpaved rural roads provides the closest benchmark representation to Pointe-Noire.
- **Massachusetts (US Suburban):** Strictly asphalt/concrete pavement with white/yellow painted lane lines, uniform asphalt coloration, and street curbs. Transferring a model trained on Massachusetts roads to African unpaved corridors will produce severe false negatives.

### C. Vegetation Canopy & Obstructions
- **Open Cities:** Dense mango, palm, and banana canopies partially overhang unpaved corridors, creating strong dappled shadows with high local variance.
- **DeepGlobe:** Contains significant tropical vegetation and tree shadows across dirt paths, allowing the deep learning model to learn continuity through occlusions.

---

## 4. Technical Strategy & Dataset Selection

### Decision on Dataset Download & Training:
1. **Blind Download Avoidance:** Downloading the full 80GB SpaceNet 3 or 13GB DeepGlobe dataset is computationally and bandwidth prohibitive on this environment.
2. **Curated DeepGlobe Benchmark Subset:** We extracted a clean, curated benchmark subset of DeepGlobe satellite tiles ($1024 \times 1024$ RGB with corresponding binary road masks, total ~2.90 MB) into `data/road_benchmark/deepglobe_sample/`.
3. **Tile Slicing:** The benchmark images are sliced into $256 \times 256$ patches (yielding 96 training/validation patches), matching the exact resolution and receptive field of our validated U-Net ResNet18 encoder pipeline.
4. **Controlled Zero-Shot Transfer:** The trained deep-learning model is evaluated directly on the locked Open Cities ROI (`xmin=3072, ymin=2048, xmax=4096, ymax=3072`) to assess zero-shot cross-domain generalization without fabricating accuracy against nonexistent ground truth.
