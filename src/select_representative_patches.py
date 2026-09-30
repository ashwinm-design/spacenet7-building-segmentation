"""
Refined selection script ensuring diverse spatial coordinates (top, left)
across scenes for each of the 7 categories:
1. isolated_buildings (1-4 discrete separated buildings)
2. dense_buildings (high building density, tight spacing)
3. touching_adjacent (pronounced touching instances fused in binary)
4. small_buildings (clusters of small building footprints)
5. large_buildings (large commercial / institutional structures)
6. irregular_buildings (complex non-rectangular footprints)
7. empty_background (pure landscape / terrain / no buildings)
"""

import json
import cv2
import numpy as np

with open("results/experiments/3class_targets/patch_records.json") as f:
    records = json.load(f)

analyzed = []
for r in records:
    target = cv2.imread(r["target_path"], cv2.IMREAD_GRAYSCALE)
    bin_m = (target > 0).astype(np.uint8)
    interior_m = (target == 1).astype(np.uint8)
    
    if r["is_empty"]:
        analyzed.append({
            **r,
            "num_buildings": 0,
            "max_area": 0,
            "mean_area": 0,
            "irregularity": 0,
            "touching_delta": 0,
            "coord": f"r{r['top']:04d}_c{r['left']:04d}"
        })
        continue
        
    num_bin, _, stats_bin, _ = cv2.connectedComponentsWithStats(bin_m)
    num_int, _, _, _ = cv2.connectedComponentsWithStats(interior_m)
    
    areas = stats_bin[1:, cv2.CC_STAT_AREA] if num_bin > 1 else [0]
    max_area = int(np.max(areas))
    mean_area = float(np.mean(areas))
    touching_delta = (num_int - 1) - (num_bin - 1)
    
    contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregularity = 0.0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 80:
            p = cv2.arcLength(cnt, True)
            ratio = p / (np.sqrt(a) + 1e-5)
            if ratio > irregularity:
                irregularity = ratio
                
    analyzed.append({
        **r,
        "num_buildings": num_bin - 1,
        "max_area": max_area,
        "mean_area": mean_area,
        "irregularity": irregularity,
        "touching_delta": touching_delta,
        "coord": f"r{r['top']:04d}_c{r['left']:04d}"
    })

selected = []
used_coords = set()
used_patches = set()

def pick_distinct(candidates, category, sort_key, reverse=True, n=2):
    sorted_c = sorted(candidates, key=sort_key, reverse=reverse)
    picked = 0
    for c in sorted_c:
        # Prefer new spatial coordinates if possible
        if c["patch_name"] not in used_patches:
            selected.append({"category": category, **c})
            used_patches.add(c["patch_name"])
            used_coords.add(c["coord"])
            picked += 1
            if picked >= n:
                break

# 1. Empty background (0 buildings)
cands = [p for p in analyzed if p["is_empty"]]
pick_distinct(cands, "empty_background", lambda x: x["patch_name"], reverse=False, n=2)

# 2. Isolated buildings (coverage 0.02% to 2.5%, 1 to 5 buildings, separated)
cands = [p for p in analyzed if 0.02 <= p["building_coverage_pct"] <= 3.0 and 1 <= p["num_buildings"] <= 6 and p["touching_delta"] <= 1]
pick_distinct(cands, "isolated_buildings", lambda x: x["building_coverage_pct"], reverse=False, n=2)

# 3. Dense buildings (coverage > 12%, many buildings)
cands = [p for p in analyzed if p["building_coverage_pct"] > 12.0 and p["coord"] not in used_coords]
if not cands: cands = [p for p in analyzed if p["building_coverage_pct"] > 12.0]
pick_distinct(cands, "dense_buildings", lambda x: x["building_coverage_pct"], reverse=True, n=2)

# 4. Touching / adjacent buildings (high touching_delta)
cands = [p for p in analyzed if p["touching_delta"] >= 15 and p["coord"] not in used_coords]
if not cands: cands = [p for p in analyzed if p["touching_delta"] >= 15]
pick_distinct(cands, "touching_adjacent", lambda x: x["touching_delta"], reverse=True, n=2)

# 5. Small buildings (mean area < 90 px, >= 15 buildings)
cands = [p for p in analyzed if p["mean_area"] < 90 and p["num_buildings"] >= 15 and p["coord"] not in used_coords]
if not cands: cands = [p for p in analyzed if p["mean_area"] < 90 and p["num_buildings"] >= 15]
pick_distinct(cands, "small_buildings", lambda x: x["mean_area"], reverse=False, n=2)

# 6. Large buildings (max area > 300 px)
cands = [p for p in analyzed if p["max_area"] >= 280 and p["coord"] not in used_coords]
if not cands: cands = [p for p in analyzed if p["max_area"] >= 280]
pick_distinct(cands, "large_buildings", lambda x: x["max_area"], reverse=True, n=2)

# 7. Irregular buildings (complex polygon shapes)
cands = [p for p in analyzed if p["irregularity"] > 5.0 and p["coord"] not in used_coords]
if not cands: cands = [p for p in analyzed if p["irregularity"] > 5.0]
pick_distinct(cands, "irregular_buildings", lambda x: x["irregularity"], reverse=True, n=2)

print(f"Total selected patches: {len(selected)}")
with open("results/experiments/3class_targets/selected_representative.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2)

for idx, s in enumerate(selected, 1):
    print(f"  {idx:02d}. [{s['category']:<20}] {s['patch_name']} ({s['split']:<5}) | Coord: {s['coord']} | Cov: {s['building_coverage_pct']:5.2f}% | Bldgs: {s['num_buildings']:3d} | MaxA: {s['max_area']:3d} | TouchDelta: {s['touching_delta']:3d}")
