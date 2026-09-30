"""
src/health_check.py

SIH26012 Comprehensive System Health Check.
Verifies Python runtime, deep learning packages, geospatial libraries,
frozen model checkpoints, input imagery, backend availability, and frontend build.

Usage:
    python src/health_check.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Safe terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_subsystems():
    results = {}

    # 1. Python runtime
    py_ver = sys.version_info
    results["Python"] = {
        "status": "PASS" if (py_ver.major == 3 and py_ver.minor >= 10) else "FAIL",
        "detail": f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    }

    # 2. PyTorch
    try:
        import torch
        results["PyTorch"] = {
            "status": "PASS",
            "detail": f"v{torch.__version__} (CUDA: {torch.cuda.is_available()})",
        }
    except Exception as e:
        results["PyTorch"] = {"status": "FAIL", "detail": str(e)}

    # 3. Model checkpoint (Building)
    bldg_cp = PROJECT_ROOT / "results/checkpoints/best_model.pth"
    if bldg_cp.is_file() and bldg_cp.stat().st_size > 100 * 1024 * 1024:
        results["Model checkpoint"] = {
            "status": "PASS",
            "detail": f"{bldg_cp.stat().st_size / (1024*1024):.1f} MB (SpaceNet 7 ResNet34)",
        }
    else:
        results["Model checkpoint"] = {"status": "FAIL", "detail": "Missing or corrupted"}

    # 4. Road DL checkpoint
    road_cp = PROJECT_ROOT / "results/experiments/road_dl/checkpoints/best_road_model.pth"
    if road_cp.is_file() and road_cp.stat().st_size > 40 * 1024 * 1024:
        results["Road DL checkpoint"] = {
            "status": "PASS",
            "detail": f"{road_cp.stat().st_size / (1024*1024):.1f} MB (ResNet18 U-Net)",
        }
    else:
        results["Road DL checkpoint"] = {"status": "FAIL", "detail": "Missing or corrupted"}

    # 5. Demo imagery
    demo_tif = PROJECT_ROOT / "data/open_cities_raw/ptn/f49f31/f49f31.tif"
    if demo_tif.is_file() and demo_tif.stat().st_size > 5 * 1024 * 1024:
        results["Demo imagery"] = {
            "status": "PASS",
            "detail": f"{demo_tif.stat().st_size / (1024*1024):.1f} MB (f49f31.tif)",
        }
    else:
        results["Demo imagery"] = {"status": "FAIL", "detail": "Missing or corrupted"}

    # 6. GeoPandas
    try:
        import geopandas as gpd
        results["GeoPandas"] = {"status": "PASS", "detail": f"v{gpd.__version__}"}
    except Exception as e:
        results["GeoPandas"] = {"status": "FAIL", "detail": str(e)}

    # 7. Rasterio
    try:
        import rasterio
        results["Rasterio"] = {"status": "PASS", "detail": f"v{rasterio.__version__}"}
    except Exception as e:
        results["Rasterio"] = {"status": "FAIL", "detail": str(e)}

    # 8. Shapely
    try:
        import shapely
        results["Shapely"] = {"status": "PASS", "detail": f"v{shapely.__version__}"}
    except Exception as e:
        results["Shapely"] = {"status": "FAIL", "detail": str(e)}

    # 9. Backend check
    # Check if live HTTP service is up, else check backend module importable
    try:
        import requests
        resp = requests.get("http://127.0.0.1:8000/api/health", timeout=1.5)
        if resp.status_code == 200:
            results["Backend"] = {"status": "PASS", "detail": "Live on http://127.0.0.1:8000"}
        else:
            results["Backend"] = {"status": "WARN", "detail": f"HTTP {resp.status_code}"}
    except Exception:
        # Fallback: check if FastAPI app module can be loaded
        try:
            from backend.main import app
            results["Backend"] = {"status": "PASS", "detail": "Importable (Offline)"}
        except Exception as e:
            results["Backend"] = {"status": "FAIL", "detail": str(e)}

    # 10. Frontend build
    dist_index = PROJECT_ROOT / "frontend/dist/index.html"
    dist_js = list((PROJECT_ROOT / "frontend/dist/assets").glob("*.js")) if (PROJECT_ROOT / "frontend/dist/assets").is_dir() else []
    if dist_index.is_file() and len(dist_js) > 0:
        results["Frontend build"] = {
            "status": "PASS",
            "detail": f"Built ({dist_index.stat().st_size} bytes HTML, {dist_js[0].stat().st_size / 1024:.1f} KB JS)",
        }
    else:
        results["Frontend build"] = {
            "status": "WARN",
            "detail": "frontend/dist/ missing (run 'npm run build' in frontend/)",
        }

    return results


def run_health_check():
    results = check_subsystems()

    # Determine overall status
    all_pass = all(v["status"] == "PASS" for v in results.values())
    overall_status = "READY" if all_pass else "NOT READY"

    print("========================================")
    print("SIH26012 SYSTEM HEALTH CHECK")
    print("========================================")
    for name, info in results.items():
        print(f"{name:<24} {info['status']}")

    print("\nOverall:")
    print(overall_status)
    print("========================================")

    # Save to phase16_hardening
    out_dir = PROJECT_ROOT / "results/experiments/phase16_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": overall_status,
        "subsystems": results,
    }
    with open(out_dir / "health_check.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return overall_status == "READY"


if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
