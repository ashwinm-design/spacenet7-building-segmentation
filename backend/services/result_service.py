"""
backend/services/result_service.py

Retrieves analysis results, GeoJSON features, CSV statistics, and building properties.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from backend.storage import storage


def get_analysis_metadata(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves stored metadata for an analysis job."""
    return storage.get_metadata(analysis_id)


def get_geojson_dict(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Reads and parses the stored buildings.geojson as a Python dictionary."""
    p = storage.get_geojson_path(analysis_id)
    if not p or not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_building_feature(analysis_id: str, building_id: int) -> Optional[Dict[str, Any]]:
    """
    Finds and returns attributes and geometry for a specific building ID
    from the analysis GeoJSON.
    """
    gj = get_geojson_dict(analysis_id)
    if not gj or "features" not in gj:
        return None

    for feat in gj["features"]:
        props = feat.get("properties", {})
        if props.get("id") == building_id:
            return {
                "id": props.get("id"),
                "area": props.get("area"),
                "perimeter": props.get("perimeter"),
                "area_unit": props.get("area_unit"),
                "perimeter_unit": props.get("perimeter_unit"),
                "centroid_x": props.get("centroid_x"),
                "centroid_y": props.get("centroid_y"),
                "confidence": props.get("confidence"),
                "min_confidence": props.get("min_confidence"),
                "is_georeferenced": props.get("is_georeferenced", False),
                "geometry": feat.get("geometry"),
                "feature": feat,
            }

    return None
