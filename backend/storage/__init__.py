"""
backend/storage/__init__.py

Manages persistence of analysis uploads, generated GeoJSON, CSV statistics,
PNG overlays, and metadata JSON under results/prototype/api/<analysis_id>/.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

API_STORAGE_ROOT = Path("results/prototype/api")


class AnalysisStorage:
    """Manages file storage and directory structures for analysis jobs."""

    def __init__(self, root_dir: Path = API_STORAGE_ROOT):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_analysis_dir(self, analysis_id: str) -> Path:
        """Returns the directory path for an analysis_id."""
        return self.root_dir / analysis_id

    def exists(self, analysis_id: str) -> bool:
        """Returns True if the analysis directory exists and contains metadata."""
        p = self.get_analysis_dir(analysis_id)
        return p.is_dir() and (p / "metadata.json").is_file()

    def create_analysis_dir(self, analysis_id: str) -> Path:
        """Creates and returns a new analysis directory."""
        p = self.get_analysis_dir(analysis_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_uploaded_file(self, analysis_id: str, filename: str, content: bytes) -> Path:
        """Saves uploaded binary content into the analysis directory."""
        dir_p = self.create_analysis_dir(analysis_id)
        # Preserve original extension
        ext = Path(filename).suffix.lower()
        target_path = dir_p / f"input{ext}"
        with open(target_path, "wb") as f:
            f.write(content)
        return target_path

    def save_metadata(self, analysis_id: str, metadata: Dict[str, Any]) -> Path:
        """Saves metadata.json into the analysis directory."""
        dir_p = self.get_analysis_dir(analysis_id)
        dir_p.mkdir(parents=True, exist_ok=True)
        meta_path = dir_p / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return meta_path

    def get_metadata(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Reads metadata.json for the given analysis_id if it exists."""
        meta_path = self.get_analysis_dir(analysis_id) / "metadata.json"
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_geojson_path(self, analysis_id: str) -> Optional[Path]:
        p = self.get_analysis_dir(analysis_id) / "buildings.geojson"
        return p if p.is_file() else None

    def get_statistics_path(self, analysis_id: str) -> Optional[Path]:
        p = self.get_analysis_dir(analysis_id) / "statistics.csv"
        return p if p.is_file() else None

    def get_roads_geojson_path(self, analysis_id: str, source: str = "classical") -> Optional[Path]:
        if source == "hybrid":
            p = self.get_analysis_dir(analysis_id) / "roads_hybrid.geojson"
            if p.is_file():
                return p
            exp_p = Path("results/experiments/road_hybrid/roads_hybrid.geojson")
            return exp_p if exp_p.is_file() else None

        if source == "deep_learning":
            p = self.get_analysis_dir(analysis_id) / "roads_deep_learning.geojson"
            if p.is_file():
                return p
            exp_p = Path("results/experiments/road_dl/roads_deep_learning.geojson")
            return exp_p if exp_p.is_file() else None

        p = self.get_analysis_dir(analysis_id) / "roads.geojson"
        if p.is_file():
            return p
        exp_p = Path("results/experiments/road_extraction/roads.geojson")
        return exp_p if exp_p.is_file() else None

    def get_parcels_geojson_path(self, analysis_id: str, source: str = "classical") -> Optional[Path]:
        if source == "hybrid":
            p = self.get_analysis_dir(analysis_id) / "parcels_hybrid.geojson"
            if p.is_file():
                return p
            p2 = self.get_analysis_dir(analysis_id) / "candidate_parcels_hybrid_constrained.geojson"
            if p2.is_file():
                return p2
            exp_p = Path("results/experiments/road_hybrid/candidate_parcels_hybrid_constrained.geojson")
            return exp_p if exp_p.is_file() else None

        if source == "deep_learning":
            p = self.get_analysis_dir(analysis_id) / "candidate_parcels_dl_constrained.geojson"
            if p.is_file():
                return p
            exp_p = Path("results/experiments/road_dl/candidate_parcels_dl_constrained.geojson")
            return exp_p if exp_p.is_file() else None

        p = self.get_analysis_dir(analysis_id) / "candidate_parcels.geojson"
        if p.is_file():
            return p
        exp_p = Path("results/experiments/road_constrained_parcels/candidate_parcels_road_constrained.geojson")
        return exp_p if exp_p.is_file() else None

    def get_overlay_path(self, analysis_id: str) -> Optional[Path]:
        p = self.get_analysis_dir(analysis_id) / "overlay.png"
        return p if p.is_file() else None


# Default storage instance
storage = AnalysisStorage()
