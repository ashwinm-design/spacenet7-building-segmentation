"""
src/prototype package

SIH26012 Prototype Inference Engine:
High-resolution aerial/drone building-footprint segmentation and instance polygonization.
"""

from .config import InferenceConfig, DEFAULT_CONFIG
from .geospatial import read_geospatial_metadata, transform_polygons_to_crs, calculate_polygon_metrics
from .preprocessing import load_image_and_metadata, create_tile_grid, extract_and_pad_tile
from .inference import BuildingSegmentor
from .polygonization import extract_building_polygons, build_footprint_geodataframe, save_inference_outputs

__all__ = [
    "InferenceConfig",
    "DEFAULT_CONFIG",
    "read_geospatial_metadata",
    "transform_polygons_to_crs",
    "calculate_polygon_metrics",
    "load_image_and_metadata",
    "create_tile_grid",
    "extract_and_pad_tile",
    "BuildingSegmentor",
    "extract_building_polygons",
    "build_footprint_geodataframe",
    "save_inference_outputs",
]
