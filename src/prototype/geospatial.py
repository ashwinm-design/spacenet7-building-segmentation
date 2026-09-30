"""
src/prototype/geospatial.py

Geospatial coordinate handling, GeoTIFF metadata inspection, CRS preservation,
and area/perimeter calculations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import transform as shapely_transform
import pyproj


@dataclass
class GeospatialMetadata:
    """Stores geospatial metadata for an input image."""
    is_geotiff: bool
    crs: Optional[CRS]
    transform: Optional[Affine]
    width: int
    height: int
    bands: int
    dtype: str
    resolution: Optional[Tuple[float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    nodata: Optional[float] = None
    crs_wkt: Optional[str] = None
    is_projected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_geotiff": self.is_geotiff,
            "crs": str(self.crs) if self.crs else None,
            "crs_wkt": self.crs_wkt,
            "is_projected": self.is_projected,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": self.dtype,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "nodata": self.nodata,
        }


def read_geospatial_metadata(image_path: str) -> GeospatialMetadata:
    """Inspects an image path and extracts geospatial reference if available."""
    try:
        with rasterio.open(image_path) as ds:
            crs = ds.crs
            transform = ds.transform
            is_proj = bool(crs.is_projected) if crs else False
            res = (abs(float(ds.res[0])), abs(float(ds.res[1]))) if ds.res else None
            bounds = (float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top))
            return GeospatialMetadata(
                is_geotiff=bool(crs is not None and transform is not None),
                crs=crs,
                transform=transform,
                width=ds.width,
                height=ds.height,
                bands=ds.count,
                dtype=str(ds.dtypes[0]),
                resolution=res,
                bounds=bounds,
                nodata=float(ds.nodata) if ds.nodata is not None else None,
                crs_wkt=crs.to_wkt() if crs else None,
                is_projected=is_proj,
            )
    except Exception:
        # Fallback for non-geotiff files
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
            bands = len(im.getbands())
            mode = im.mode
            return GeospatialMetadata(
                is_geotiff=False,
                crs=None,
                transform=None,
                width=w,
                height=h,
                bands=bands,
                dtype=mode,
                resolution=None,
                bounds=None,
                nodata=None,
                crs_wkt=None,
                is_projected=False,
            )


def pixel_to_geo_transform(poly: Polygon, transform: Affine) -> Polygon:
    """Transforms a Shapely polygon from pixel coordinates (col, row) to geospatial coordinates."""
    def forward_fn(x, y):
        cols = np.asarray(x)
        rows = np.asarray(y)
        # Affine mapping: x_geo = a*col + b*row + c, y_geo = d*col + e*row + f
        geo_x = transform.a * cols + transform.b * rows + transform.c
        geo_y = transform.d * cols + transform.e * rows + transform.f
        return geo_x, geo_y

    return shapely_transform(forward_fn, poly)


def calculate_polygon_metrics(
    poly: Polygon,
    geo_meta: GeospatialMetadata,
    pixel_poly: Optional[Polygon] = None
) -> Dict[str, Any]:
    """
    Computes area, perimeter, centroid, and bounding box for a building polygon.
    Ensures that physical area is NEVER computed directly from raw lat/lon degrees.
    """
    if pixel_poly is None:
        pixel_poly = poly

    px_area = float(pixel_poly.area)
    px_perimeter = float(pixel_poly.length)
    px_centroid = pixel_poly.centroid

    if not geo_meta.is_geotiff or geo_meta.crs is None:
        # Plain non-georeferenced RGB image
        minx, miny, maxx, maxy = pixel_poly.bounds
        return {
            "area": round(px_area, 2),
            "perimeter": round(px_perimeter, 2),
            "area_unit": "pixels^2",
            "perimeter_unit": "pixels",
            "centroid_x": round(float(px_centroid.x), 2),
            "centroid_y": round(float(px_centroid.y), 2),
            "bbox_minx": round(float(minx), 2),
            "bbox_miny": round(float(miny), 2),
            "bbox_maxx": round(float(maxx), 2),
            "bbox_maxy": round(float(maxy), 2),
            "is_projected": False,
        }

    # Georeferenced polygon
    minx, miny, maxx, maxy = poly.bounds
    centroid = poly.centroid

    if geo_meta.is_projected:
        # CRS is already in meters (e.g. UTM)
        phys_area = float(poly.area)
        phys_perimeter = float(poly.length)
        return {
            "area": round(phys_area, 2),
            "perimeter": round(phys_perimeter, 2),
            "area_unit": "m^2",
            "perimeter_unit": "meters",
            "centroid_x": round(float(centroid.x), 4),
            "centroid_y": round(float(centroid.y), 4),
            "bbox_minx": round(float(minx), 4),
            "bbox_miny": round(float(miny), 4),
            "bbox_maxx": round(float(maxx), 4),
            "bbox_maxy": round(float(maxy), 4),
            "is_projected": True,
        }
    else:
        # Geographic CRS (lat/lon in degrees, e.g. EPSG:4326)
        # Reproject to an appropriate local UTM zone to compute metric area
        lon, lat = centroid.x, centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        
        project = pyproj.Transformer.from_crs(geo_meta.crs, utm_crs, always_xy=True).transform
        utm_poly = shapely_transform(project, poly)
        
        phys_area = float(utm_poly.area)
        phys_perimeter = float(utm_poly.length)
        
        return {
            "area": round(phys_area, 2),
            "perimeter": round(phys_perimeter, 2),
            "area_unit": "m^2 (reprojected)",
            "perimeter_unit": "meters (reprojected)",
            "centroid_x": round(float(centroid.x), 6),
            "centroid_y": round(float(centroid.y), 6),
            "bbox_minx": round(float(minx), 6),
            "bbox_miny": round(float(miny), 6),
            "bbox_maxx": round(float(maxx), 6),
            "bbox_maxy": round(float(maxy), 6),
            "is_projected": False,
        }


def transform_polygons_to_crs(
    pixel_polygons: List[Polygon],
    geo_meta: GeospatialMetadata
) -> List[Polygon]:
    """Converts a list of pixel-space polygons into source geospatial coordinates."""
    if not geo_meta.is_geotiff or geo_meta.transform is None:
        return pixel_polygons

    geo_polys = []
    for p in pixel_polygons:
        gp = pixel_to_geo_transform(p, geo_meta.transform)
        if not gp.is_valid:
            gp = gp.buffer(0)
        if not gp.is_empty:
            geo_polys.append(gp)
    return geo_polys
