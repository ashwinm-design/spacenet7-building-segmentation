"""
src/prototype/preprocessing.py

Robust image loading, multi-band handling, alpha/NoData masking, and 256x256 tiling.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import rasterio
from PIL import Image

from .geospatial import read_geospatial_metadata, GeospatialMetadata


def load_image_and_metadata(image_path: str) -> Tuple[np.ndarray, np.ndarray, GeospatialMetadata]:
    """
    Loads an aerial or satellite image (GeoTIFF, PNG, JPEG) and converts to standard RGB uint8.
    
    Returns:
        rgb_image: np.ndarray [H, W, 3] uint8
        valid_mask: np.ndarray [H, W] bool (True where pixels are valid, False for NoData/Alpha=0)
        geo_meta: GeospatialMetadata
    """
    geo_meta = read_geospatial_metadata(image_path)

    # 1. Attempt GeoTIFF loading via rasterio
    if geo_meta.is_geotiff:
        try:
            with rasterio.open(image_path) as ds:
                bands_count = ds.count
                h, w = ds.height, ds.width
                nodata_val = ds.nodata

                if bands_count == 1:
                    # Grayscale GeoTIFF
                    raw = ds.read(1)
                    if raw.dtype != np.uint8:
                        raw = normalize_to_uint8(raw)
                    rgb_image = np.stack([raw, raw, raw], axis=-1)
                    valid_mask = (raw != nodata_val) if nodata_val is not None else np.ones((h, w), dtype=bool)

                elif bands_count == 3:
                    # Standard RGB
                    b1 = ds.read(1)
                    b2 = ds.read(2)
                    b3 = ds.read(3)
                    rgb_image = np.stack([
                        normalize_to_uint8(b1) if b1.dtype != np.uint8 else b1,
                        normalize_to_uint8(b2) if b2.dtype != np.uint8 else b2,
                        normalize_to_uint8(b3) if b3.dtype != np.uint8 else b3,
                    ], axis=-1)
                    if nodata_val is not None:
                        valid_mask = ~((b1 == nodata_val) & (b2 == nodata_val) & (b3 == nodata_val))
                    else:
                        valid_mask = np.ones((h, w), dtype=bool)

                elif bands_count >= 4:
                    # RGBA or Multispectral
                    b1 = ds.read(1)
                    b2 = ds.read(2)
                    b3 = ds.read(3)
                    b4 = ds.read(4)

                    rgb_image = np.stack([
                        normalize_to_uint8(b1) if b1.dtype != np.uint8 else b1,
                        normalize_to_uint8(b2) if b2.dtype != np.uint8 else b2,
                        normalize_to_uint8(b3) if b3.dtype != np.uint8 else b3,
                    ], axis=-1)

                    # Band 4 is typically Alpha or NIR. If alpha, 0 is transparent/nodata.
                    if ds.colorinterp and rasterio.enums.ColorInterp.alpha in ds.colorinterp:
                        valid_mask = (b4 > 0)
                    else:
                        # General NoData check
                        if nodata_val is not None:
                            valid_mask = (b1 != nodata_val) & (b2 != nodata_val)
                        else:
                            # Heuristic: non-zero in alpha or image
                            valid_mask = (b4 > 0) if b4.max() > 0 else np.ones((h, w), dtype=bool)
                else:
                    raise ValueError(f"Unsupported GeoTIFF band count: {bands_count}")

                return rgb_image, valid_mask, geo_meta

        except Exception as e:
            # Fall back to PIL if rasterio read fails
            pass

    # 2. Standard image loading via PIL
    with Image.open(image_path) as pil_img:
        mode = pil_img.mode
        if mode == "RGBA":
            arr = np.array(pil_img)
            rgb_image = arr[:, :, :3]
            valid_mask = (arr[:, :, 3] > 0)
        elif mode == "RGB":
            rgb_image = np.array(pil_img)
            valid_mask = np.ones(rgb_image.shape[:2], dtype=bool)
        elif mode in ("L", "1"):
            arr = np.array(pil_img.convert("L"))
            rgb_image = np.stack([arr, arr, arr], axis=-1)
            valid_mask = np.ones(arr.shape, dtype=bool)
        else:
            conv = pil_img.convert("RGB")
            rgb_image = np.array(conv)
            valid_mask = np.ones(rgb_image.shape[:2], dtype=bool)

    return rgb_image, valid_mask, geo_meta


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalizes arbitrary float or uint16 arrays to 0..255 uint8."""
    if arr.dtype == np.uint8:
        return arr
    min_v, max_v = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if max_v <= min_v:
        max_v = min_v + 1e-5
    scaled = np.clip((arr - min_v) / (max_v - min_v) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def create_tile_grid(
    image_shape: Tuple[int, int],
    tile_size: int = 256,
    tile_stride: int = 256
) -> List[Dict[str, int]]:
    """
    Partitions image dimensions (H, W) into tile coordinates.
    Handles dimensions not divisible by tile_size by capturing remaining edge pixels.
    """
    h, w = image_shape[:2]
    tiles = []
    tile_idx = 0

    top = 0
    while top < h:
        left = 0
        actual_h = min(tile_size, h - top)
        while left < w:
            actual_w = min(tile_size, w - left)
            tiles.append({
                "tile_idx": tile_idx,
                "top": top,
                "left": left,
                "actual_h": actual_h,
                "actual_w": actual_w,
            })
            tile_idx += 1
            left += tile_stride
        top += tile_stride

    return tiles


def extract_and_pad_tile(
    image: np.ndarray,
    top: int,
    left: int,
    tile_size: int = 256
) -> Tuple[np.ndarray, int, int]:
    """
    Extracts a tile from the image and zero-pads it to exactly [tile_size, tile_size, C]
    if the tile lies on an edge with smaller dimensions.
    """
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    
    actual_h = min(tile_size, h - top)
    actual_w = min(tile_size, w - left)
    
    sub_img = image[top:top+actual_h, left:left+actual_w]
    
    if actual_h == tile_size and actual_w == tile_size:
        return sub_img, actual_h, actual_w
        
    # Pad to tile_size x tile_size
    if image.ndim == 3:
        padded = np.zeros((tile_size, tile_size, c), dtype=image.dtype)
        padded[:actual_h, :actual_w, :] = sub_img
    else:
        padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
        padded[:actual_h, :actual_w] = sub_img
        
    return padded, actual_h, actual_w
