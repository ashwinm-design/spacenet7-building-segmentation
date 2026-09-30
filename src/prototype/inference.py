"""
src/prototype/inference.py

Batched model inference and tile stitching for building-footprint segmentation.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from .config import InferenceConfig, DEFAULT_CONFIG
from .preprocessing import extract_and_pad_tile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model import get_model


class BuildingSegmentor:
    """Manages frozen model loading, batched tile inference, and seamless canvas stitching."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Select compute device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Build architecture
        self.model = get_model(
            self.config.arch,
            self.config.encoder_name,
            in_channels=self.config.in_channels,
            classes=self.config.num_classes,
            pretrained=False
        )

        # Load weights
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.config.checkpoint_path}")

        ckpt = torch.load(self.config.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.mean = np.array(self.config.imagenet_mean, dtype=np.float32)
        self.std  = np.array(self.config.imagenet_std, dtype=np.float32)

    def predict_full_image(
        self,
        rgb_image: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        tile_grid: Optional[List[Dict[str, int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs tiled inference over the full image and stitches probabilities.

        Args:
            rgb_image: [H, W, 3] uint8
            valid_mask: [H, W] bool (True for valid image data)
            tile_grid: list of tile descriptors from create_tile_grid

        Returns:
            pred_class: [H, W] uint8 with classes {0: background, 1: interior, 2: boundary}
            building_prob: [H, W] float32 combined probability of interior + boundary
            prob_canvas: [3, H, W] float32 multiclass probability distribution
        """
        h, w = rgb_image.shape[:2]
        tile_size = self.config.tile_size
        batch_size = self.config.batch_size

        if tile_grid is None:
            from .preprocessing import create_tile_grid
            tile_grid = create_tile_grid((h, w), tile_size, self.config.tile_stride)

        # Canvases for accumulation
        prob_canvas = np.zeros((3, h, w), dtype=np.float32)
        weight_canvas = np.zeros((h, w), dtype=np.float32)

        # Run batched inference
        n_tiles = len(tile_grid)
        for i in range(0, n_tiles, batch_size):
            batch_meta = tile_grid[i : i + batch_size]
            batch_tensors = []

            for tile_info in batch_meta:
                top = tile_info["top"]
                left = tile_info["left"]
                padded_tile, _, _ = extract_and_pad_tile(rgb_image, top, left, tile_size)
                
                # Normalize [0..1] -> ImageNet
                norm_tile = (padded_tile.astype(np.float32) / 255.0 - self.mean) / self.std
                t = torch.from_numpy(norm_tile.transpose(2, 0, 1)).float()
                batch_tensors.append(t)

            batch_stack = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_stack)
                probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, 3, tile_size, tile_size]

            # Place back into full-image canvas
            for j, tile_info in enumerate(batch_meta):
                top = tile_info["top"]
                left = tile_info["left"]
                act_h = tile_info["actual_h"]
                act_w = tile_info["actual_w"]

                tile_probs = probs[j, :, :act_h, :act_w]
                prob_canvas[:, top:top+act_h, left:left+act_w] += tile_probs
                weight_canvas[top:top+act_h, left:left+act_w] += 1.0

        # Normalize by overlapping weights
        prob_canvas /= np.maximum(weight_canvas, 1e-6)

        # Mask NoData / invalid pixels
        if valid_mask is not None:
            invalid = ~valid_mask
            prob_canvas[0, invalid] = 1.0
            prob_canvas[1, invalid] = 0.0
            prob_canvas[2, invalid] = 0.0

        pred_class = np.argmax(prob_canvas, axis=0).astype(np.uint8)
        building_prob = (prob_canvas[1] + prob_canvas[2]).astype(np.float32)

        return pred_class, building_prob, prob_canvas
