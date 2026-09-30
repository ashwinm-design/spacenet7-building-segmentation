"""
src/road_dl/dataset.py

Dataset preparation, patch slicing, and PyTorch Dataset class for road extraction.
Slices high-resolution satellite/aerial images into 256x256 patches matching the
native receptive field of the U-Net ResNet18 encoder pipeline.
"""

import os
import glob
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def prepare_benchmark_patches(
    raw_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Slices 1024x1024 benchmark tiles into 256x256 training/validation patches.
    Saves patches and generates deterministic train/val splits.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    img_out = output_dir / "images"
    mask_out = output_dir / "masks"
    splits_out = output_dir / "splits"

    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    sat_files = sorted(glob.glob(str(raw_dir / "*_sat.jpg")))
    patch_ids = []

    for sat_path in sat_files:
        base_name = Path(sat_path).stem.replace("_sat", "")
        mask_path = raw_dir / f"{base_name}_mask.png"
        if not mask_path.is_file():
            continue

        img = cv2.imread(sat_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8) * 255

        h, w = img.shape[:2]
        r_steps = h // patch_size
        c_steps = w // patch_size

        for r in range(r_steps):
            for c in range(c_steps):
                y1 = r * patch_size
                y2 = y1 + patch_size
                x1 = c * patch_size
                x2 = x1 + patch_size

                patch_img = img[y1:y2, x1:x2]
                patch_mask = mask_bin[y1:y2, x1:x2]

                patch_name = f"{base_name}_r{r:02d}_c{c:02d}"
                cv2.imwrite(str(img_out / f"{patch_name}.png"), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_out / f"{patch_name}.png"), patch_mask)
                patch_ids.append(patch_name)

    # Deterministic split
    rng = random.Random(seed)
    rng.shuffle(patch_ids)
    n_train = int(len(patch_ids) * train_ratio)
    train_ids = sorted(patch_ids[:n_train])
    val_ids = sorted(patch_ids[n_train:])

    split_data = {
        "dataset_name": "deepglobe_curated_benchmark",
        "patch_size": patch_size,
        "total_patches": len(patch_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_patches": train_ids,
        "val_patches": val_ids,
    }

    with open(splits_out / "train_val_split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    return split_data


class RoadPatchDataset(Dataset):
    """
    PyTorch Dataset yielding (image, mask) pairs for binary road segmentation.
    Images normalized with ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    """
    def __init__(
        self,
        patch_dir: Path,
        patch_ids: List[str],
        augment: bool = False,
    ):
        self.patch_dir = Path(patch_dir)
        self.patch_ids = patch_ids
        self.augment = augment
        self.img_dir = self.patch_dir / "images"
        self.mask_dir = self.patch_dir / "masks"

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.patch_ids[idx]
        img_p = self.img_dir / f"{pid}.png"
        mask_p = self.mask_dir / f"{pid}.png"

        img = cv2.imread(str(img_p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                img = np.rot90(img, k=rot_k).copy()
                mask = np.rot90(mask, k=rot_k).copy()

        # Normalize image to float32 tensor [3, H, W]
        img_f = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1))

        # Binary road target [1, H, W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_t, mask_t
