import os
import json
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ── Augmentation settings (requirement G) ────────────────────────────────
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotate_90": True,        # 0 / 90 / 180 / 270
    "random_crop_scale": (0.75, 1.0),  # crop between 75-100% then resize back
    "brightness_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15),
    "gaussian_blur_prob": 0.15,
    "gaussian_blur_radius": (0.5, 1.5),
    "gaussian_noise_prob": 0.15,
    "gaussian_noise_std": 0.02,
}


def apply_aerial_augmentations(img_arr, mask_arr, cfg=None):
    """
    Apply realistic aerial-image augmentations to a paired (image, mask).
    Spatial transforms are applied identically to both.
    Photometric transforms are applied only to the image.

    Args:
        img_arr: np.ndarray (H, W, 3) float32 in [0, 1]
        mask_arr: np.ndarray (H, W) float32 binary {0, 1}
        cfg: augmentation config dict

    Returns:
        img_arr, mask_arr (augmented)
    """
    if cfg is None:
        cfg = AUGMENTATION_CONFIG

    h, w = img_arr.shape[:2]

    # ── Spatial transforms (applied to BOTH image and mask) ──────────
    # Horizontal flip
    if cfg.get("horizontal_flip") and random.random() > 0.5:
        img_arr = np.fliplr(img_arr).copy()
        mask_arr = np.fliplr(mask_arr).copy()

    # Vertical flip
    if cfg.get("vertical_flip") and random.random() > 0.5:
        img_arr = np.flipud(img_arr).copy()
        mask_arr = np.flipud(mask_arr).copy()

    # 90-degree rotation
    if cfg.get("rotate_90"):
        k = random.randint(0, 3)
        if k > 0:
            img_arr = np.rot90(img_arr, k).copy()
            mask_arr = np.rot90(mask_arr, k).copy()

    # Random scale/crop → resize back
    crop_range = cfg.get("random_crop_scale", (1.0, 1.0))
    if crop_range[0] < 1.0 and random.random() > 0.4:
        scale = random.uniform(crop_range[0], crop_range[1])
        ch, cw = int(h * scale), int(w * scale)
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img_arr = cv2.resize(img_arr[top:top+ch, left:left+cw], (w, h),
                             interpolation=cv2.INTER_LINEAR)
        mask_arr = cv2.resize(mask_arr[top:top+ch, left:left+cw], (w, h),
                              interpolation=cv2.INTER_NEAREST)

    # ── Photometric transforms (applied ONLY to image) ───────────────
    # Brightness
    br = cfg.get("brightness_range", (1.0, 1.0))
    if br[0] != br[1]:
        factor = random.uniform(br[0], br[1])
        img_arr = np.clip(img_arr * factor, 0.0, 1.0)

    # Contrast
    cr = cfg.get("contrast_range", (1.0, 1.0))
    if cr[0] != cr[1]:
        factor = random.uniform(cr[0], cr[1])
        mean_val = img_arr.mean()
        img_arr = np.clip((img_arr - mean_val) * factor + mean_val, 0.0, 1.0)

    # Mild Gaussian blur
    if cfg.get("gaussian_blur_prob", 0) > 0 and random.random() < cfg["gaussian_blur_prob"]:
        radius = random.uniform(*cfg["gaussian_blur_radius"])
        # Use PIL for Gaussian blur
        pil_img = Image.fromarray((img_arr * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0

    # Mild Gaussian noise
    if cfg.get("gaussian_noise_prob", 0) > 0 and random.random() < cfg["gaussian_noise_prob"]:
        noise_std = cfg["gaussian_noise_std"]
        noise = np.random.normal(0, noise_std, img_arr.shape).astype(np.float32)
        img_arr = np.clip(img_arr + noise, 0.0, 1.0)

    return img_arr, mask_arr


class BuildingDataset(Dataset):
    """
    PyTorch Dataset for Aerial Building Segmentation.
    Loads paired aerial images and binary building masks (0: background, 1: building).
    Supports realistic aerial augmentations during training (G).
    """
    def __init__(self, image_paths, mask_paths, target_size=(256, 256), is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load RGB image
        img = Image.open(img_path).convert("RGB")
        # Load single-channel mask (0 or 1)
        mask = Image.open(mask_path).convert("L")

        # Resize if not already target size
        if img.size != self.target_size:
            img = img.resize(self.target_size, resample=Image.Resampling.BILINEAR)
            mask = mask.resize(self.target_size, resample=Image.Resampling.NEAREST)

        # Convert to numpy float [0, 1]
        img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        mask_arr = np.array(mask, dtype=np.float32)
        mask_arr = (mask_arr > 0.5).astype(np.float32)     # binary {0, 1}

        # Apply realistic augmentations during training (G)
        if self.is_train:
            img_arr, mask_arr = apply_aerial_augmentations(img_arr, mask_arr)

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std

        # Permute to (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()  # (1, H, W)

        return img_tensor, mask_tensor, os.path.basename(img_path)

def rasterize_coco_polygons(coco_data, raw_dir, output_dir, target_size=(256, 256)):
    """
    Rasterize vector polygons from COCO format to binary masks (0 = background, 1 = building).
    Saves aligned processed images and masks, plus train/val/test splits.
    """
    img_by_id = {img['id']: img for img in coco_data['images']}
    anns_by_img = {}
    for a in coco_data['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)

    # Group by region
    regions = {}
    for img in coco_data['images']:
        name = img['file_name']
        prefix = name.split('-')[0].split('_')[0]
        regions.setdefault(prefix, []).append(img)

    np.random.seed(42)
    for r in regions:
        np.random.shuffle(regions[r])

    # Geographic split: Kherson + Kharkiv for train, Mykolaiv for val, Tripoli for test
    train_candidates = [img for img in regions.get('Kherson', [])] + [img for img in regions.get('Kharkiv', [])]
    val_candidates = [img for img in regions.get('Mykolaiv', [])]
    test_candidates = [img for img in regions.get('Tripoli', [])]

    # Select samples with building annotations
    train_selected = [img for img in train_candidates if len(anns_by_img.get(img['id'], [])) > 0][:160]
    val_selected = [img for img in val_candidates if len(anns_by_img.get(img['id'], [])) > 0][:35]
    test_selected = [img for img in test_candidates if len(anns_by_img.get(img['id'], [])) > 0][:35]

    splits = {
        'train': train_selected,
        'val': val_selected,
        'test': test_selected
    }

    processed_images_dir = os.path.join(output_dir, 'images')
    processed_masks_dir = os.path.join(output_dir, 'masks')
    splits_dir = os.path.join(output_dir, 'splits')

    for s in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_images_dir, s), exist_ok=True)
        os.makedirs(os.path.join(processed_masks_dir, s), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    summary = {}

    for split_name, img_list in splits.items():
        split_records = []
        for img_meta in img_list:
            src_img_path = os.path.join(raw_dir, img_meta['file_name'])
            if not os.path.exists(src_img_path):
                continue

            orig_w, orig_h = img_meta['width'], img_meta['height']
            # Create full-res binary mask
            mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
            anns = anns_by_img.get(img_meta['id'], [])
            for ann in anns:
                for seg in ann['segmentation']:
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask_orig, [pts], 1)

            # Load original image and resize to target_size
            with Image.open(src_img_path) as im:
                im_rgb = im.convert("RGB").resize(target_size, resample=Image.Resampling.BILINEAR)

            mask_resized = cv2.resize(mask_orig, target_size, interpolation=cv2.INTER_NEAREST)

            base_name = f"{split_name}_{img_meta['id']:04d}_{img_meta['file_name']}"
            dst_img_path = os.path.join(processed_images_dir, split_name, base_name)
            dst_mask_path = os.path.join(processed_masks_dir, split_name, base_name.replace('.jpg', '.png'))

            im_rgb.save(dst_img_path)
            # Save mask with 0 and 255 for standard PNG visualization / inspection
            mask_png = Image.fromarray((mask_resized * 255).astype(np.uint8))
            mask_png.save(dst_mask_path)

            split_records.append(f"{dst_img_path},{dst_mask_path}\n")

        with open(os.path.join(splits_dir, f"{split_name}.txt"), "w") as f_split:
            f_split.writelines(split_records)
        summary[split_name] = len(split_records)

    return summary

def get_dataloaders(processed_dir, batch_size=8, target_size=(256, 256), num_workers=0):
    splits_dir = os.path.join(processed_dir, 'splits')
    loaders = {}

    for split in ['train', 'val', 'test']:
        split_file = os.path.join(splits_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found. Run dataset preparation first.")

        with open(split_file, 'r') as f:
            lines = [line.strip().split(',') for line in f.readlines() if line.strip()]

        img_paths = [l[0] for l in lines]
        mask_paths = [l[1] for l in lines]

        is_train = (split == 'train')
        ds = BuildingDataset(img_paths, mask_paths, target_size=target_size, is_train=is_train)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False
        )

    return loaders


def rasterize_spacenet7_dataset(
    dataset_root="sample/L15-0506E-1204N_2027_3374_13",
    output_dir="data/processed_spacenet7",
    patch_size=256,
    stride=256
):
    """
    Rasterize SpaceNet 7 multi-temporal imagery and building polygons into
    image-mask patches with temporal splits (compatible with get_dataloaders).
    """
    from src.prepare_spacenet7 import prepare_spacenet7_dataset
    return prepare_spacenet7_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride
    )


class Building3ClassDataset(Dataset):
    """
    PyTorch Dataset for 3-Class Semantic Segmentation:
      Class 0: Background
      Class 1: Building Interior
      Class 2: Building Boundary

    Loads paired aerial images (RGB) and 3-class target masks (uint8, values in {0, 1, 2}).
    Masks remain pure integer IDs — NO normalization to [0, 1] or conversion to 255.
    Returns:
      img_tensor: float32 (3, H, W) normalized with ImageNet stats
      mask_tensor: long (H, W) with exact integer class IDs in {0, 1, 2}
      patch_name: filename string
    """
    def __init__(self, image_paths, mask_paths, target_size=(256, 256), is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load RGB image
        img = Image.open(img_path).convert("RGB")
        # Load 3-class integer mask (uint8 with values 0, 1, 2)
        mask = Image.open(mask_path)

        if img.size != self.target_size:
            img = img.resize(self.target_size, resample=Image.Resampling.BILINEAR)
            mask = mask.resize(self.target_size, resample=Image.Resampling.NEAREST)

        img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        mask_arr = np.array(mask, dtype=np.float32)        # (H, W) values in {0, 1, 2}

        # Apply aerial augmentations during training (spatial interpolation uses INTER_NEAREST)
        if self.is_train:
            img_arr, mask_arr = apply_aerial_augmentations(img_arr, mask_arr)

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std

        # Permute to (C, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        # Ensure mask is int64 (long) with values in {0, 1, 2}
        mask_tensor = torch.from_numpy(np.round(mask_arr)).long()  # (H, W)

        return img_tensor, mask_tensor, os.path.basename(img_path)


def get_3class_dataloaders(processed_dir="data/processed_spacenet7_3class", batch_size=8, target_size=(256, 256), num_workers=0):
    """
    Construct DataLoaders for 3-class SpaceNet 7 dataset.
    Reads splits/train.txt, splits/val.txt, splits/test.txt.
    """
    splits_dir = os.path.join(processed_dir, "splits")
    loaders = {}

    for split in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found.")

        with open(split_file, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",") for line in f.readlines() if line.strip()]

        img_paths = [l[0] for l in lines]
        mask_paths = [l[1] for l in lines]

        is_train = (split == "train")
        ds = Building3ClassDataset(img_paths, mask_paths, target_size=target_size, is_train=is_train)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False
        )

    return loaders


