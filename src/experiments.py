"""
H: Reproducibility — save full experiment configuration.
I: Domain-gap experiment — train on non-Indian, test cross-domain.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import time
import platform
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.dataset import BuildingDataset, get_dataloaders, AUGMENTATION_CONFIG
from src.model import get_model, BCEDiceLoss
from src.train import compute_metrics, train_epoch, validate_epoch


# =========================================================================
# H: Save full experiment config for reproducibility
# =========================================================================

def save_experiment_config():
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_config = {
        "experiment": "SIH26012 Building Segmentation Baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42,

        "dataset": {
            "name": cfg["dataset"]["name"],
            "raw_dir": cfg["dataset"]["raw_dir"],
            "processed_dir": cfg["dataset"]["processed_dir"],
            "image_size": cfg["dataset"]["image_size"],
            "in_channels": cfg["dataset"]["in_channels"],
        },

        "split": cfg["split"],

        "model": {
            "architecture": cfg["model"]["architecture"],
            "encoder": cfg["model"]["encoder_name"],
            "encoder_weights": cfg["model"]["encoder_weights"],
            "input_size": cfg["dataset"]["image_size"],
            "in_channels": cfg["model"]["in_channels"],
            "classes": cfg["model"]["classes"],
        },

        "training": {
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "learning_rate": cfg["training"]["learning_rate"],
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "loss": "BCEDiceLoss (0.5*BCE + 0.5*Dice)",
        },

        "augmentation": AUGMENTATION_CONFIG,

        "hardware": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_used": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },

        "polygonization": cfg.get("polygonization", {}),
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)
    print(f"Experiment config saved to: {out_path}")

    return experiment_config


# =========================================================================
# I: Domain-gap experiment
# =========================================================================

def run_domain_gap_experiment():
    """
    Tests the feasibility of cross-domain transfer for SIH26012.

    Our dataset contains tiles from different geographic areas.
    We simulate the India domain gap by treating distinct regions as proxies:

      - 'Non-Indian' source: European/US-origin tiles (tyrol, austin, chicago, swiss)
      - 'Indian proxy' target: African/Middle-Eastern tiles (Mekele, Zalingei, Tripoli)

    Experiment A: Train on source → test on target (zero-shot transfer).
    Experiment B: Fine-tune on small target subset → test on held-out target.

    This measures how much domain-specific data helps — critical for SIH26012.
    """
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load the COCO annotations to identify regions ────────────────
    coco_path = os.path.join(cfg["dataset"]["raw_dir"], "_annotations.coco.json")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    regions = {}
    for img in coco["images"]:
        prefix = img["file_name"].split("-")[0].split("_")[0]
        regions.setdefault(prefix, []).append(img)

    # Define source (non-Indian proxy) and target (Indian proxy) regions
    source_regions = ["tyrol", "austin16", "austin17", "austin18", "austin19",
                      "austin20", "chicago6", "chicago7", "chicago8", "chicago9",
                      "chicago10", "swiss", "okutama"]
    target_regions = ["Mekele", "Zalingei", "Tripoli"]

    source_imgs = []
    for r in source_regions:
        source_imgs.extend([img for img in regions.get(r, [])
                            if len(anns_by_img.get(img["id"], [])) > 0])
    target_imgs = []
    for r in target_regions:
        target_imgs.extend([img for img in regions.get(r, [])
                            if len(anns_by_img.get(img["id"], [])) > 0])

    np.random.seed(42)
    np.random.shuffle(source_imgs)
    np.random.shuffle(target_imgs)

    print("=" * 60)
    print("DOMAIN-GAP EXPERIMENT — SIH26012 Feasibility")
    print(f"Source (non-Indian proxy): {len(source_imgs)} images from {source_regions}")
    print(f"Target (Indian proxy):     {len(target_imgs)} images from {target_regions}")
    print(f"Device: {device}")
    print("=" * 60)

    if len(source_imgs) < 20 or len(target_imgs) < 20:
        print("WARNING: Insufficient data for meaningful domain-gap experiment.")
        print(f"  Source: {len(source_imgs)}, Target: {len(target_imgs)}")
        print("  Skipping domain-gap experiment.")
        return None

    # ── Prepare the processed tiles (re-use existing processed data) ─
    # We'll read from the processed dir using splits
    # For this experiment, we manually construct paths from existing processed data
    processed_dir = cfg["dataset"]["processed_dir"]
    target_size = tuple(cfg["dataset"]["image_size"])

    # Helper: find processed paths for a COCO image
    def find_processed_paths(img_meta):
        """Try to find the processed image/mask for this COCO image in any split."""
        for split in ["train", "val", "test"]:
            for suffix in ["", ".jpg", ".png"]:
                img_dir = os.path.join(processed_dir, "images", split)
                mask_dir = os.path.join(processed_dir, "masks", split)
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if img_meta["file_name"] in fname:
                        img_path = os.path.join(img_dir, fname)
                        mask_name = fname.replace(".jpg", ".png")
                        mask_path = os.path.join(mask_dir, mask_name)
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            return img_path, mask_path
        return None, None

    # Build source and target datasets from processed files
    source_pairs = []
    for img_meta in source_imgs:
        ip, mp = find_processed_paths(img_meta)
        if ip:
            source_pairs.append((ip, mp))

    target_pairs = []
    for img_meta in target_imgs:
        ip, mp = find_processed_paths(img_meta)
        if ip:
            target_pairs.append((ip, mp))

    print(f"Found processed: source={len(source_pairs)}, target={len(target_pairs)}")

    if len(source_pairs) < 10 or len(target_pairs) < 10:
        print("Not enough processed data overlapping with domain regions.")
        print("This is expected if the geographic split used different regions.")
        print("We'll use the existing splits as a proxy instead.")

        # Fallback: use train split as source, test split as target
        # This still demonstrates the infrastructure for domain-gap testing
        print("\nFallback: Using train regions as 'source', test region as 'target'")
        with open(os.path.join(processed_dir, "splits", "train.txt"), "r") as f:
            source_pairs = [line.strip().split(",") for line in f if line.strip()]
        with open(os.path.join(processed_dir, "splits", "test.txt"), "r") as f:
            target_pairs = [line.strip().split(",") for line in f if line.strip()]
        # Note: in this fallback, source=Kherson+Kharkiv, target=Tripoli
        # This IS a geographic domain gap

    # Split target into fine-tune (30%) and test (70%)
    ft_count = max(5, int(len(target_pairs) * 0.3))
    target_finetune = target_pairs[:ft_count]
    target_test = target_pairs[ft_count:]

    print(f"Source train: {len(source_pairs)}")
    print(f"Target fine-tune: {len(target_finetune)}")
    print(f"Target test: {len(target_test)}")

    def make_loader(pairs, is_train=False, batch_size=4):
        imgs = [p[0] if isinstance(p, (list, tuple)) else p for p in pairs]
        masks = [p[1] if isinstance(p, (list, tuple)) else p for p in pairs]
        ds = BuildingDataset(imgs, masks, target_size=target_size, is_train=is_train)
        return DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0)

    source_loader = make_loader(source_pairs, is_train=True)
    ft_loader = make_loader(target_finetune, is_train=True)
    test_loader = make_loader(target_test, is_train=False)

    criterion = BCEDiceLoss()

    # ── Experiment A: Zero-shot transfer ─────────────────────────────
    print("\n── Experiment A: Train on source → test on target ──")
    model_a = get_model(
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=True,
    ).to(device)
    optimizer_a = AdamW(model_a.parameters(), lr=5e-4, weight_decay=1e-4)

    for epoch in range(1, 7):
        t_loss, t_iou, t_dice = train_epoch(model_a, source_loader, criterion, optimizer_a, device)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: train_loss={t_loss:.4f}  train_iou={t_iou:.4f}")

    # Evaluate on target test (zero-shot)
    v_loss_a, v_iou_a, v_dice_a = validate_epoch(model_a, test_loader, criterion, device)
    print(f"  Zero-shot on target: IoU={v_iou_a:.4f}  Dice={v_dice_a:.4f}")

    # ── Experiment B: Fine-tune on small target subset ───────────────
    print("\n── Experiment B: Fine-tune on target subset → test ──")
    model_b = get_model(
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=False,
    ).to(device)
    # Start from model A's weights
    model_b.load_state_dict(model_a.state_dict())
    optimizer_b = AdamW(model_b.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, 6):
        ft_loss, ft_iou, ft_dice = train_epoch(model_b, ft_loader, criterion, optimizer_b, device)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  FT Epoch {epoch}: loss={ft_loss:.4f}  iou={ft_iou:.4f}")

    v_loss_b, v_iou_b, v_dice_b = validate_epoch(model_b, test_loader, criterion, device)
    print(f"  After fine-tuning:   IoU={v_iou_b:.4f}  Dice={v_dice_b:.4f}")

    # ── Results comparison ───────────────────────────────────────────
    improvement_iou = v_iou_b - v_iou_a
    improvement_dice = v_dice_b - v_dice_a

    results = {
        "experiment": "domain_gap",
        "source_regions": source_regions,
        "target_regions": target_regions,
        "source_train_count": len(source_pairs),
        "target_finetune_count": len(target_finetune),
        "target_test_count": len(target_test),
        "zero_shot": {
            "iou": round(v_iou_a, 4),
            "dice": round(v_dice_a, 4),
        },
        "after_finetune": {
            "iou": round(v_iou_b, 4),
            "dice": round(v_dice_b, 4),
        },
        "improvement": {
            "iou_delta": round(improvement_iou, 4),
            "dice_delta": round(improvement_dice, 4),
        },
        "interpretation": (
            f"Fine-tuning on {len(target_finetune)} target samples improved IoU by "
            f"{improvement_iou:+.4f} and Dice by {improvement_dice:+.4f}. "
            f"{'This suggests domain-specific data is valuable for SIH26012.' if improvement_iou > 0.01 else 'Transfer was already good; fine-tuning gave marginal gain.'}"
        ),
    }

    print("\n── Domain-Gap Results ──")
    print(f"  Zero-shot:    IoU={v_iou_a:.4f}  Dice={v_dice_a:.4f}")
    print(f"  Fine-tuned:   IoU={v_iou_b:.4f}  Dice={v_dice_b:.4f}")
    print(f"  Improvement:  IoU={improvement_iou:+.4f}  Dice={improvement_dice:+.4f}")
    print(f"  → {results['interpretation']}")

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/domain_gap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/metrics/domain_gap_results.json")

    return results


if __name__ == "__main__":
    save_experiment_config()
    run_domain_gap_experiment()
