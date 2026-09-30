import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    """
    Lightweight, fast baseline U-Net architecture for aerial building segmentation.
    Designed for rapid convergence and efficient CPU/GPU computation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution producing raw logits
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=1, pretrained=True):
    """
    Factory to construct U-Net model. Uses segmentation_models_pytorch if available,
    falling back to clean native SimpleUNet.
    """
    if HAS_SMP:
        try:
            weights = "imagenet" if pretrained else None
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=classes
            )
            return model
        except Exception as e:
            print(f"Warning: Failed loading pretrained SMP model ({e}). Using native SimpleUNet.")
    return SimpleUNet(in_channels=in_channels, out_channels=classes)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss + DiceLoss for binary building footprint segmentation.
    Balancing binary cross-entropy (pixel-level accuracy) and Dice (overlap metric).
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class MulticlassFocalDiceLoss(nn.Module):
    """
    Combined Multiclass Focal Loss + Multiclass Dice Loss for 3-class building segmentation:
      Class 0: Background
      Class 1: Building Interior
      Class 2: Building Boundary

    Weights are applied once per loss component (Focal and Dice) across class dimensions,
    preventing any accidental double-weighting.
    """
    def __init__(self, class_weights=(1.0, 4.0, 3.0), gamma=2.0, focal_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K, H, W) raw unnormalized logits
            targets: (B, H, W) integer ground-truth class IDs in {0, ..., K-1}
        Returns:
            total_loss, focal_loss, dice_loss
        """
        B, K, H, W = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # ── 1. Multiclass Focal Loss ──
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B, H, W)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        focal_factor = (1.0 - target_probs) ** self.gamma
        w_expanded = self.weights[targets]                                       # (B, H, W)

        pixel_loss = - w_expanded * focal_factor * target_log_probs
        loss_focal = pixel_loss.sum() / (w_expanded.sum() + 1e-6)

        # ── 2. Multiclass Dice Loss ──
        targets_onehot = F.one_hot(targets, num_classes=K).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss_per_class = 1.0 - dice_per_class
        loss_dice = torch.sum(self.weights * dice_loss_per_class) / torch.sum(self.weights)

        total_loss = self.focal_weight * loss_focal + self.dice_weight * loss_dice
        return total_loss, loss_focal, loss_dice

