# =============================================================================
# model.py — ResNet-50 Model for EuroSAT Land Cover Classification
# Author: Venus Patel | Roll No: 20231173
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, WEIGHTS_PATH


class EuroSATResNet(nn.Module):
    """
    Fine-tuned ResNet-50 for EuroSAT land cover classification.

    Architecture:
        - Backbone : ResNet-50 pretrained on ImageNet (V2 weights)
        - Head     : Dropout → Linear(2048→512) → ReLU → Dropout → Linear(512→10)

    Args:
        num_classes (int): Number of output classes. Default: 10
        freeze_backbone (bool): Whether to freeze backbone layers. Default: True
    """
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(EuroSATResNet, self).__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        # Freeze backbone if required (Phase 1 training)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final FC layer with custom classification head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✅ Backbone unfrozen for fine-tuning.")

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params     : {total:,}")
        print(f"Trainable params : {trainable:,}")
        print(f"Frozen params    : {total - trainable:,}")


def load_model(weights_path=WEIGHTS_PATH, num_classes=NUM_CLASSES, device=None):
    """
    Load a trained EuroSATResNet model from saved weights.

    Args:
        weights_path (str): Path to the .pth weights file.
        num_classes   (int): Number of output classes.
        device       : torch.device to load model onto.

    Returns:
        EuroSATResNet: Model with loaded weights in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EuroSATResNet(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {weights_path}")
    return model
