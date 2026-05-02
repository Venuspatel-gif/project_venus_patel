# =============================================================================
# dataset.py — Custom Dataset and DataLoader for EuroSAT Classification
# Author: Venus Patel | Roll No: 20231173
# =============================================================================

import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    DATA_DIR, resize_x, resize_y, input_channels,
    IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_RATIO, VAL_RATIO, SEED, batchsize
)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Custom Dataset Class ──────────────────────────────────────────────────────
class EuroSATDataset(datasets.ImageFolder):
    """
    Custom Dataset for EuroSAT land cover classification.
    Inherits from ImageFolder — expects data organized in class subfolders.

    Args:
        root (str): Path to dataset root directory.
        transform: Torchvision transforms to apply.
    """
    def __init__(self, root=DATA_DIR, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label

    def __len__(self):
        return super().__len__()


# ── DataLoader Factory ────────────────────────────────────────────────────────
def eurosat_dataloader(
    data_dir=DATA_DIR,
    batch_size=batchsize,
    num_workers=0,
    seed=SEED
):
    """
    Creates train, validation, and test DataLoaders for EuroSAT.

    Args:
        data_dir  (str): Path to EuroSAT dataset root.
        batch_size (int): Batch size for all loaders.
        num_workers(int): Number of worker processes.
        seed       (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    full_dataset = EuroSATDataset(root=data_dir, transform=train_transforms)

    total      = len(full_dataset)
    train_size = int(TRAIN_RATIO * total)
    val_size   = int(VAL_RATIO   * total)
    test_size  = total - train_size - val_size

    indices = list(range(total))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    train_dataset = Subset(EuroSATDataset(root=data_dir, transform=train_transforms),    train_indices)
    val_dataset   = Subset(EuroSATDataset(root=data_dir, transform=val_test_transforms), val_indices)
    test_dataset  = Subset(EuroSATDataset(root=data_dir, transform=val_test_transforms), test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, full_dataset.classes


# ── Convenience: single loader for inference ──────────────────────────────────
def eurosat_infer_loader(image_paths, batch_size=batchsize):
    """
    Creates a DataLoader from a list of image file paths for inference.

    Args:
        image_paths (list): List of image file paths.
        batch_size   (int): Batch size.

    Returns:
        DataLoader
    """
    from torch.utils.data import Dataset
    from PIL import Image

    class InferDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths     = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]

    infer_dataset = InferDataset(image_paths, val_test_transforms)
    return DataLoader(infer_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=0)
