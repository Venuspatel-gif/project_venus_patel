# =============================================================================
# predict.py — Inference for EuroSAT Land Cover Classification
# Author: Venus Patel | Roll No: 20231173
# =============================================================================

import torch
from PIL import Image
from torchvision import transforms

from config import (
    CLASS_NAMES, WEIGHTS_PATH,
    resize_x, resize_y,
    IMAGENET_MEAN, IMAGENET_STD
)
from model import load_model

# ── Inference transform (no augmentation) ────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def classify_satellite_images(list_of_img_paths, weights_path=WEIGHTS_PATH,
                               device=None):
    """
    Run inference on a list of satellite image file paths.

    This function:
      1. Loads the trained model from weights_path.
      2. Opens and preprocesses each image.
      3. Runs inference and returns predicted class labels.

    Args:
        list_of_img_paths (list): List of image file paths (str).
                                  These should be paths to raw .jpg images
                                  as found in the data/ directory.
        weights_path      (str) : Path to the trained model weights (.pth).
        device                  : torch.device. Auto-detected if None.

    Returns:
        list: Predicted class label strings for each input image.

    Example:
        >>> paths = ["data/AnnualCrop_001.jpg", "data/Forest_002.jpg"]
        >>> labels = classify_satellite_images(paths)
        >>> print(labels)
        ['AnnualCrop', 'Forest']
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(weights_path=weights_path, device=device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for img_path in list_of_img_paths:
            # Load and preprocess image
            img    = Image.open(img_path).convert("RGB")
            tensor = infer_transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Forward pass
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)
            conf, pred_idx = probs.max(dim=1)

            pred_label = CLASS_NAMES[pred_idx.item()]
            confidence = conf.item()

            predictions.append(pred_label)
            print(f"  {img_path:<40} → {pred_label:<25} (conf: {confidence:.4f})")

    return predictions


def classify_batch(list_of_img_paths, weights_path=WEIGHTS_PATH,
                   batch_size=32, device=None):
    """
    Run batch inference on a list of image file paths (faster for large sets).

    Args:
        list_of_img_paths (list): List of image file paths.
        weights_path      (str) : Path to trained weights.
        batch_size        (int) : Batch size for inference.
        device                  : torch.device.

    Returns:
        list of dicts with keys: path, predicted_class, confidence.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torch.utils.data import Dataset, DataLoader

    class PathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths     = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]

    dataset = PathDataset(list_of_img_paths, infer_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_model(weights_path=weights_path, device=device)
    model.eval()

    results = []
    with torch.no_grad():
        for tensors, paths in loader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            for path, pred, conf in zip(paths, preds.cpu(), confs.cpu()):
                results.append({
                    "path":            path,
                    "predicted_class": CLASS_NAMES[pred.item()],
                    "confidence":      round(conf.item(), 4)
                })

    return results
