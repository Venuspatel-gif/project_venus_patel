# Satellite Land Cover Classification Using Convolutional Neural Networks

**Course:** Image and Video Processing with Deep Learning (DS3273)  
**Author:** Venus Patel  
**Roll No:** 20231173  
**Dataset:** EuroSAT (Sentinel-2 Satellite Imagery)  
**Model:** ResNet-50 (Fine-tuned, 2-Phase Transfer Learning)  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Image Processing Pipeline](#6-image-processing-pipeline)
7. [How to Run](#7-how-to-run)
8. [File Descriptions](#8-file-descriptions)
9. [Results](#9-results)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Outputs and Visualizations](#11-outputs-and-visualizations)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

This project builds an automated **multi-class land cover classifier** for satellite imagery. Given a 64×64 RGB satellite image captured by the Sentinel-2 satellite, the model predicts which of **10 land cover categories** the image belongs to.

### Why This Problem Matters
Manual interpretation of satellite imagery is time-consuming and requires significant domain expertise. An automated deep learning system enables large-scale, rapid land cover analysis — critical for:
- **Environmental monitoring** (deforestation, wetland changes)
- **Urban planning** (tracking residential/industrial expansion)
- **Agricultural management** (crop type identification)
- **Climate studies** (vegetation and water body tracking)

### The 10 Land Cover Classes
| Class | Description |
|-------|-------------|
| AnnualCrop | Fields with annually harvested crops |
| Forest | Dense forest cover |
| HerbaceousVegetation | Low-lying grasses and shrubs |
| Highway | Major roads and motorways |
| Industrial | Factories and industrial zones |
| Pasture | Grassland used for grazing |
| PermanentCrop | Vineyards, orchards, permanent plantations |
| Residential | Urban housing areas |
| River | Rivers and waterways |
| SeaLake | Sea, lakes, and large water bodies |

### Core Challenge
Several classes share visually similar textures and color distributions — for example, Pasture vs HerbaceousVegetation, or Residential vs Industrial. The model must learn subtle spatial patterns like texture structure, edge formations (roads, rivers), and region layouts to distinguish these classes reliably.

---

## 2. Directory Structure

```
project_venus_patel/
│
├── checkpoints/
│   └── final_weights.pth          ← Trained model weights (best validation accuracy)
│
├── data/
│   ├── AnnualCrop/                ← 10 sample images
│   ├── Forest/                    ← 10 sample images
│   ├── HerbaceousVegetation/      ← 10 sample images
│   ├── Highway/                   ← 10 sample images
│   ├── Industrial/                ← 10 sample images
│   ├── Pasture/                   ← 10 sample images
│   ├── PermanentCrop/             ← 10 sample images
│   ├── Residential/               ← 10 sample images
│   ├── River/                     ← 10 sample images
│   └── SeaLake/                   ← 10 sample images
│
├── config.py                      ← All hyperparameters and paths
├── dataset.py                     ← Custom Dataset class and DataLoader
├── model.py                       ← ResNet-50 model architecture
├── train.py                       ← Training loop (Phase 1 & Phase 2)
├── predict.py                     ← Inference on image file paths
├── interface.py                   ← Standardised interface for grading
└── README.md                      ← This file
```

---

## 3. Dataset

### EuroSAT Dataset
- **Source:** [Kaggle — apollo2506/eurosat-dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **Satellite:** Sentinel-2
- **Total Images:** ~27,000 labeled images
- **Image Size:** 64×64 pixels (Ground Sampling Distance: 10 meters/pixel)
- **Format:** RGB `.jpg` images organized in class-specific folders
- **Labels:** Fully annotated — no additional labeling required

### Dataset Split
The full dataset is split into three subsets:

| Split | Ratio | Approximate Size |
|-------|-------|-----------------|
| Train | 70%   | ~18,900 images  |
| Validation | 15% | ~4,050 images |
| Test  | 15%   | ~4,050 images  |

Splits are created using random index-based shuffling with a fixed seed (`SEED=42`) for full reproducibility. Each split gets its own appropriate transform — augmentation only on train, clean resize+normalize on val/test.

### Sample Data (`data/` directory)
The `data/` directory contains **10 raw images per class (100 total)** in their original 64×64 `.jpg` format as found in the EuroSAT dataset — not resized or preprocessed.

---

## 4. Model Architecture

### Backbone: ResNet-50
ResNet-50 is a 50-layer deep residual network pretrained on ImageNet (1.2M images, 1000 classes). It is well-suited for satellite imagery because:
- Its convolutional filters naturally capture **local spatial patterns** like texture, edges, and region structure
- Residual (skip) connections prevent vanishing gradients in deep networks
- ImageNet pretraining provides useful low-level features (edges, textures) that transfer well to satellite imagery

### Custom Classification Head
The original ResNet-50 FC layer (1000-class) is replaced with a custom head for 10-class EuroSAT classification:

```
ResNet-50 Backbone (frozen in Phase 1)
        ↓
   Global Average Pooling
        ↓
   [2048-dimensional feature vector]
        ↓
   Dropout(p=0.4)
        ↓
   Linear(2048 → 512)
        ↓
   ReLU
        ↓
   Dropout(p=0.3)
        ↓
   Linear(512 → 10)
        ↓
   Output logits (10 classes)
```

### Parameter Count
| Component | Parameters |
|-----------|-----------|
| Total (ResNet-50) | ~23.5 million |
| Trainable (Phase 1, head only) | ~1.05 million |
| Trainable (Phase 2, full model) | ~23.5 million |

---

## 5. Training Strategy

Training is done in **two phases** to avoid overfitting and make the most of the pretrained weights.

### Phase 1 — Head Training (20 Epochs)
- **Backbone:** Frozen (weights locked, no gradient updates)
- **Trainable:** Only the custom FC head (~1M parameters)
- **Purpose:** Teach the new classification head to map ResNet-50 features to EuroSAT classes without disturbing pretrained weights
- **Learning Rate:** `1e-4`
- **Scheduler:** CosineAnnealingLR (smoothly decays LR from `1e-4` to `1e-6`)

### Phase 2 — Full Fine-tuning (10 Epochs)
- **Backbone:** Unfrozen (all layers trainable)
- **Trainable:** All ~23.5M parameters
- **Purpose:** Fine-tune the entire network on satellite imagery — adapts low-level features for the specific domain
- **Learning Rate:** `1e-5` (10× lower than Phase 1 to avoid destroying pretrained weights)
- **Scheduler:** CosineAnnealingLR (decays from `1e-5` to `1e-7`)

### Why Two Phases?
If you unfreeze the entire model from the start with a high learning rate, the pretrained weights get destroyed and training is unstable. Phase 1 first adapts the head, then Phase 2 gently fine-tunes everything — this is standard transfer learning practice and consistently achieves better results.

### Loss Function
**CrossEntropyLoss with Label Smoothing (α=0.1)**  
Label smoothing prevents the model from becoming overconfident — instead of targeting a hard 1.0 probability for the correct class, it targets 0.9, which improves generalization.

### Optimizer
**AdamW** with `weight_decay=1e-4`  
AdamW decouples weight decay from the gradient update, making it more effective than standard Adam for regularization.

### Checkpoint System
A robust checkpoint system saves the model every epoch so training can resume after disconnection:
- `latest_checkpoint_phase1.pth` — saved every epoch during Phase 1
- `latest_checkpoint_phase2.pth` — saved every epoch during Phase 2
- `final_weights.pth` — the best model by validation accuracy
- Auto-resume: on re-run, the training loop detects existing checkpoints and continues from the last saved epoch

---

## 6. Image Processing Pipeline

### Training Transforms (with Augmentation)
```
Input: 64×64 raw satellite image
  ↓  Resize → 224×224  (required by ResNet-50)
  ↓  RandomHorizontalFlip (p=0.5)
  ↓  RandomVerticalFlip (p=0.5)
  ↓  RandomRotation (±30°)
  ↓  ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  ↓  ToTensor → [0,1] float tensor
  ↓  Normalize (ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
Output: [3, 224, 224] normalized tensor
```

**Why these augmentations?**
- **Flips:** Satellite images have no "correct" orientation — a forest looks the same upside down
- **Rotation:** Land cover patterns are rotationally invariant
- **ColorJitter:** Accounts for varying lighting conditions and atmospheric effects across satellite captures
- **ImageNet Normalization:** Required because the pretrained ResNet-50 was trained on ImageNet-normalized inputs

### Validation/Test Transforms (no augmentation)
```
Input: 64×64 raw satellite image
  ↓  Resize → 224×224
  ↓  ToTensor
  ↓  Normalize (ImageNet stats)
Output: [3, 224, 224] normalized tensor
```

---

## 7. How to Run

### Prerequisites
```bash
pip install torch torchvision tqdm scikit-learn matplotlib seaborn pillow pandas numpy
```

### Option A — Run on Kaggle (Recommended)
1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the EuroSAT dataset: **Add Data → search "eurosat-dataset" by apollo2506**
3. Enable GPU: **Settings → Accelerator → GPU T4**
4. Paste and run all cells from the notebook in order

### Option B — Run Locally

**Step 1: Download the EuroSAT dataset**
```
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
```
Extract it so the folder structure is:
```
EuroSAT/
  AnnualCrop/
  Forest/
  ...
```

**Step 2: Update DATA_DIR in config.py**
```python
DATA_DIR = "path/to/your/EuroSAT/"
```

**Step 3: Train the model**
```python
from config import *
from dataset import eurosat_dataloader
from model import EuroSATResNet
from train import run_full_training
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, class_names = eurosat_dataloader()
model = EuroSATResNet(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)

history = run_full_training(model, train_loader, val_loader, device)
```

**Step 4: Run inference on new images**
```python
from predict import classify_satellite_images

image_paths = [
    "data/AnnualCrop/AnnualCrop_00001.jpg",
    "data/Forest/Forest_00001.jpg",
    "data/River/River_00001.jpg",
]

predictions = classify_satellite_images(image_paths)
print(predictions)
# ['AnnualCrop', 'Forest', 'River']
```

**Step 5: Use the standardised interface**
```python
from interface import (
    TheModel, the_trainer, the_predictor,
    TheDataset, the_dataloader,
    the_batch_size, total_epochs
)

# Load data
train_loader, val_loader, test_loader, classes = the_dataloader()

# Build model
model = TheModel(num_classes=10, freeze_backbone=True)

# Train
import torch.nn as nn, torch.optim as optim
loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
history   = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer, val_loader)

# Predict
labels = the_predictor(["data/Forest/Forest_00001.jpg"])
print(labels)  # ['Forest']
```

---

## 8. File Descriptions

### `config.py`
Central configuration file. Contains **all hyperparameters** so you never need to change values scattered across multiple files.

| Variable | Value | Description |
|----------|-------|-------------|
| `batchsize` | 64 | Number of images per batch |
| `epochs` | 20 | Phase 1 training epochs |
| `FINETUNE_EPOCHS` | 10 | Phase 2 fine-tuning epochs |
| `LR` | 1e-4 | Phase 1 learning rate |
| `LR_FINETUNE` | 1e-5 | Phase 2 learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `resize_x` | 224 | Input image width |
| `resize_y` | 224 | Input image height |
| `input_channels` | 3 | RGB channels |
| `TRAIN_RATIO` | 0.70 | Training split fraction |
| `VAL_RATIO` | 0.15 | Validation split fraction |
| `SEED` | 42 | Random seed for reproducibility |

### `dataset.py`
- **`EuroSATDataset`** — Custom Dataset class inheriting from `torchvision.datasets.ImageFolder`. Handles image loading and transform application.
- **`eurosat_dataloader()`** — Returns `(train_loader, val_loader, test_loader, class_names)`. Handles 70/15/15 split internally using index-based shuffling.
- **`eurosat_infer_loader()`** — Creates a DataLoader from a list of raw image file paths for inference.

### `model.py`
- **`EuroSATResNet`** — Main model class. Loads pretrained ResNet-50 and replaces the FC head. Supports `freeze_backbone=True/False` to control Phase 1 vs Phase 2.
- **`unfreeze_backbone()`** — Method to unfreeze all layers for Phase 2 fine-tuning.
- **`load_model()`** — Utility to load trained weights from a `.pth` file.

### `train.py`
- **`train_model()`** — Core training loop. Accepts model, epochs, loaders, loss function, and optimizer. Saves best model checkpoint automatically. Matches the professor's required function signature exactly.
- **`run_full_training()`** — Convenience wrapper that runs both Phase 1 and Phase 2 sequentially.

### `predict.py`
- **`classify_satellite_images(list_of_img_paths)`** — Takes a list of image file paths, loads the trained model, and returns predicted class label strings. Works directly with paths from the `data/` directory.
- **`classify_batch()`** — Faster batch inference with confidence scores for each prediction.

### `interface.py`
Standardised interface mapping internal names to the grading program's expected names:
```python
TheModel      ← EuroSATResNet
the_trainer   ← train_model
the_predictor ← classify_satellite_images
TheDataset    ← EuroSATDataset
the_dataloader← eurosat_dataloader
the_batch_size← batchsize (64)
total_epochs  ← epochs (20)
```

---

## 9. Results

### Training Performance
| Phase | Epochs | Best Val Accuracy |
|-------|--------|-------------------|
| Phase 1 (head only) | 1–20 | ~88–91% |
| Phase 2 (full fine-tune) | 21–30 | ~93–96% |

### Test Set Performance
| Metric | Score |
|--------|-------|
| Test Accuracy | ~95%+ |
| Weighted F1-Score | ~0.95+ |

### Per-Class Performance
Classes like **Forest**, **SeaLake**, and **Residential** are typically classified with very high accuracy (F1 > 0.97) because they have highly distinctive visual patterns.

Classes like **HerbaceousVegetation** vs **Pasture** and **AnnualCrop** vs **PermanentCrop** are harder to distinguish due to similar color and texture distributions — these typically show slightly lower F1 scores (~0.88–0.92).

---

## 10. Evaluation Metrics

### Accuracy
Overall fraction of correctly classified images across all 10 classes.

### Weighted F1-Score
F1 is the harmonic mean of Precision and Recall. The weighted version accounts for class imbalance by weighting each class by its support (number of samples). This is the primary metric for this project.

### Confusion Matrix
A 10×10 matrix showing how often each true class was predicted as each other class. The diagonal shows correct predictions. Off-diagonal entries reveal common misclassifications — e.g., HerbaceousVegetation being confused with Pasture.

### Per-Class F1 Bar Chart
Bar chart showing individual F1 scores for each of the 10 classes, colour-coded:
- 🟢 Green: F1 ≥ 0.90 (excellent)
- 🟡 Orange: 0.80 ≤ F1 < 0.90 (good)
- 🔴 Red: F1 < 0.80 (needs improvement)

---

## 11. Outputs and Visualizations

All outputs are saved to `/kaggle/working/` and can be downloaded from the Kaggle Output tab:

| File | Description |
|------|-------------|
| `best_resnet50_eurosat.pth` | Best model weights by validation accuracy |
| `training_history.pth` | Full training history dict (loss & accuracy per epoch) |
| `training_curves.png` | Loss and accuracy curves for all 30 epochs (both phases) |
| `confusion_matrix.png` | Raw count + normalised confusion matrices side by side |
| `per_class_f1.png` | Per-class F1-Score bar chart colour-coded by performance |
| `predictions_grid.png` | 4×4 grid of test images with true vs predicted labels |
| `predictions.csv` | Full test set predictions with confidence scores |
| `sample_images.png` | One sample image per class (10 total) |
| `class_distribution.png` | Bar chart of image count per class |
| `samples_per_class.png` | 10×10 grid — 10 samples per class |

---

## 12. Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning framework |
| `torchvision` | ≥0.15 | Pretrained models and transforms |
| `numpy` | ≥1.23 | Numerical operations |
| `pandas` | ≥1.5 | CSV handling for predictions |
| `matplotlib` | ≥3.6 | Plotting training curves and grids |
| `seaborn` | ≥0.12 | Confusion matrix heatmaps |
| `scikit-learn` | ≥1.1 | Metrics (F1, accuracy, confusion matrix) |
| `Pillow` | ≥9.0 | Image loading and conversion |
| `tqdm` | ≥4.64 | Training progress bars |

Install all dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

---

## References

1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). **EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** CVPR 2016.
3. EuroSAT Dataset on Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
