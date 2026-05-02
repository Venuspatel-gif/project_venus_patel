# =============================================================================
# config.py — All hyperparameters and configuration for EuroSAT Classification
# Author: Venus Patel | Roll No: 20231173
# =============================================================================

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/"               # path to the data directory
NUM_CLASSES  = 10
CLASS_NAMES  = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# ── Image Dimensions ──────────────────────────────────────────────────────────
resize_x       = 224
resize_y       = 224
input_channels = 3

# ── Training Hyperparameters ──────────────────────────────────────────────────
batchsize      = 64
epochs         = 20
FINETUNE_EPOCHS= 10
LR             = 1e-4
LR_FINETUNE    = 1e-5
WEIGHT_DECAY   = 1e-4
SEED           = 42

# ── Dataset Splits ────────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# TEST_RATIO   = 0.15 (remainder)

# ── Normalization (ImageNet stats) ────────────────────────────────────────────
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints/"
WEIGHTS_PATH   = "checkpoints/final_weights.pth"
