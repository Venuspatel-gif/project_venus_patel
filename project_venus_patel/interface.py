# =============================================================================
# interface.py — Standardised Interface for EuroSAT Classification Project
# Author: Venus Patel | Roll No: 20231173
# =============================================================================
# This file standardises all function and class names for the grading program.
# Do NOT rename anything in this file.
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────────────
# EuroSATResNet is our custom ResNet-50 based classification model
from model import EuroSATResNet as TheModel

# ── Training ──────────────────────────────────────────────────────────────────
# train_model is the function that runs the training loop
from train import train_model as the_trainer

# ── Prediction ────────────────────────────────────────────────────────────────
# classify_satellite_images takes a list of image paths and returns predictions
from predict import classify_satellite_images as the_predictor

# ── Dataset ───────────────────────────────────────────────────────────────────
# EuroSATDataset is our custom Dataset class (inherits from ImageFolder)
from dataset import EuroSATDataset as TheDataset

# ── DataLoader ────────────────────────────────────────────────────────────────
# eurosat_dataloader returns (train_loader, val_loader, test_loader, class_names)
from dataset import eurosat_dataloader as the_dataloader

# ── Config ────────────────────────────────────────────────────────────────────
from config import batchsize as the_batch_size
from config import epochs    as total_epochs
