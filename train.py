# =============================================================================
# train.py — Training Loop for EuroSAT Land Cover Classification
# Author: Venus Patel | Roll No: 20231173
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    batchsize, epochs, FINETUNE_EPOCHS,
    LR, LR_FINETUNE, WEIGHT_DECAY,
    CHECKPOINT_DIR, WEIGHTS_PATH, NUM_CLASSES
)


def train_model(
    model,
    num_epochs,
    train_loader,
    loss_fn,
    optimizer,
    val_loader=None,
    scheduler=None,
    device=None,
    checkpoint_dir=CHECKPOINT_DIR,
    phase=1
):
    """
    Runs the training loop for one phase.

    Args:
        model        : The neural network model.
        num_epochs   (int): Number of epochs to train.
        train_loader : DataLoader for training data.
        loss_fn      : Loss function (e.g. CrossEntropyLoss).
        optimizer    : Optimizer (e.g. AdamW).
        val_loader   : DataLoader for validation data (optional).
        scheduler    : LR scheduler (optional).
        device       : torch.device to train on.
        checkpoint_dir(str): Directory to save checkpoints.
        phase        (int): Training phase (1=frozen, 2=fine-tune).

    Returns:
        dict: Training history with keys train_loss, train_acc, val_loss, val_acc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc   = 0.0
    history        = {"train_loss": [], "train_acc": [],
                      "val_loss":   [], "val_acc":   []}

    print(f"\n{'='*65}")
    print(f"  Phase {phase} Training — {num_epochs} epochs on {device}")
    print(f"{'='*65}")
    print(f"{'Epoch':>5} | {'T-Loss':>7} | {'T-Acc':>6} | {'V-Loss':>7} | {'V-Acc':>6}")
    print(f"{'='*65}")

    for epoch in range(1, num_epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)
            correct      += (outputs.argmax(dim=1) == labels).sum().item()
            total        += batch.size(0)

        t_loss = running_loss / total
        t_acc  = correct / total
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)

        # ── Validate ──────────────────────────────────────────────────────────
        v_loss, v_acc = 0.0, 0.0
        if val_loader is not None:
            model.eval()
            v_running, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for batch, labels in val_loader:
                    batch, labels = batch.to(device), labels.to(device)
                    outputs = model(batch)
                    loss    = loss_fn(outputs, labels)
                    v_running += loss.item() * batch.size(0)
                    v_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    v_total   += batch.size(0)
            v_loss = v_running / v_total
            v_acc  = v_correct / v_total

        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if scheduler is not None:
            scheduler.step()

        print(f"{epoch:>5} | {t_loss:>7.4f} | {t_acc:>5.2%} | {v_loss:>7.4f} | {v_acc:>5.2%}")

        # ── Save best model ───────────────────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"         ✓ Best model saved (val_acc={best_val_acc:.4f})")

    print(f"{'='*65}")
    print(f"✅ Phase {phase} complete. Best Val Acc: {best_val_acc:.4f}")
    return history


def run_full_training(model, train_loader, val_loader, device=None):
    """
    Runs full 2-phase training:
      Phase 1 — frozen backbone, train head only.
      Phase 2 — unfreeze backbone, fine-tune all layers.

    Args:
        model        : EuroSATResNet model.
        train_loader : Training DataLoader.
        val_loader   : Validation DataLoader.
        device       : torch.device.

    Returns:
        dict: Combined training history across both phases.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=epochs, eta_min=1e-6
    )
    history1 = train_model(model, epochs, train_loader, loss_fn,
                           optimizer1, val_loader, scheduler1, device, phase=1)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    model.unfreeze_backbone()
    optimizer2 = optim.AdamW(model.parameters(), lr=LR_FINETUNE,
                             weight_decay=WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=FINETUNE_EPOCHS, eta_min=1e-7
    )
    history2 = train_model(model, FINETUNE_EPOCHS, train_loader, loss_fn,
                           optimizer2, val_loader, scheduler2, device, phase=2)

    # Merge histories
    combined = {k: history1[k] + history2[k] for k in history1}
    return combined
