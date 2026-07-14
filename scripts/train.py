"""
scripts/train.py
================
Main training entry point for HyperVision KYC AI.

Usage (from project root, activate the venv first):
    .venv\\Scripts\\python.exe scripts/train.py

Key design decisions
--------------------
* AdamW optimizer    : weight decay acts as L2 regularisation on the weights,
                       which generalises better than plain Adam for vision tasks.
* CosineAnnealingLR  : smoothly anneals the learning rate from LR_MAX down to
                       LR_MIN following a cosine curve.  This avoids the sharp
                       LR drops of StepLR that can cause the loss to spike.
* BCE + Dice loss    : BCE gives a per-pixel signal while Dice drives the
                       overlap metric directly.  Using both together empirically
                       outperforms either loss alone on imbalanced masks.
* Gradient clipping  : caps the L2 norm of all gradients at 1.0 to prevent
                       exploding gradients in early training.
* AMP (mixed precision): torch.cuda.amp halves GPU memory usage and runs
                       ~1.5-2x faster on Ampere-class GPUs (RTX 4050 included)
                       with no loss in accuracy.
"""

import sys
import os

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Make project root importable regardless of CWD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
from torch.utils.data import DataLoader, random_split
import mlflow

from src.config.settings import (
    TRAIN_CSV, DEVICE, CKPT_ROOT,
)
from src.data.annotations  import load_csv_records
from src.data.dataset       import CsvSegDataset
from src.data.transforms    import get_train_transforms, get_val_transforms
from src.models.unet        import UNet
from src.training.trainer   import run_epoch
from src.training.checkpoint import save_checkpoint

# ---------------------------------------------------------------------------
# Hyper-parameters  (edit here or extend with argparse / YAML later)
# ---------------------------------------------------------------------------
EPOCHS        = 50
BATCH_SIZE    = 8           # safe for 6 GB VRAM; raise to 16 if GPU allows
LR_MAX        = 3e-4        # AdamW peak learning rate
LR_MIN        = 1e-6        # cosine annealing floor
WEIGHT_DECAY  = 1e-4        # L2 regularisation coefficient
VAL_FRACTION  = 0.2         # 20 % of training data for validation
NUM_WORKERS   = 0           # Windows: use 0 to avoid multiprocessing issues
SEED          = 42

os.makedirs(CKPT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Dataset  →  80/20 split
# ---------------------------------------------------------------------------
print("[Data] Loading CSV records ...")
all_records = load_csv_records(TRAIN_CSV)
print(f"[Data] Total records: {len(all_records)}")

n_val   = int(len(all_records) * VAL_FRACTION)
n_train = len(all_records) - n_val

# random_split uses a generator for reproducibility
generator = torch.Generator().manual_seed(SEED)
train_records, val_records = random_split(all_records, [n_train, n_val], generator=generator)
# random_split returns Subset objects; convert indices back to plain lists
train_records = [all_records[i] for i in train_records.indices]
val_records   = [all_records[i] for i in val_records.indices]

print(f"[Data] Train: {len(train_records)} | Val: {len(val_records)}")

train_ds = CsvSegDataset(train_records, transform=get_train_transforms())
val_ds   = CsvSegDataset(val_records,   transform=get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ---------------------------------------------------------------------------
# Model, Optimizer, Scheduler, AMP scaler
# ---------------------------------------------------------------------------
model = UNet(base=32).to(DEVICE)
print(f"[Model] UNet initialized on {DEVICE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR_MIN
)

# Automatic Mixed Precision: speeds up training on CUDA with no accuracy loss
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------
mlflow.set_experiment("HyperVision-UNet-Training")

with mlflow.start_run(run_name=f"unet_base32_e{EPOCHS}_bs{BATCH_SIZE}"):

    # Log all hyper-parameters once at the start of the run
    mlflow.log_params({
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "lr_max":        LR_MAX,
        "lr_min":        LR_MIN,
        "weight_decay":  WEIGHT_DECAY,
        "val_fraction":  VAL_FRACTION,
        "img_size":      256,
        "model":         "UNet-base32",
        "optimizer":     "AdamW",
        "scheduler":     "CosineAnnealingLR",
        "loss":          "0.5*BCE + 0.5*Dice",
        "device":        DEVICE,
    })

    best_val_dice = 0.0

    print(f"\n[Train] Starting training for {EPOCHS} epochs on {DEVICE} ...\n")

    for epoch in range(1, EPOCHS + 1):

        # ---- Training pass ----
        train_loss, train_dice = run_epoch(
            model, train_loader, optimizer, DEVICE, is_train=True
        )

        # ---- Validation pass ----
        val_loss, val_dice = run_epoch(
            model, val_loader, optimizer, DEVICE, is_train=False
        )

        # ---- LR scheduler step (after each epoch) ----
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- Console log ----
        print(
            f"Epoch [{epoch:03d}/{EPOCHS}]  "
            f"train_loss={train_loss:.4f}  train_dice={train_dice:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ---- MLflow: log per-epoch metrics ----
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss":   val_loss,
            "val_dice":   val_dice,
            "lr":         current_lr,
        }, step=epoch)

        # ---- Checkpoint: save whenever validation Dice improves ----
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = save_checkpoint(model, optimizer, epoch, val_dice,
                                        filename="best_model.pth")
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            print(f"  [*] New best val_dice={val_dice:.4f} — checkpoint saved.")

    print(f"\n[Done] Training complete. Best val_dice = {best_val_dice:.4f}")
    print(f"[Done] Best model saved to: {os.path.join(CKPT_ROOT, 'best_model.pth')}")
    mlflow.log_metric("best_val_dice", best_val_dice)
