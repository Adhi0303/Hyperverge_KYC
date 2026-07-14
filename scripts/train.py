"""
scripts/train.py  —  v2 (Optimized Transfer Learning Edition)
==============================================================
Key upgrades over v1:
  1. Pretrained EfficientNet-B0 encoder (segmentation-models-pytorch)
     → The encoder already knows edges, textures, shapes from ImageNet.
       The model only needs to learn "what is a document", not "what is an edge".
       This cuts convergence from 50 epochs → 15-20 epochs.

  2. OneCycleLR scheduler ("super convergence")
     → LR linearly warms up to max, then cosine-anneals down.
       Proven to converge 5-10x faster than vanilla CosineAnnealingLR.

  3. Batch size 16 (doubled)
     → More GPU utilization per iteration = faster epoch wall-time.

  4. Larger input resolution: 384px
     → Document edges are thin — higher resolution = better boundary accuracy.

  5. Encoder frozen for first 2 epochs (warm-up)
     → Prevents the pre-trained weights from being destroyed by a randomly
        initialized decoder in the first iterations.

Usage:
  (press Ctrl+C on the old run first, then:)
  .venv\\Scripts\\python.exe scripts/train.py
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import warnings
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split
import mlflow

warnings.filterwarnings("ignore")

from src.config.settings import TRAIN_CSV, DEVICE, CKPT_ROOT
from src.data.annotations  import load_csv_records
from src.data.dataset       import CsvSegDataset
from src.data.transforms    import get_train_transforms, get_val_transforms
from src.training.checkpoint import save_checkpoint

# ─────────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────────
EPOCHS         = 20          # 20 epochs with transfer learning >> 50 from scratch
BATCH_SIZE     = 16          # doubled — more GPU parallelism
IMG_SIZE       = 384         # larger resolution for sharper document edges
LR_MAX         = 1e-3        # OneCycleLR peak (higher is safe with warm-up)
WEIGHT_DECAY   = 1e-4
VAL_FRACTION   = 0.2
NUM_WORKERS    = 0           # Windows: keep 0
ENCODER_FREEZE_EPOCHS = 2   # freeze encoder for first 2 epochs (warm-up decoder)
SEED           = 42

os.makedirs(CKPT_ROOT, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
#  Update transforms to use new IMG_SIZE
# ─────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config.settings import MEAN, STD

def get_train_tf():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=20,
                           border_mode=0, p=0.6),
        A.Perspective(scale=(0.04, 0.12), p=0.4),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
        A.HueSaturationValue(10, 25, 25, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.2),
        A.RandomShadow(p=0.2),
        A.CLAHE(clip_limit=3.0, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_tf():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────
#  Dataset & DataLoaders
# ─────────────────────────────────────────────
print("[Data] Loading CSV records ...")
all_records = load_csv_records(TRAIN_CSV)
print(f"[Data] Total records: {len(all_records)}")

n_val   = int(len(all_records) * VAL_FRACTION)
n_train = len(all_records) - n_val
generator = torch.Generator().manual_seed(SEED)
train_records, val_records = random_split(all_records, [n_train, n_val], generator=generator)
train_records = [all_records[i] for i in train_records.indices]
val_records   = [all_records[i] for i in val_records.indices]

print(f"[Data] Train: {len(train_records)} | Val: {len(val_records)}")

train_ds = CsvSegDataset(train_records, transform=get_train_tf())
val_ds   = CsvSegDataset(val_records,   transform=get_val_tf())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ─────────────────────────────────────────────
#  Model: Unet++ with pretrained EfficientNet-B0
#  encoder_weights="imagenet" downloads ~20 MB weights automatically
# ─────────────────────────────────────────────
model = smp.UnetPlusPlus(
    encoder_name    = "efficientnet-b0",
    encoder_weights = "imagenet",          # pretrained!
    in_channels     = 3,
    classes         = 1,
    activation      = None,                # raw logits — we apply sigmoid manually
)
model = model.to(DEVICE)
print(f"[Model] UNet++ / EfficientNet-B0 (pretrained) initialized on {DEVICE}")

# ─────────────────────────────────────────────
#  Loss: SMP built-in combined DiceBCE loss
# ─────────────────────────────────────────────
criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss  = nn.BCEWithLogitsLoss()

def combined_loss(logits, masks):
    return 0.5 * bce_loss(logits, masks) + 0.5 * criterion(logits, masks)

# ─────────────────────────────────────────────
#  Optimizer & Scheduler
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX / 10, weight_decay=WEIGHT_DECAY)

# OneCycleLR: warmup + cosine anneal in one cycle — "super convergence"
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr         = LR_MAX,
    steps_per_epoch= len(train_loader),
    epochs         = EPOCHS,
    pct_start      = 0.2,        # 20% of training is warm-up
    anneal_strategy= "cos",
    div_factor     = 10.0,       # initial_lr = max_lr / 10
    final_div_factor=1000.0,     # final_lr   = max_lr / 1000
)

scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

# ─────────────────────────────────────────────
#  Helper: compute pixel Dice on a batch
# ─────────────────────────────────────────────
def batch_dice(logits, masks, eps=1e-6):
    probs = torch.sigmoid(logits).detach()
    preds = (probs > 0.5).float()
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
    return ((2.0 * intersection + eps) / (union + eps)).mean().item()

# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
from tqdm import tqdm

def run_epoch(is_train):
    model.train(is_train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    loader = train_loader if is_train else val_loader
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, masks in tqdm(loader, desc="train" if is_train else "val  ", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(images)
                loss   = combined_loss(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()       # OneCycleLR steps every ITERATION

            total_loss += loss.item()
            total_dice += batch_dice(logits, masks)
            n += 1

    return total_loss / n, total_dice / n

# ─────────────────────────────────────────────
#  MLflow run
# ─────────────────────────────────────────────
mlflow.set_experiment("HyperVision-UNet-Training")

with mlflow.start_run(run_name=f"unetpp_effb0_e{EPOCHS}_bs{BATCH_SIZE}_res{IMG_SIZE}"):
    mlflow.log_params({
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "img_size": IMG_SIZE,
        "lr_max": LR_MAX, "weight_decay": WEIGHT_DECAY,
        "model": "UNet++ / EfficientNet-B0 (pretrained)",
        "scheduler": "OneCycleLR", "loss": "0.5*BCE + 0.5*Dice",
        "device": DEVICE,
    })

    best_val_dice = 0.0
    print(f"\n[Train] Starting {EPOCHS}-epoch transfer-learning run on {DEVICE} ...\n")

    for epoch in range(1, EPOCHS + 1):

        # --- Encoder warm-up: freeze encoder for first N epochs ---
        if epoch == 1:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("[Warm-up] Encoder frozen — training decoder only for first 2 epochs")
        elif epoch == ENCODER_FREEZE_EPOCHS + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print("[Warm-up] Encoder unfrozen — full fine-tuning begins")

        train_loss, train_dice = run_epoch(is_train=True)
        val_loss,   val_dice   = run_epoch(is_train=False)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"train_loss={train_loss:.4f}  train_dice={train_dice:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
            f"lr={current_lr:.2e}"
        )

        mlflow.log_metrics({
            "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss,     "val_dice": val_dice,
            "lr": current_lr,
        }, step=epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = save_checkpoint(model, optimizer, epoch, val_dice,
                                        filename="best_model.pth")
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            print(f"  [*] New best val_dice={val_dice:.4f} — checkpoint saved.")

    print(f"\n[Done] Training complete.  Best val_dice = {best_val_dice:.4f}")
    mlflow.log_metric("best_val_dice", best_val_dice)
