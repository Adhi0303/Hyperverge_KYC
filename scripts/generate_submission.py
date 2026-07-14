"""
scripts/generate_submission.py
================================
Loads the best trained checkpoint and runs inference on all 1,000 test images,
producing a submission-ready pred.csv in the exact format required by the
HyperVision KYC assignment.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts/generate_submission.py

Output:
    data/predictions/pred.csv
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import warnings
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.config.settings import TEST_IMG_DIR, CKPT_ROOT, PRED_ROOT, DEVICE, MEAN, STD
from src.postprocessing.polygon import mask_to_polygons

# ── Config ────────────────────────────────────────────────
CHECKPOINT   = os.path.join(CKPT_ROOT, "best_model.pth")
OUTPUT_CSV   = os.path.join(PRED_ROOT, "pred.csv")
IMG_SIZE     = 384      # must match training resolution
MASK_THRESH  = 0.5      # sigmoid threshold for foreground
MIN_AREA     = 200      # ignore noise blobs smaller than this (px²)

os.makedirs(PRED_ROOT, exist_ok=True)

# ── Load Model ────────────────────────────────────────────
print(f"[Model] Loading checkpoint: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"No checkpoint found at {CHECKPOINT}.\n"
        "Please train the model first: .venv\\Scripts\\python.exe scripts/train.py"
    )

model = smp.UnetPlusPlus(
    encoder_name    = "efficientnet-b0",
    encoder_weights = None,     # weights come from checkpoint
    in_channels     = 3,
    classes         = 1,
    activation      = None,
)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)
print(f"[Model] Loaded — epoch {ckpt['epoch']}  |  best val_dice {ckpt['val_dice']:.4f}")

# ── Inference helpers ──────────────────────────────────────
MEAN_arr = np.array(MEAN, dtype=np.float32)
STD_arr  = np.array(STD,  dtype=np.float32)

def preprocess(bgr_img):
    """BGR OpenCV image -> normalised (1,3,H,W) float32 CUDA tensor."""
    rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - MEAN_arr) / STD_arr
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor

@torch.no_grad()
def predict_mask(bgr_img):
    """Returns a uint8 binary mask at the ORIGINAL image resolution."""
    h, w = bgr_img.shape[:2]
    tensor = preprocess(bgr_img)
    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W) float32
    binary = (prob > MASK_THRESH).astype(np.uint8)
    # Resize back to original image dimensions for accurate polygon coordinates
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

# ── Run on all test images ─────────────────────────────────
test_images = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not test_images:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

print(f"\n[Inference] Running on {len(test_images)} test images ...")
results = []

for fname in tqdm(test_images, desc="Predicting"):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    bgr = cv2.imread(img_path)

    if bgr is None:
        print(f"  [WARN] Could not read {fname} — writing empty polygon.")
        results.append({"image": fname, "polygon": "[]"})
        continue

    mask = predict_mask(bgr)
    h, w = bgr.shape[:2]

    # Convert binary mask → normalized polygon coordinates
    # Pass (w, h) so coords are divided by actual image dimensions → [0, 1]
    polys = mask_to_polygons(mask, size=(w, h), min_area=MIN_AREA)

    # Encode as JSON string in the required submission format
    # [[x1,y1],[x2,y2],...] — coordinates already normalised to [0,1]
    poly_json = json.dumps([p for p in polys]) if polys else "[]"
    results.append({"image": fname, "polygon": poly_json})

# ── Write CSV ──────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "polygon"])
    writer.writeheader()
    writer.writerows(results)

total    = len(results)
non_empty = sum(1 for r in results if r["polygon"] != "[]")
print(f"\n[Done] pred.csv written to: {OUTPUT_CSV}")
print(f"       {non_empty}/{total} images have predicted polygons ({100*non_empty/total:.1f}%)")
print(f"       {total - non_empty} images returned empty polygon []")
