# HyperVision KYC AI 🔍

> **Enterprise-grade Document Segmentation Pipeline** — UNet++ · EfficientNet-B0 · FastAPI · React

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react)](https://vitejs.dev)
[![Dice Score](https://img.shields.io/badge/Val%20Dice-0.9939-brightgreen)]()
[![Submission Dice](https://img.shields.io/badge/Test%20Dice-0.9392-green)]()

---

## 📋 Overview

HyperVision KYC AI is a production-ready, end-to-end document segmentation system built for the **Hyperverge Document Segmentation Challenge**. Given a photo containing a document (ID card, passport, etc.), the system predicts a precise polygon marking the document's boundary against the background — a critical preprocessing step for any KYC OCR pipeline.

While the baseline provided a monolithic Jupyter Notebook with a basic U-Net, this project re-architects the entire pipeline into an enterprise-grade modular codebase, wraps inference in a **FastAPI backend**, and integrates with a full **React/Vite frontend** for real-world deployment.

### 🏆 Final Results

| Metric | Score |
|--------|-------|
| Validation Dice (local) | **0.9939** |
| Test Dice (leaderboard) | **0.9392** |
| F1 @ IoU ≥ 0.50 | **0.7066** |
| Precision | **0.7102** |
| Recall | **0.7031** |
| True Positives / 1000 images | **1061** |
| Training Epochs | **20** |
| Training Time (RTX 4050) | **~47 min** |

---

## 🏗️ Architecture

### Model: UNet++ + EfficientNet-B0

```
Input Image (384×384)
        │
   ┌────▼────────────────────────────────────┐
   │        EfficientNet-B0 Encoder          │
   │   (Pretrained on ImageNet — frozen      │
   │    for 2 epochs, then fine-tuned)       │
   └──┬──────┬──────┬──────┬──────┬──────────┘
      │      │      │      │      │  (multi-scale features)
   ┌──▼──────▼──────▼──────▼──────▼──────────┐
   │     UNet++ Decoder (Dense Skip Paths)   │
   │   (Nested connections preserve edges)   │
   └──────────────────────┬──────────────────┘
                          │
              ┌───────────▼───────────┐
              │   1×1 Conv → Sigmoid  │
              └───────────┬───────────┘
                          │
               Binary Mask (384×384)
                          │
              ┌───────────▼───────────┐
              │  findContours + DPS   │
              │  approxPolyDP         │
              └───────────┬───────────┘
                          │
           Normalized Polygon [[x,y], ...]
```

### System Architecture

```
HyperVision KYC AI/
├── src/                          # Core ML Package
│   ├── config/
│   │   └── settings.py           # All hyperparameters & paths
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset + DataLoader
│   │   ├── transforms.py         # Albumentations augmentation pipeline
│   │   └── annotations.py        # CSV polygon → binary mask converter
│   ├── inference/
│   │   └── predictor.py          # Model loading + single-image inference
│   ├── models/
│   │   └── unet.py               # Architecture definitions
│   ├── postprocessing/
│   │   └── polygon.py            # Mask → normalized polygon (Douglas-Peucker)
│   └── training/
│       ├── trainer.py            # Epoch loop + AMP + metric logging
│       ├── losses.py             # Combined BCE + Dice loss
│       └── checkpoint.py         # Best-model saver (metric-driven)
│
├── scripts/                      # Entry Points
│   ├── train.py                  # Main training script
│   ├── generate_submission.py    # Inference on 1000 test images → pred.csv
│   ├── submit_to_api.py          # POST pred.csv to Hyperverge API
│   ├── visual_check.py           # Draw predicted polygons on test images
│   └── run_api.py                # FastAPI server launcher
│
├── backend/
│   └── main.py                   # FastAPI app (health + /predict endpoints)
│
├── frontend/                     # React/Vite Web Interface
│   └── src/
│       ├── routes/analysis.tsx   # Document upload + live polygon visualization
│       └── lib/api.ts            # Typed API client
│
└── data/
    ├── checkpoints/best_model.pth
    └── predictions/pred.csv
```

---

## 🧠 ML Concepts & Design Decisions

### 1. UNet++ (Dense Skip Connections)
Standard U-Net has a single skip connection per resolution level. **UNet++** adds nested, dense skip connections between encoder and decoder nodes — forcing the model to learn at multiple semantic depths simultaneously. This recovers fine-grained boundary details crucial for precise document corner prediction.

### 2. Transfer Learning — EfficientNet-B0
Instead of random weight initialization, we inject an **EfficientNet-B0** encoder pretrained on 1.2M ImageNet images. The encoder already understands edges, textures, and shapes. We only needed 20 epochs to adapt it to "document vs. background" rather than the 100+ epochs required from scratch.

### 3. Encoder Freezing (Catastrophic Forgetting Prevention)
For the first 2 warm-up epochs, the EfficientNet encoder is **frozen** — only the UNet++ decoder learns. This prevents the large error signal from a fresh decoder from destroying the pretrained weights. After epoch 2, the full network is fine-tuned jointly.

### 4. OneCycleLR — Super Convergence
A linear warm-up to `max_lr=1e-3` followed by cosine annealing to `1e-7`. This "super convergence" technique allows using a much higher peak learning rate than typical, producing faster convergence and escaping poor local minima. Achieved 0.97+ Dice by epoch 5.

### 5. Automatic Mixed Precision (AMP / FP16)
`torch.amp.autocast('cuda')` selectively runs safe operations in 16-bit floats. This halves GPU memory footprint, allowing **Batch Size = 16** on a 6GB RTX 4050 — effectively doubling throughput vs. FP32 at Batch Size = 8.

### 6. Combined Loss: BCE + Dice
```python
loss = 0.5 * BCE(logits, mask) + 0.5 * DiceLoss(sigmoid(logits), mask)
```
- **BCE** optimizes per-pixel accuracy independently
- **Dice** optimizes global shape overlap (IoU-like)
- Combined: achieves both sharp pixel boundaries and correct overall shape

### 7. AdamW Optimizer (Decoupled Weight Decay)
Standard Adam decays weights poorly (coupling it with gradient updates). **AdamW** mathematically separates weight decay (`wd=1e-4`), providing stronger regularization and preventing overfitting on the 5,000-image training set.

### 8. Albumentations Augmentation Pipeline
```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomShadow(p=0.3),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(p=0.2),
])
```
Applied on-the-fly every epoch — effectively creating infinite variations of the 5,000 training images.

### 9. Geometric Post-Processing (Douglas-Peucker)
Neural network output → sigmoid → binary mask → `cv2.findContours` → `cv2.approxPolyDP` (Douglas-Peucker algorithm). This simplifies potentially hundreds of jagged contour pixels into a clean, minimal polygon. Coordinates are normalized per image: `x /= width`, `y /= height` → all values in `[0, 1]`.

### 10. Metric-Driven Checkpointing
`best_model.pth` is only saved when `val_dice` reaches a new all-time high. This guarantees the deployed model is always the most generalized version, not just the final epoch.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- CUDA-capable GPU (tested on RTX 4050 6GB)
- `uv` package manager

### 1. Setup Environment
```bash
uv venv --python 3.11
.venv\Scripts\activate          # Windows
uv pip install -r requirement.txt
uv pip install segmentation-models-pytorch timm
```

### 2. Prepare Dataset
Download from Kaggle: `yashvardhangera/hv-doc-data`

Extract to:
```
src/data/archive/student/student/
├── images/train/   (5,000 images)
├── images/test/    (1,000 images)
└── labels/train_round_1.csv
```

### 3. Train the Model
```bash
python scripts/train.py
```
**Expected output:**
```
[Warm-up] Encoder frozen — training decoder only for first 2 epochs
Epoch [01/20]  train_dice=0.3421  val_dice=0.9490  lr=2.32e-04
[*] New best val_dice=0.9490 — checkpoint saved.
...
Epoch [20/20]  train_dice=0.0184  val_dice=0.9939  lr=1.00e-07
[Done] Training complete. Best val_dice=0.9939 at epoch 19
```

### 4. Generate Submission CSV
```bash
python scripts/generate_submission.py
# Output: data/predictions/pred.csv (1000 rows, coords in [0,1])
```

### 5. Submit to Leaderboard
```bash
python scripts/submit_to_api.py
```

### 6. Visual Sanity Check
```bash
python scripts/visual_check.py
# Opens: data/predictions/visual_check/ — polygons drawn on test images
```

---

## 🌐 Web Application

The model is deployed as a live web application.

### Start Backend API
```bash
python scripts/run_api.py
# API live at: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

#### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model status + device info |
| `POST` | `/predict` | Upload image → polygon + mask overlay |
| `POST` | `/predict/base64` | Base64 image → polygon + mask overlay |

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Navigate to the localhost URL. Drag-and-drop any document image to see the AI draw the polygon boundary in real-time.

---

## 📊 Training Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | UNet++ |
| Encoder | EfficientNet-B0 (ImageNet pretrained) |
| Input Resolution | 384 × 384 |
| Batch Size | 16 (AMP enabled) |
| Epochs | 20 (2 warm-up + 18 fine-tune) |
| Optimizer | AdamW (`lr=1e-3`, `wd=1e-4`) |
| Scheduler | OneCycleLR (`max_lr=1e-3`) |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Hardware | NVIDIA RTX 4050 6GB |
| Training Time | ~47 minutes |

---

## 💡 Key Findings

> *"The most significant finding was that combining Transfer Learning (pretrained EfficientNet-B0 encoder) with UNet++ architecture achieved a 0.9392 Test Dice in just 20 epochs. Using the OneCycleLR scheduler with Mixed Precision (AMP) allowed doubling the batch size to 16, cutting training time by ~60%. The biggest engineering challenge was normalizing polygon coordinates correctly to [0,1] relative to each image's actual pixel dimensions rather than the model's internal resolution."*

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| [`scripts/train.py`](scripts/train.py) | Main training entry point |
| [`scripts/generate_submission.py`](scripts/generate_submission.py) | Batch inference → pred.csv |
| [`src/inference/predictor.py`](src/inference/predictor.py) | Single-image inference |
| [`src/postprocessing/polygon.py`](src/postprocessing/polygon.py) | Mask → polygon extraction |
| [`src/data/transforms.py`](src/data/transforms.py) | Augmentation pipeline |
| [`backend/main.py`](backend/main.py) | FastAPI application |
| [`frontend/src/routes/analysis.tsx`](frontend/src/routes/analysis.tsx) | Document analysis UI |

---

## 📝 Submission

- **Leaderboard API:** `http://3.108.8.61:8991/submit`
- **Test Dice Score:** 0.9392
- **Attempt:** 1 of 20
- **GitHub Repo:** https://github.com/Adhi0303/Hyperverge_KYC

---

*Built for the Hyperverge Document Segmentation Challenge — ADARSH S*
