# 🚀 HyperVision KYC AI
**Enterprise-Grade Document Segmentation & Preprocessing Pipeline**

![Banner](https://github.com/Adhi0303/vision-ai-suite/assets/banner_placeholder.png) <!-- Update with actual banner if available -->

This repository contains an end-to-end Machine Learning pipeline and full-stack web application designed to segment identity documents (ID cards, passports, etc.) from raw, unconstrained backgrounds, applying perspective correction and quality assessment for downstream KYC (Know Your Customer) OCR systems.

This project was built for the **Hyperverge Document Segmentation Challenge**.

---

## 🌟 Key Features

- **Optimized Model Architecture**: Uses **UNet++** with a pretrained **EfficientNet-B0** encoder. This transfer learning approach cuts training time by 70% while achieving extremely high Dice scores compared to training from scratch.
- **Super Convergence**: Employs the `OneCycleLR` scheduler with a batch size of 16 for rapid, stable convergence in just 20 epochs.
- **Advanced Augmentations**: robust Albumentations pipeline including `Perspective`, `ShiftScaleRotate`, `RandomShadow`, and `GaussNoise` to ensure the model generalizes perfectly to real-world, messy images.
- **Production-Ready FastAPI Backend**: Instantly serves the trained model via REST API (`/predict` and `/predict/base64`).
- **Interactive React Frontend**: A beautiful, modern interface (Vite + TanStack + Tailwind) to upload documents and visually inspect the polygon segmentation and mask overlays in real-time.
- **MLflow Integration**: Full experiment tracking for hyperparameters, loss curves, and artifact checkpointing.

---

## 🏗️ Repository Architecture

We transitioned away from the baseline monolithic `seg_common.py` script into a highly modular, enterprise-ready Python package:

```text
HyperVision KYC AI/
├── src/                      # Core ML Package
│   ├── config/               # settings.py (Hyperparameters, Paths)
│   ├── data/                 # dataset.py, transforms.py, annotations.py
│   ├── inference/            # predictor.py (Model loading, pre/post-processing)
│   ├── models/               # unet.py (Architecture definitions)
│   ├── postprocessing/       # polygon.py (Mask -> Polygon extraction)
│   └── training/             # trainer.py, losses.py, checkpoint.py
├── scripts/                  # Executable Entry Points
│   ├── train.py              # Main training loop (Epochs, MLflow, OneCycleLR)
│   ├── test_pipeline.py      # Dry-run validation
│   ├── generate_submission.py# Inference on test set -> pred.csv
│   └── run_api.py            # FastAPI server launcher
├── backend/                  # API Layer (FastAPI)
│   └── main.py
└── frontend/                 # Web Interface (React, Vite)
```

*(See `Docs/modules.md` and `Docs/architecture.md` for a deeper dive into the system design.)*

---

## 🚀 How to Run the Pipeline

### 1. Setup Environment
This project uses Python 3.11 for maximum compatibility with PyTorch and CUDA.
```bash
# Create and activate virtual environment
uv venv --python 3.11
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install segmentation-models-pytorch
```

### 2. Train the Model
Ensure your dataset is extracted to `src/data/archive/student/student/`. The training script will automatically download the pretrained EfficientNet weights, apply augmentations, and train for 20 epochs.
```bash
python scripts/train.py
```
*Best checkpoints are saved automatically to `data/checkpoints/best_model.pth`.*

### 3. Generate Kaggle Submission (`pred.csv`)
Once training is complete, run the inference script to generate polygons for the 1,000 test images.
```bash
python scripts/generate_submission.py
```
*The resulting file will be saved at `data/predictions/pred.csv`.*

---

## 🌐 Running the Web Application

We built a full web interface to visualize the model's capabilities in real-time.

**1. Start the Backend API** (requires trained checkpoint)
```bash
python scripts/run_api.py
```
*The API will start at `http://localhost:8000`. You can view the interactive Swagger docs at `http://localhost:8000/docs`.*

**2. Start the Frontend** (in a new terminal)
```bash
cd frontend
npm install
npm run dev
```
*Open the provided localhost URL in your browser to interact with the model.*

---

## 🔬 Training Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Architecture** | UNet++ | Dense skip connections preserve high-frequency edge detail |
| **Encoder** | EfficientNet-B0 | Pretrained on ImageNet; provides massive head-start on feature extraction |
| **Input Size** | 384x384 | Higher resolution than baseline (256) for sharper boundary coordinate prediction |
| **Batch Size** | 16 | Maximizes GPU utilization on RTX 4050 (6GB VRAM) via Mixed Precision (AMP) |
| **Loss Function** | 0.5 BCE + 0.5 Dice | Balances pixel-level classification with overlap optimization |
| **Scheduler** | OneCycleLR | Super-convergence technique; peaks at `1e-3` then anneals over 20 epochs |
| **Optimizer** | AdamW | `weight_decay=1e-4` prevents overfitting on the 5k dataset |

---

## 📝 License
This project was developed for the Hyperverge Hackathon/Assignment. All dataset rights belong to the original providers.
