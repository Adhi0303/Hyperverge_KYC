# 📦 MODULES.md

# HyperVision KYC AI

## Module & Submodule Design Document

Version: 1.0

---

# Module Overview

The project is divided into independent functional modules to ensure:

- Scalability
- Reusability
- Maintainability
- Separation of Concerns
- Production Readiness

Each module performs one dedicated responsibility while communicating with other modules through clearly defined interfaces.

---

# Module Architecture

User
 │
 ▼
Frontend
 │
 ▼
API Gateway
 │
 ▼
Inference Engine
 │
 ├──────────────┐
 │              │
 ▼              ▼
Preprocessing   Model
 │              │
 └──────┬───────┘
        ▼
Post Processing
        │
        ▼
Result Generator
        │
        ▼
Frontend

Training Pipeline

Dataset
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
MLflow
 ↓
Azure ML

---

# Module 1
# Project Configuration

Purpose

Centralized project configuration.

Responsible for

- Global settings
- Config files
- Environment variables
- Paths
- Hyperparameters

Submodules

### 1.1 Environment Manager

Responsibilities

- Load .env
- Secrets
- Azure credentials

---

### 1.2 Configuration Loader

Responsibilities

- YAML
- JSON
- Runtime configs

---

### 1.3 Path Manager

Responsibilities

- Dataset paths
- Output paths
- Artifact paths

---

### 1.4 Logging Manager

Responsibilities

- Console logging
- File logging
- Error logs

---

# Module 2
# Data Management

Purpose

Manage the complete dataset lifecycle.

Submodules

---

### 2.1 Dataset Downloader

Responsibilities

- Download dataset
- Verify integrity

---

### 2.2 Dataset Loader

Responsibilities

- Load images
- Load annotations
- Create dataset object

---

### 2.3 Dataset Validator

Responsibilities

- Missing images
- Invalid polygons
- Corrupted files

---

### 2.4 Dataset Explorer

Responsibilities

- Statistics
- Image dimensions
- Label distribution

---

### 2.5 Dataset Splitter

Responsibilities

- Train
- Validation

---

# Module 3
# Annotation Processing

Purpose

Convert polygon annotations into training masks.

Submodules

---

### 3.1 Polygon Parser

Convert CSV polygons into coordinates.

---

### 3.2 Mask Generator

Convert polygons into binary masks.

---

### 3.3 Annotation Validator

Validate polygon correctness.

---

### 3.4 Coordinate Converter

Convert normalized coordinates.

---

# Module 4
# Image Preprocessing

Purpose

Prepare images for training.

Submodules

---

### 4.1 Image Loader

---

### 4.2 Resize

---

### 4.3 Normalization

---

### 4.4 Image Augmentation

Possible augmentations

- Flip
- Rotation
- Brightness
- Contrast
- Blur
- Noise

---

### 4.5 Tensor Conversion

Convert to PyTorch tensors.

---

# Module 5
# Training Pipeline

Purpose

Train segmentation model.

Submodules

---

### 5.1 Model Builder

Creates

- U-Net
- Future architectures

---

### 5.2 Loss Manager

Supports

- Dice Loss
- BCE Loss
- Hybrid Loss

---

### 5.3 Optimizer Manager

Supports

- Adam
- AdamW
- SGD

---

### 5.4 Scheduler

Learning rate scheduler.

---

### 5.5 Trainer

Runs

- Forward pass
- Backpropagation
- Weight update

---

### 5.6 Checkpoint Manager

Save

- Best model
- Latest model

---

# Module 6
# Validation & Evaluation

Purpose

Evaluate model performance.

Submodules

---

### 6.1 Validation Engine

---

### 6.2 Dice Calculator

---

### 6.3 IoU Calculator

---

### 6.4 Precision

---

### 6.5 Recall

---

### 6.6 F1 Score

---

### 6.7 Visualization

Generate

- Prediction samples
- Overlay masks

---

# Module 7
# Inference Engine

Purpose

Generate predictions.

Submodules

---

### 7.1 Image Predictor

---

### 7.2 Batch Predictor

---

### 7.3 Confidence Calculator

---

### 7.4 Prediction Formatter

Creates submission CSV.

---

# Module 8
# Post Processing

Purpose

Convert segmentation masks into usable document outputs.

Submodules

---

### 8.1 Contour Detection

Find document contour.

---

### 8.2 Polygon Extraction

Convert mask to polygon.

---

### 8.3 Polygon Simplification

Reduce unnecessary points.

---

### 8.4 Perspective Transformation

Straighten document.

---

### 8.5 Document Cropping

Crop document.

---

### 8.6 Image Enhancement

Improve quality.

Possible operations

- Sharpen
- Contrast
- Brightness
- CLAHE

---

### 8.7 Document Quality Assessment

Evaluate

- Blur
- Brightness
- Skew
- Visibility

---

# Module 9
# MLflow Integration

Purpose

Track experiments.

Submodules

---

### 9.1 Experiment Manager

---

### 9.2 Parameter Logger

Logs

- Learning rate
- Epochs
- Batch size

---

### 9.3 Metric Logger

Logs

- Dice
- IoU
- Precision
- Recall

---

### 9.4 Artifact Logger

Logs

- Model
- Masks
- Images

---

### 9.5 Model Registry

Register best model.

---

# Module 10
# Azure Machine Learning

Purpose

Cloud ML workflow.

Submodules

---

### 10.1 Workspace Manager

Workspace

hyperverge-ml-workspace

---

### 10.2 Compute Manager

Cluster

hv-dev-cluster

Instance

suriyaadhi0072

---

### 10.3 Job Manager

Submit training jobs.

---

### 10.4 Dataset Manager

Register datasets.

---

### 10.5 Model Registry

Store trained models.

---

### 10.6 Deployment Manager (Optional)

Future online endpoint.

---

# Module 11
# Backend API

Purpose

Serve predictions.

Framework

FastAPI

Submodules

---

### 11.1 Upload API

POST /upload

---

### 11.2 Predict API

POST /predict

---

### 11.3 Health API

GET /health

---

### 11.4 Model Loader

Load trained model.

---

### 11.5 Response Generator

Return

- Polygon
- Confidence
- Enhanced Image

---

# Module 12
# Frontend

Purpose

User Interface

Framework

Lovable

Submodules

---

### 12.1 Upload Screen

---

### 12.2 Prediction Dashboard

Display

- Original Image
- Mask
- Polygon

---

### 12.3 Before vs After

Perspective correction comparison.

---

### 12.4 Confidence Visualization

---

### 12.5 Download Results

---

### 12.6 Prediction History

---

# Module 13
# Utilities

Purpose

Shared helper functions.

Submodules

---

### 13.1 File Utilities

---

### 13.2 Image Utilities

---

### 13.3 Geometry Utilities

---

### 13.4 Polygon Utilities

---

### 13.5 Visualization Utilities

---

# Module Dependencies

Configuration
      │
      ▼
Data Management
      │
      ▼
Annotation Processing
      │
      ▼
Image Preprocessing
      │
      ▼
Training Pipeline
      │
      ▼
Validation
      │
      ▼
MLflow
      │
      ▼
Inference
      │
      ▼
Post Processing
      │
      ▼
Backend API
      │
      ▼
Frontend

---

# Future Expansion

The current system represents only the document segmentation stage.

Future enterprise modules may include:

- OCR Engine
- Document Classification
- Face Detection
- Face Matching
- Deepfake Detection
- Fraud Detection
- Identity Verification
- Digital KYC Workflow
- Human Review Queue
- Audit Logging
- Multi-document Processing

This modular architecture allows these capabilities to be integrated without restructuring the existing codebase.

---
Store trained models.

---

### 10.6 Deployment Manager (Optional)

Future online endpoint.

---

# Module 11
# Backend API

Purpose

Serve predictions.

Framework

FastAPI

Submodules

---

### 11.1 Upload API

POST /upload

---

### 11.2 Predict API

POST /predict

---

### 11.3 Health API

GET /health

---

### 11.4 Model Loader

Load trained model.

---

### 11.5 Response Generator

Return

- Polygon
- Confidence
- Enhanced Image

---

# Module 12
# Frontend

Purpose

User Interface

Framework

Lovable

Submodules

---

### 12.1 Upload Screen

---

### 12.2 Prediction Dashboard

Display

- Original Image
- Mask
- Polygon

---

### 12.3 Before vs After

Perspective correction comparison.

---

### 12.4 Confidence Visualization

---

### 12.5 Download Results

---

### 12.6 Prediction History

---

# Module 13
# Utilities

Purpose

Shared helper functions.

Submodules

---

### 13.1 File Utilities

---

### 13.2 Image Utilities

---

### 13.3 Geometry Utilities

---

### 13.4 Polygon Utilities

---

### 13.5 Visualization Utilities

---

# Module Dependencies

Configuration
      │
      ▼
Data Management
      │
      ▼
Annotation Processing
      │
      ▼
Image Preprocessing
      │
      ▼
Training Pipeline
      │
      ▼
Validation
      │
      ▼
MLflow
      │
      ▼
Inference
      │
      ▼
Post Processing
      │
      ▼
Backend API
      │
      ▼
Frontend

---

# Future Expansion

The current system represents only the document segmentation stage.

Future enterprise modules may include:

- OCR Engine
- Document Classification
- Face Detection
- Face Matching
- Deepfake Detection
- Fraud Detection
- Identity Verification
- Digital KYC Workflow
- Human Review Queue
- Audit Logging
- Multi-document Processing

This modular architecture allows these capabilities to be integrated without restructuring the existing codebase.

---

# 🚀 Implementation Progress Tracking

*This section is automatically updated to track implementation progress.*

- [x] **Module 1: Project Configuration** (Completed - `src/config/settings.py`)
- [x] **Module 2: Data Management** (Completed - `src/data/dataset.py`)
- [x] **Module 3: Annotation Processing** (Completed - `src/data/annotations.py`)
- [x] **Module 4: Image Preprocessing** (Completed - `src/data/transforms.py`)
- [x] **Module 5: Training Pipeline** (Completed - `src/models/unet.py`, `src/training/losses.py`, `src/training/trainer.py`, `src/training/checkpoint.py`, `scripts/train.py`)
- [x] **Module 6: Validation & Evaluation** (Completed - `src/evaluation/metrics.py`)
- [x] **Module 7: Inference Engine** (Partially Completed - `src/inference/predictor.py`)
- [x] **Module 8: Post Processing** (Partially Completed - `src/postprocessing/polygon.py`)
- [x] **Module 9: MLflow Integration** (Completed - embedded in `scripts/train.py`)
- [ ] **Module 10: Submission Generator** (`scripts/generate_submission.py`)
- [ ] **Module 11: Backend API**
- [ ] **Module 12: Frontend**
- [ ] **Module 13: Utilities**
