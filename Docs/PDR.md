# 📄 Project Design Report (PDR)

# HyperVision KYC AI
### AI-Powered Document Segmentation & Preprocessing Pipeline

---

# Version

| Field | Value |
|--------|--------|
| Version | 1.0 |
| Project Type | Computer Vision |
| Domain | Document AI / Digital KYC |
| Framework | PyTorch |
| Cloud Platform | Microsoft Azure Machine Learning |
| Experiment Tracking | MLflow |
| Frontend | Lovable |
| Backend | FastAPI |

---

# 1. Introduction

HyperVision KYC AI is an AI-powered document segmentation system designed to detect and isolate documents from unconstrained real-world images.

The system acts as the first stage of a Digital KYC pipeline where uploaded documents are automatically identified, segmented, corrected, and prepared for downstream tasks such as OCR, identity verification, fraud detection, and document analysis.

Rather than treating document segmentation as an isolated research problem, this project models how enterprise identity verification platforms preprocess customer documents before extracting information.

---

# 2. Problem Statement

Document images captured by users often contain:

- Background clutter
- Perspective distortion
- Rotation
- Shadows
- Partial visibility
- Variable lighting
- Blur

These factors reduce OCR accuracy and negatively affect downstream identity verification systems.

The objective is to automatically detect the document boundary and generate an accurate segmentation polygon that separates the document from the surrounding background.

---

# 3. Business Motivation

Digital onboarding platforms process millions of identity documents every day.

Before OCR or identity verification can begin, the system must first determine:

- Where is the document?
- What are its boundaries?
- Can it be corrected?
- Is it suitable for OCR?

This segmentation stage significantly improves:

- OCR accuracy
- Face matching
- Identity verification
- Fraud detection
- User onboarding experience

---

# 4. Project Objectives

## Primary Objectives

- Train a robust document segmentation model.
- Predict document polygons accurately.
- Generate submission-ready predictions.
- Achieve high Dice Score and IoU.

---

## Secondary Objectives

Build a production-inspired preprocessing pipeline including:

- Polygon extraction
- Perspective correction
- Document cropping
- Image enhancement
- Confidence estimation

---

# 5. Scope

## Included

✔ Document Segmentation

✔ Polygon Prediction

✔ Image Preprocessing

✔ Perspective Transformation

✔ Azure ML Training

✔ MLflow Experiment Tracking

✔ REST API

✔ Interactive Frontend

---

## Excluded

OCR

Identity Verification

Face Matching

Fraud Detection

Document Classification

(These are downstream stages.)

---

# 6. Dataset Overview

Dataset Source:

HyperVerge Document Segmentation Dataset

Training Images

5,000

Test Images

1,000

Annotation Files

train_round_1.csv

train_round_2.csv

train_round_3.csv

Each annotation contains normalized polygon coordinates representing document boundaries.

---

# 7. Machine Learning Problem

Problem Type

Semantic Segmentation

Input

RGB Image

Output

Binary Segmentation Mask

Post Processing

Polygon Extraction

Evaluation Metrics

- Dice Score
- IoU
- Precision
- Recall
- F1 Score

---

# 8. Solution Overview

The proposed solution consists of multiple stages.

User Upload

↓

Image Validation

↓

Preprocessing

↓

Deep Learning Model

↓

Segmentation Mask

↓

Polygon Extraction

↓

Perspective Correction

↓

Enhanced Document

↓

Frontend Visualization

---

# 9. System Architecture

                Frontend (Lovable)
                        │
                        ▼
                 FastAPI Backend
                        │
                        ▼
             Image Preprocessing Module
                        │
                        ▼
                Segmentation Model
                        │
                        ▼
                  Binary Mask
                        │
                        ▼
               Polygon Extraction
                        │
                        ▼
           Perspective Transformation
                        │
                        ▼
              Enhanced Output Image

---

# 10. Data Pipeline

Image Upload

↓

Validation

↓

Resize

↓

Normalization

↓

Dataset Loader

↓

Training

↓

Inference

↓

Mask Generation

↓

Polygon Extraction

↓

Perspective Correction

↓

Result Serialization

↓

Frontend

---

# 11. Preprocessing Pipeline

Image Loading

↓

Resize

↓

Normalize

↓

Data Augmentation

↓

Mask Generation

↓

Tensor Conversion

↓

Model

Possible augmentations:

- Horizontal Flip
- Rotation
- Brightness
- Contrast
- Blur
- Random Crop

---

# 12. Model Architecture

Baseline

U-Net

Potential Improvements

- UNet++
- DeepLabV3+
- Attention U-Net

Loss Functions

- Dice Loss
- BCE Loss
- Dice + BCE Hybrid

Optimizer

AdamW

Learning Rate Scheduler

Cosine Annealing

---

# 13. Training Pipeline

Load Dataset

↓

Generate Masks

↓

Data Loader

↓

Train Model

↓

Validation

↓

Metric Logging

↓

Checkpoint Saving

↓

Best Model Selection

---

# 14. Experiment Tracking

All experiments will be tracked using MLflow.

Tracked Parameters

- Learning Rate
- Epochs
- Batch Size
- Optimizer
- Loss Function
- Encoder
- Image Resolution

Tracked Metrics

- Dice
- IoU
- Precision
- Recall
- F1

Artifacts

- Model
- Masks
- Prediction Samples
- Training Curves

---

# 15. Azure Machine Learning Integration

Azure Workspace

hyperverge-ml-workspace

Compute Cluster

hv-dev-cluster

Compute Instance

suriyaadhi0072

Azure Components

- Workspace
- Compute Cluster
- Training Jobs
- MLflow Tracking
- Model Registry

Purpose

Provide scalable cloud-based experimentation while maintaining reproducibility.

---

# 16. Backend Design

Framework

FastAPI

Endpoints

POST /predict

Input

Image

Output

- Segmentation Mask
- Polygon
- Confidence
- Cropped Document

---

# 17. Frontend Design

Framework

Lovable

Features

- Drag & Drop Upload
- Prediction Dashboard
- Polygon Overlay
- Before vs After
- Confidence Meter
- Download Result

---

# 18. Evaluation Strategy

Metrics

Primary

- Dice Score

Secondary

- IoU

Additional

- Precision
- Recall
- F1

Visual Evaluation

- Polygon Accuracy
- Boundary Smoothness
- Cropping Quality

---

# 19. Expected Deliverables

- Trained Model
- Prediction CSV
- FastAPI Backend
- Lovable Frontend
- Azure ML Experiment
- MLflow Logs
- Documentation
- Source Code

---

# 20. Future Scope

The segmentation module can become the first stage of a complete enterprise KYC pipeline.

Future modules include:

Document Classification

↓

OCR

↓

Face Detection

↓

Face Matching

↓

Deepfake Detection

↓

Fraud Analysis

↓

Identity Verification

↓

Customer Onboarding

---

# 21. Conclusion

HyperVision KYC AI demonstrates how modern computer vision can automate document localization and preprocessing for digital identity verification systems.

The project combines deep learning, cloud-based machine learning workflows, experiment tracking, and production-ready APIs into a modular architecture inspired by enterprise AI platforms.

This approach ensures the solution is scalable, reproducible, and extensible for future KYC and document intelligence applications.