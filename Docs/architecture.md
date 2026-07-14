# 🏗️ ARCHITECTURE.md

# HyperVision KYC AI

## System Architecture Design

Version: 1.0

---

# 1. Overview

HyperVision KYC AI is designed as a modular Computer Vision system inspired by enterprise-grade Digital KYC pipelines.

Rather than treating document segmentation as a standalone model, the system is structured into independent services that work together to produce production-ready document outputs.

The architecture follows a layered approach consisting of:

- Presentation Layer
- API Layer
- AI Inference Layer
- Machine Learning Layer
- Cloud Layer
- Experiment Tracking Layer

This modular architecture allows each component to evolve independently while maintaining scalability and maintainability.

---

# 2. High-Level Architecture

                    ┌──────────────────────────┐
                    │       User / Browser      │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Lovable Frontend    │
                    └────────────┬──────────────┘
                                 │ REST API
                                 ▼
                    ┌──────────────────────────┐
                    │      FastAPI Backend     │
                    └────────────┬──────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
     Image Validation     Image Preprocessing    Model Loader
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Document Segmentation AI │
                    │        (U-Net)           │
                    └────────────┬──────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Binary Segmentation Mask │
                    └────────────┬──────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Document Intelligence    │
                    │ Pipeline                 │
                    └────────────┬──────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      Polygon Extraction   Perspective Fix    Quality Check
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ JSON Response Generator  │
                    └────────────┬──────────────┘
                                 ▼
                         Frontend Dashboard

---

# 3. Layered Architecture

## Layer 1 — Presentation Layer

Responsibilities

- Upload Image
- View Prediction
- Display Polygon
- Show Confidence
- Download Result

Technology

- Lovable

---

## Layer 2 — API Layer

Responsibilities

- Receive Upload
- Validate Input
- Call AI Pipeline
- Return JSON Response

Technology

- FastAPI

---

## Layer 3 — AI Processing Layer

Responsibilities

- Image Preprocessing
- Model Inference
- Mask Prediction

Technology

- PyTorch

---

## Layer 4 — Document Intelligence Layer

Responsibilities

- Polygon Extraction
- Perspective Correction
- Document Cropping
- Image Enhancement
- Quality Assessment

Technology

- OpenCV

---

## Layer 5 — Cloud Layer

Responsibilities

- Model Training
- Compute Resources
- Model Registry

Technology

- Azure Machine Learning

Workspace

hyperverge-ml-workspace

Compute Cluster

hv-dev-cluster

Compute Instance

suriyaadhi0072

---

## Layer 6 — Experiment Layer

Responsibilities

- Track Metrics
- Track Parameters
- Save Artifacts
- Register Models

Technology

- MLflow

---

# 4. Component Interaction

Frontend

↓

Backend

↓

Preprocessing

↓

Segmentation Model

↓

Mask

↓

Polygon Extraction

↓

Perspective Correction

↓

Response

↓

Frontend

---

# 5. Training Architecture

Dataset

↓

Annotation Parser

↓

Mask Generator

↓

Augmentation

↓

Data Loader

↓

Training

↓

Validation

↓

MLflow Logging

↓

Checkpoint

↓

Azure Model Registry

---

# 6. Inference Architecture

Input Image

↓

Validation

↓

Resize

↓

Normalize

↓

Model

↓

Segmentation Mask

↓

Contour Detection

↓

Polygon

↓

Perspective Transform

↓

Quality Score

↓

JSON Response

---

# 7. Deployment Architecture

Developer

↓

Git Repository

↓

Azure ML Workspace

↓

Training Job

↓

Registered Model

↓

FastAPI Service

↓

Frontend

---

# 8. Future Expansion

Current Architecture

Document Segmentation

↓

Future

OCR

↓

Document Classification

↓

Face Matching

↓

Deepfake Detection

↓

Fraud Detection

↓

Identity Verification

↓

Digital Onboarding

The modular architecture ensures these components can be integrated without changing the existing pipeline.

# 🔄 DATAFLOW.md

# HyperVision KYC AI

## End-to-End Data Flow

Version: 1.0

---

# 1. Overview

This document describes how data moves throughout the HyperVision KYC AI system.

The pipeline begins when a user uploads an image and ends when the processed document and segmentation results are returned.

---

# 2. Complete Pipeline

User Upload

↓

Frontend

↓

Backend

↓

Validation

↓

Preprocessing

↓

Segmentation Model

↓

Mask

↓

Polygon

↓

Perspective Correction

↓

Quality Assessment

↓

Response

↓

Frontend

---

# 3. Detailed Data Flow

## Step 1

User uploads image.

Input

JPG / PNG

↓

Frontend receives image.

---

## Step 2

Frontend sends image.

Method

POST /predict

↓

FastAPI

---

## Step 3

Backend validation.

Checks

- File exists
- File type
- Image size
- Corruption

↓

Accepted

---

## Step 4

Image preprocessing.

Operations

- Resize
- Normalize
- Convert Tensor

↓

Model Ready

---

## Step 5

Model inference.

Input

Tensor

↓

U-Net

↓

Binary Mask

---

## Step 6

Post Processing.

Mask

↓

Contour Detection

↓

Largest Contour

↓

Polygon Simplification

↓

Document Polygon

---

## Step 7

Perspective Correction.

Polygon

↓

Homography

↓

Warp Perspective

↓

Straightened Document

---

## Step 8

Quality Assessment.

Calculate

- Blur
- Brightness
- Contrast
- Skew

↓

Quality Score

---

## Step 9

Backend prepares response.

Includes

Original Image

Segmentation Mask

Polygon

Confidence

Quality Score

Enhanced Document

---

## Step 10

Frontend visualization.

Displays

Original

↓

Mask

↓

Polygon Overlay

↓

Corrected Document

↓

Download Button

---

# 4. Training Data Flow

CSV Annotation

↓

Polygon Parser

↓

Mask Generator

↓

Dataset

↓

Augmentation

↓

Data Loader

↓

Training

↓

Validation

↓

Metrics

↓

MLflow

↓

Checkpoint

---

# 5. Azure Data Flow

Dataset

↓

Azure Workspace

↓

Training Job

↓

Compute Cluster

↓

MLflow

↓

Registered Model

↓

Inference

---

# 6. MLflow Flow

Training

↓

Parameters

↓

Metrics

↓

Artifacts

↓

Model

↓

Experiment Dashboard

---

# 7. API Data Flow

Client

↓

POST /predict

↓

Backend

↓

Model

↓

Prediction

↓

JSON

↓

Frontend

---

# 8. Response Schema

Prediction Result

├── Prediction Status

├── Confidence

├── Polygon

├── Binary Mask

├── Corrected Document

├── Quality Score

└── Processing Time

---

# 9. Future Data Flow

Current

Image

↓

Segmentation

↓

Output

Future

Image

↓

Segmentation

↓

OCR

↓

Document Classification

↓

Face Detection

↓

Face Match

↓

Deepfake Detection

↓

Fraud Analysis

↓

Identity Verification

↓

Decision Engine