"""
backend/main.py
================
FastAPI application entry point for HyperVision KYC AI.

Endpoints:
  GET  /health              — Health check + model status
  POST /predict             — Upload an image, get back polygon + mask overlay
  POST /predict/base64      — Same but image sent as base64 string
  GET  /docs                — Auto-generated Swagger UI
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import json
import base64
import warnings
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from src.inference.predictor import load_model, predict_single
from src.postprocessing.polygon import mask_to_polygons
from src.config.settings import CKPT_ROOT, DEVICE

# ── Model singleton ────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")
    return _model

# ── Startup / Shutdown ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print("[API] Loading model checkpoint ...")
    try:
        _model = load_model("best_model.pth")
        print(f"[API] Model ready on {DEVICE}")
    except FileNotFoundError as e:
        print(f"[API] WARNING: {e}")
        print("[API] Server will start but /predict will return 503 until model is trained.")
    yield
    print("[API] Shutting down.")

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title       = "HyperVision KYC AI",
    description = "AI-Powered Document Segmentation & Preprocessing API",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Schemas ────────────────────────────────────────────────
class Base64Request(BaseModel):
    image_b64: str          # base64-encoded image (any common format)
    filename: str = "image.jpg"

class PredictionResponse(BaseModel):
    filename:     str
    has_document: bool
    polygon:      list       # list of [x, y] normalised points
    confidence:   float      # mean sigmoid probability in the predicted region
    mask_b64:     str        # base64-encoded PNG of the binary mask overlay
    device:       str

# ── Helpers ────────────────────────────────────────────────
def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG/PNG.")
    return img

def _build_response(filename: str, bgr: np.ndarray) -> dict:
    model = get_model()
    mask, prob = predict_single(model, bgr)

    # Polygon extraction
    polys = mask_to_polygons(mask, size=None, min_area=200)
    polygon = polys[0] if polys else []

    # Confidence: mean probability inside predicted region
    if mask.sum() > 0:
        confidence = float(prob[mask == 1].mean())
    else:
        confidence = 0.0

    # Build mask overlay (green tint on the document region)
    overlay = bgr.copy()
    overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([0, 200, 80]) * 0.5
    _, buf = cv2.imencode(".png", overlay.astype(np.uint8))
    mask_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "filename":     filename,
        "has_document": len(polygon) > 0,
        "polygon":      polygon,
        "confidence":   round(confidence, 4),
        "mask_b64":     mask_b64,
        "device":       DEVICE,
    }

# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "device":       DEVICE,
        "checkpoint":   os.path.join(CKPT_ROOT, "best_model.pth"),
        "checkpoint_exists": os.path.exists(os.path.join(CKPT_ROOT, "best_model.pth")),
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """Upload a document image file and receive the predicted polygon + mask overlay."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image (jpg/png).")
    data = await file.read()
    bgr  = _decode_image_bytes(data)
    return _build_response(file.filename, bgr)

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(req: Base64Request):
    """Send image as a base64 string — useful for browser/frontend clients."""
    try:
        data = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string.")
    bgr = _decode_image_bytes(data)
    return _build_response(req.filename, bgr)

@app.get("/")
def root():
    return {"message": "HyperVision KYC AI API", "docs": "/docs", "health": "/health"}
