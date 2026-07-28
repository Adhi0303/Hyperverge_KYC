"""
backend/main.py
================
FastAPI application entry point for HyperVision KYC AI.

Endpoints:
  GET  /health              — Health check + model status
  POST /predict             — Upload an image → full pipeline results
  POST /predict/base64      — Same but image sent as base64 string
  GET  /docs                — Auto-generated Swagger UI
"""

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64
import warnings
from typing import Optional

import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    version     = "2.1.0",
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
    filename:  str = "image.jpg"

class PredictionResponse(BaseModel):
    filename:      str
    has_document:  bool
    polygons:      list          # ALL detected polygons — list of [[x,y],...]
    polygon:       list          # first polygon (backward-compat alias)
    confidence:    float         # mean sigmoid probability inside the mask region
    mask_b64:      str           # base64 PNG: original image + green mask overlay
    corrected_b64: Optional[str] # base64 PNG: perspective-corrected document crop
    enhanced_b64:  Optional[str] # base64 PNG: CLAHE-enhanced version of corrected crop
    crops:         list          # list of dicts: {"polygon": list, "corrected_b64": str, "enhanced_b64": str}
    quality:       dict          # real per-image quality metrics (0-100 each)
    inference_ms:  float         # actual model inference wall-clock time in ms
    device:        str

# ── Image helpers ──────────────────────────────────────────
def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPG/PNG.")
    return img


def _encode_b64(img: np.ndarray) -> str:
    """Encode a BGR ndarray as a base64 PNG string."""
    _, buf = cv2.imencode(".png", img.astype(np.uint8))
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 pixel-space points into: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]     # top-left     (smallest x+y)
    rect[2] = pts[np.argmax(s)]     # bottom-right (largest  x+y)
    rect[1] = pts[np.argmin(diff)]  # top-right    (smallest x-y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left  (largest  x-y)
    return rect


def _perspective_correct(bgr: np.ndarray, poly_norm: list) -> Optional[np.ndarray]:
    """
    Deskew and flatten a document using a 4-point perspective transform.

    poly_norm : [[x, y], ...]  normalised [0, 1] coordinates.
    Returns the warped crop, or None when a quadrilateral cannot be formed.
    """
    if not poly_norm:
        return None
    h, w = bgr.shape[:2]
    pts  = np.array([[x * w, y * h] for x, y in poly_norm], dtype=np.float32)

    # Reduce to exactly 4 corners via convex hull + Douglas-Peucker
    if len(pts) != 4:
        hull   = cv2.convexHull(pts.reshape(-1, 1, 2))
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        pts    = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 4:
        return None

    pts  = pts[:4]
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if maxW <= 0 or maxH <= 0:
        return None

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (maxW, maxH))


def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE in LAB colour space for adaptive contrast / visibility enhancement."""
    lab        = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_enh    = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)


def _poly_area(poly: list) -> float:
    """Shoelace formula — unsigned area of a normalised polygon."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _compute_quality(bgr: np.ndarray, confidence: float) -> dict:
    """
    Compute real per-image quality metrics from the original BGR frame.
    All scores are normalised to [0, 100].
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Blur: Laplacian variance — ~500 = sharp; clamp at 500 → 100 %
    lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur       = round(min(100.0, lap_var / 5.0), 1)

    # Brightness: deviation from ideal mid-point (127.5)
    mean_lum   = float(gray.mean())
    brightness = round(max(0.0, 100.0 - abs(mean_lum - 127.5) / 127.5 * 100.0), 1)

    # Contrast: std dev of luminance; 64 std ≈ rich dynamic range
    contrast   = round(min(100.0, float(gray.std()) / 64.0 * 100.0), 1)

    # Rotation proxy: model confidence correlates with document alignment quality
    rotation   = round(min(100.0, confidence * 100.0), 1)

    # Visibility: weighted blend of sharpness and model confidence
    visibility = round(min(100.0, blur * 0.5 + confidence * 100.0 * 0.5), 1)

    return {
        "blur":       blur,
        "brightness": brightness,
        "contrast":   contrast,
        "rotation":   rotation,
        "visibility": visibility,
    }

# ── Core inference helper ──────────────────────────────────
def _build_response(filename: str, bgr: np.ndarray) -> dict:
    model = get_model()
    h, w  = bgr.shape[:2]

    # Inference with a lower threshold to prevent fragmented "swiss cheese" masks
    # on documents that the model is slightly less confident about.
    t0           = time.perf_counter()
    mask, prob   = predict_single(model, bgr, thresh=0.15)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Polygon extraction — pass (w, h) so coords are normalised relative to the
    # original frame dimensions, not the fixed training resolution.
    polys = mask_to_polygons(mask, size=(w, h), min_area=200)

    # Confidence: mean probability inside the predicted region
    confidence = float(prob[mask == 1].mean()) if mask.sum() > 0 else 0.0

    # Mask overlay: green tint on the detected document region
    overlay          = bgr.copy()
    overlay[mask==1] = (overlay[mask==1] * 0.5 + np.array([0, 200, 80]) * 0.5)
    mask_b64         = _encode_b64(overlay)

    # Process all detected polygons into separate crops
    crops_list = []
    for poly in polys:
        corr = _perspective_correct(bgr, poly)
        c_b64 = _encode_b64(corr) if corr is not None else None
        s_enh = corr if corr is not None else bgr
        e_b64 = _encode_b64(_clahe_enhance(s_enh))
        crops_list.append({
            "polygon": poly,
            "corrected_b64": c_b64,
            "enhanced_b64": e_b64
        })

    # Perspective correction: use the polygon with the largest area for backward compatibility
    poly_for_warp = max(polys, key=_poly_area) if polys else []
    corrected     = _perspective_correct(bgr, poly_for_warp)
    corrected_b64 = _encode_b64(corrected) if corrected is not None else None

    # CLAHE enhancement on the corrected crop (fall back to original if warp failed)
    src_for_enh  = corrected if corrected is not None else bgr
    enhanced_b64 = _encode_b64(_clahe_enhance(src_for_enh))

    # Quality metrics from the original image
    quality = _compute_quality(bgr, confidence)

    return {
        "filename":      filename,
        "has_document":  len(polys) > 0,
        "polygons":      polys,
        "polygon":       polys[0] if polys else [],   # backward compat
        "confidence":    round(confidence, 4),
        "mask_b64":      mask_b64,
        "corrected_b64": corrected_b64,
        "enhanced_b64":  enhanced_b64,
        "crops":         crops_list,
        "quality":       quality,
        "inference_ms":  inference_ms,
        "device":        DEVICE,
    }

# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_loaded":      _model is not None,
        "device":            DEVICE,
        "checkpoint":        os.path.join(CKPT_ROOT, "best_model.pth"),
        "checkpoint_exists": os.path.exists(os.path.join(CKPT_ROOT, "best_model.pth")),
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """Upload a document image file and receive the full pipeline results."""
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

