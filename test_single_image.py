import sys
import os
import cv2
import json

# Add project root to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.inference.predictor import load_model, predict_single
from src.postprocessing.polygon import mask_to_polygons
from backend.main import _compute_quality, _poly_area, _perspective_correct

def test_image(img_path):
    print(f"Loading image from: {img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("Failed to load image.")
        return
    
    print("Loading model...")
    model = load_model("best_model.pth")
    
    print("Running inference...")
    h, w = bgr.shape[:2]
    mask, prob = predict_single(model, bgr)
    
    polys = mask_to_polygons(mask, size=(w, h), min_area=200)
    confidence = float(prob[mask == 1].mean()) if mask.sum() > 0 else 0.0
    quality = _compute_quality(bgr, confidence)
    
    results = {
        "filename": os.path.basename(img_path),
        "has_document": len(polys) > 0,
        "polygons": polys,
        "confidence": round(confidence, 4),
        "quality": quality,
    }
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=2))
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = r"d:\Projects\Hyperverge\HyperVision KYC AI\src\data\archive\student\student\images\test\test_00347.jpg"
    test_image(img_path)
