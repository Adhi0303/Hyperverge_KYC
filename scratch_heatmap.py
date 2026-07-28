import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference.predictor import load_model, predict_single

def test_heatmap():
    img_path = r"d:\Projects\Hyperverge\HyperVision KYC AI\src\data\archive\student\student\images\test\test_00001.jpg"
    bgr = cv2.imread(img_path)
    model = load_model("best_model.pth")
    
    # Get raw probability map
    mask, prob = predict_single(model, bgr, thresh=0.1) # Try a much lower threshold
    
    # Save probability heatmap
    heatmap = (prob * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite("heatmap_test_00001.jpg", heatmap_colored)
    
    # Save mask with lower threshold
    from src.postprocessing.polygon import mask_to_polygons
    h, w = bgr.shape[:2]
    polys = mask_to_polygons(mask, size=(w, h), min_area=200)
    print(f"Detected {len(polys)} polygons with thresh 0.1")
    
if __name__ == "__main__":
    test_heatmap()
