import sys
import os
import cv2
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.inference.predictor import load_model, predict_single
from src.postprocessing.polygon import mask_to_polygons

def test_image_visual(img_path):
    print(f"Loading image from: {img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("Failed to load image.")
        return
    
    print("Loading model...")
    model = load_model("best_model.pth")
    
    print("Running inference...")
    h, w = bgr.shape[:2]
    mask, prob = predict_single(model, bgr, thresh=0.15)
    
    polys = mask_to_polygons(mask, size=(w, h), min_area=200)
    
    print(f"Detected {len(polys)} polygon(s). Drawing borders...")
    
    # Draw each polygon on the image
    for poly in polys:
        # Denormalize from [0,1] back to pixel coords
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
        
        # Draw thick green border
        cv2.polylines(bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
        
        # Draw small red dots at each corner
        for pt in pts:
            cv2.circle(bgr, tuple(pt), 6, (0, 0, 255), -1)

    # Save output
    out_name = f"manual_output_{os.path.basename(img_path)}"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    cv2.imwrite(out_path, bgr)
    
    print(f"\nSuccess! Visual output saved to: {out_path}")
    print("Upload the original image to your frontend and compare the borders visually!")

if __name__ == "__main__":
    img_path = r"d:\Projects\Hyperverge\HyperVision KYC AI\src\data\archive\student\student\images\test\test_00812.jpg"
    test_image_visual(img_path)
