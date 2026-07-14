"""
Quick visual sanity check.
Draws predicted polygons from pred.csv over 5 random test images.
Saves images to: data/predictions/visual_check/
"""
import os, sys, json, random
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.settings import TEST_IMG_DIR, PRED_ROOT

OUT_DIR = os.path.join(PRED_ROOT, "visual_check")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PRED_ROOT, "pred.csv"))
sample = df.sample(5, random_state=42)

for _, row in sample.iterrows():
    fname  = row["image"]
    polys  = json.loads(row["polygon"])

    img_path = os.path.join(TEST_IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {fname}")
        continue

    h, w = img.shape[:2]

    # Draw each polygon
    for poly in polys:
        # Denormalize from [0,1] back to pixel coords
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 80), thickness=3)
        # Fill with semi-transparent green overlay
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 200, 80))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        # Draw corner dots
        for pt in pts:
            cv2.circle(img, tuple(pt), 6, (0, 80, 255), -1)

    out_path = os.path.join(OUT_DIR, f"check_{fname}")
    cv2.imwrite(out_path, img)
    print(f"[OK] {fname}  ->  {len(polys)} polygon(s)  ->  saved to {out_path}")

print(f"\nDone! Open: {OUT_DIR}")
