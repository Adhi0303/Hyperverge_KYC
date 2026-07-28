import cv2
import numpy as np

from src.config.settings import IMG_SIZE

def mask_to_polygons(binary, size=None, min_area=80, epsilon_frac=0.015):
    """Convert a binary mask into a list of normalized polygons - the format the
    submission CSV expects in its `polygon` column.

    size: int (square image) OR (w, h) tuple for non-square images.
          Polygon x coords are divided by w, y coords divided by h → [0, 1].
    """
    if size is None:
        w = h = IMG_SIZE
    elif isinstance(size, (tuple, list)):
        w, h = size
    else:
        w = h = size

    # Apply morphological closing to fill large holes in the mask predictions.
    # We use a dynamic kernel size (e.g., 3% of the image max dimension) so it works on high-res images.
    k_size = max(9, int(max(w, h) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    binary_closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_closed,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue                                      # skip tiny noise blobs
            
        # Documents are generally convex (quadrilaterals). Taking the convex hull
        # smooths out inward-jagged edges caused by noisy predictions on text.
        hull = cv2.convexHull(contour)
        
        epsilon = epsilon_frac * cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, epsilon, closed=True)
        if len(approx) < 3:
            continue                                      # need at least 3 points for a polygon
        pts = approx.reshape(-1, 2).astype(np.float32)
        # Normalize x by width, y by height → all values in [0, 1]
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)
        polygons.append([[round(float(x), 5), round(float(y), 5)] for x, y in pts])
    return polygons
