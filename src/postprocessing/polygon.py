import cv2
import numpy as np

from src.config.settings import IMG_SIZE

def mask_to_polygons(binary, size=None, min_area=80, epsilon_frac=0.005):
    """Convert a binary mask into a list of normalized polygons - the format the
    submission CSV expects in its `polygon` column.

    One polygon per external contour: contours smaller than `min_area` pixels are
    dropped as noise, each contour is simplified with approxPolyDP (tolerance =
    `epsilon_frac` of its perimeter), and its points are divided by `size` so the
    coordinates land in [0, 1]. Returns a list of polygons, each itself a list of
    [x, y] points."""
    if size is None:
        size = IMG_SIZE

    contours, _ = cv2.findContours(binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue                                      # skip tiny noise blobs
        epsilon = epsilon_frac * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue                                      # need at least 3 points for a polygon
        points = approx.reshape(-1, 2).astype(np.float32) / size   # pixels -> normalized [0, 1]
        polygons.append([[round(float(x), 5), round(float(y), 5)] for x, y in points])
    return polygons
