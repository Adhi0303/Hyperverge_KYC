import os
import glob
import cv2
import numpy as np
import torch

from src.config.settings import DATA_ROOT, IMG_SIZE, MEAN, STD, DEVICE

def preprocess(img_rgb):
    """Turn an RGB image into the model's input tensor: resize to IMG_SIZE,
    scale to [0, 1], apply ImageNet Normalize, and reorder to (1, 3, H, W)."""
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    chw = normalised.transpose(2, 0, 1)                   # HWC -> CHW
    return torch.from_numpy(chw).unsqueeze(0).float()     # add the batch axis -> 1CHW


@torch.no_grad()
def save_test_predictions(model, pred_dir, thresh=0.5):
    """Run the model over every test image and save one binary mask PNG per image
    (IMG_SIZE, values {0, 255}) into `pred_dir` - the standard format test.py
    reads. Returns the number of images written."""
    os.makedirs(pred_dir, exist_ok=True)
    model.eval().to(DEVICE)

    image_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "images", "test", "*.jpg")))
    for image_path in image_paths:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        x = preprocess(image).to(DEVICE)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        mask = (prob > thresh).astype(np.uint8) * 255     # {0,1} -> {0,255} for a viewable PNG

        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(pred_dir, stem + ".png"), mask)

    return len(image_paths)
