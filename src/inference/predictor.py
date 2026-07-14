"""
src/inference/predictor.py
===========================
Single-image and batch inference utilities used by both the submission
generator (scripts/generate_submission.py) and the FastAPI backend.
"""

import os
import warnings
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

from src.config.settings import CKPT_ROOT, DEVICE, MEAN, STD

MEAN_arr = np.array(MEAN, dtype=np.float32)
STD_arr  = np.array(STD,  dtype=np.float32)
IMG_SIZE  = 384   # must match training resolution


def load_model(checkpoint: str = "best_model.pth") -> torch.nn.Module:
    """Load the UNet++ EfficientNet-B0 model from the saved checkpoint."""
    path = os.path.join(CKPT_ROOT, checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train the model first."
        )

    model = smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b0",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)
    return model


def preprocess_image(bgr_img: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV image to a normalised (1, 3, H, W) CUDA tensor."""
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    norm    = (resized.astype(np.float32) / 255.0 - MEAN_arr) / STD_arr
    tensor  = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor


@torch.no_grad()
def predict_single(model: torch.nn.Module,
                   bgr_img: np.ndarray,
                   thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single BGR image.

    Returns
    -------
    mask : uint8 ndarray at the ORIGINAL image resolution (values 0 or 1)
    prob : float32 probability map at original resolution
    """
    h, w = bgr_img.shape[:2]
    tensor = preprocess_image(bgr_img)

    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        logits = model(tensor)

    prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > thresh).astype(np.uint8)
    return mask, prob
