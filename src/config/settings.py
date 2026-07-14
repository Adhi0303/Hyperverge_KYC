import os
import torch

# Base project directory (assuming this file is in src/config/settings.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Dataset Paths (dynamically pointing to the loaded dataset)
DATA_ROOT = os.path.join(PROJECT_ROOT, "src", "data", "archive", "student", "student")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images", "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "images", "test")

# Annotations (Defaulting to round 1 as agreed)
TRAIN_CSV = os.path.join(DATA_ROOT, "labels", "train_round_1.csv")

# Output Paths
PRED_ROOT = os.path.join(PROJECT_ROOT, "data", "predictions")
CKPT_ROOT = os.path.join(PROJECT_ROOT, "data", "checkpoints")

# Model and Data Constants
IMG_SIZE = 256                       # square working resolution (divisible by 8)
IOU_THR  = 0.75                      # overlap ratio to count a predicted instance as a match
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.485, 0.456, 0.406)         # ImageNet channel means used to normalise inputs
STD  = (0.229, 0.224, 0.225)         # ImageNet channel std-devs
