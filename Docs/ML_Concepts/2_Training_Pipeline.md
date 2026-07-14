# ML Concepts: Training Pipeline

This document explains the Machine Learning and mathematical concepts behind the training pipeline we just built for HyperVision KYC AI.

---

## 1. Data Augmentation (`src/data/transforms.py`)

### Why Augment?
A model trained only on the raw dataset will memorize the training images ("overfit") and fail on new, unseen images captured in different lighting or angles. Augmentation **artificially inflates the dataset** by applying random transforms to training images, forcing the model to learn **general features** rather than memorizing specific pixels.

### Key Augmentations We Use

| Augmentation | What it simulates |
|---|---|
| `HorizontalFlip` | Document photographed from the left vs. right |
| `ShiftScaleRotate` | Camera not held perfectly level; document partially out of frame |
| `Perspective` | Looking at a document at an angle (most common real-world case) |
| `RandomBrightnessContrast` | Overexposed photos, dim lighting |
| `GaussianBlur` | Out-of-focus camera shots |
| `GaussNoise` | Low-light sensor noise |
| `RandomShadow` | A hand or object casting a shadow on the document |

> **Important:** Albumentations automatically applies the **same spatial transform to both the image AND the mask**. If the image is flipped, the mask is flipped too. This is critical — otherwise the model learns wrong targets.

---

## 2. The Combined Loss Function

We use a combined loss: **0.5 × BCE + 0.5 × Dice**

### A. Binary Cross-Entropy (BCE)
BCE asks, for every single pixel independently: *"How confident was the model? Was it right?"*

> **Formula:**  `Loss = -[y · log(p) + (1-y) · log(1-p)]`

Where `y` is the ground truth (0 or 1) and `p` is the predicted probability.
- If the model predicts `p = 0.99` for a foreground pixel (`y = 1`): loss = `-log(0.99) ≈ 0.01` (small, good)
- If the model predicts `p = 0.01` for a foreground pixel (`y = 1`): loss = `-log(0.01) ≈ 4.6` (huge penalty)

**Weakness:** BCE treats every pixel equally. When 90% of an image is background, the model can get low BCE loss by just predicting "background" for everything.

### B. Soft Dice Loss
Dice Loss directly optimizes the **overlap ratio**, making it robust to class imbalance.

> **Formula:**  `Dice Loss = 1 - (2 × |A ∩ B| + ε) / (|A| + |B| + ε)`

**Weakness:** Dice alone can be unstable early in training when predictions are near-random.

### Why Combine Both?
BCE gives **stable, per-pixel gradients** in early training. Dice **pushes the overlap metric higher** in later training. Together: `0.5 × BCE + 0.5 × Dice` gets the best of both worlds.

---

## 3. Optimizer: AdamW

AdamW is an improved version of the famous Adam optimizer.

### How Adam Works
Adam tracks a **moving average of the gradients** (momentum) and a **moving average of squared gradients** (adaptive learning rates). This means each weight in the model gets its own personal learning rate, making training much faster than vanilla SGD.

> **Update Rule:**  
> `m = β₁ × m + (1-β₁) × gradient`  (momentum)  
> `v = β₂ × v + (1-β₂) × gradient²` (variance)  
> `weight = weight - lr × m / (√v + ε)`

### The "W" in AdamW: Weight Decay
Standard Adam applies L2 regularization (penalizes large weights) incorrectly — it couples it with the adaptive learning rate, which reduces its effectiveness. **AdamW decouples weight decay** from the gradient update, applying it directly to the weights:

> `weight = weight × (1 - lr × λ)  -  lr × gradient_update`

This prevents the model from learning overly large weights that cause overfitting. We use `weight_decay = 1e-4`.

---

## 4. Learning Rate Scheduler: CosineAnnealingLR

Instead of keeping the learning rate constant throughout training, we **anneal** (reduce) it following a cosine curve.

### The Math
> `lr(epoch) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / T_max))`

**Why cosine?**
- A **high learning rate** at the start lets the model explore the loss landscape quickly.
- A **decaying learning rate** later allows the model to settle into a precise minimum.
- The cosine curve is smooth — no sharp drops that cause the loss to spike.

We go from `lr_max = 3e-4` down to `lr_min = 1e-6` over 50 epochs.

---

## 5. Automatic Mixed Precision (AMP)

PyTorch's `torch.amp.GradScaler` allows the model to use **16-bit floating point (FP16)** for most computations while keeping critical operations in **32-bit (FP32)**.

### Why This Matters for Your RTX 4050
- FP16 operations run **~2x faster** on NVIDIA Ampere GPUs (RTX 40xx series).
- FP16 uses **half the GPU memory**, allowing us to use a larger batch size.
- The `GradScaler` automatically scales the loss to prevent FP16 underflow (where gradients become too small to represent).

---

## 6. Checkpoint Manager (`src/training/checkpoint.py`)

We save the model **only when validation Dice improves**. This is called "saving the best model" and prevents overfitting from harming us even if training runs too long. The checkpoint stores:
- `model.state_dict()` — all the learned weights (millions of numbers)
- `optimizer.state_dict()` — the optimizer's momentum history (to resume training smoothly)
- `epoch` and `val_dice` — metadata for our records

---

## 7. MLflow Experiment Tracking

Every training run logs the following to MLflow:
- **Per-epoch metrics:** `train_loss`, `val_loss`, `train_dice`, `val_dice`, `lr`
- **Hyperparameters:** all settings recorded once at run start
- **Artifacts:** the `best_model.pth` checkpoint file

This allows us to compare multiple experiments side-by-side (e.g., batch_size=8 vs 16, or with/without augmentation) and always know which run produced the best result.
