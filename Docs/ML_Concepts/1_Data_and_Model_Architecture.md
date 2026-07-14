# ML Concepts: Data & Model Architecture

This document explains the Machine Learning concepts behind the modules we just built: Data Loading, Semantic Segmentation, and the U-Net Architecture.

---

## 1. The Core Problem: Semantic Segmentation

The task of finding the exact boundaries of a document within an image is modeled as a **Semantic Segmentation** problem. 

### What is Semantic Segmentation?
Unlike Image Classification (which says "this image contains a document") or Object Detection (which draws a bounding box around the document), Semantic Segmentation classifies **every single pixel** in the image.

For every pixel in an input image, the model asks: 
> *Is this pixel part of the document (Foreground - 1) or part of the background (Background - 0)?*

The output is a **Binary Mask** of the same width and height as the input image, where white pixels (1) represent the document and black pixels (0) represent the background.

---

## 2. Data Preparation: The Dataset Module (`src/data/dataset.py`)

A neural network cannot natively understand a JSON string of polygon coordinates. It only understands tensors (grids of numbers).

### From Polygons to Masks
Our `CsvSegDataset` takes the raw coordinates provided by the annotators and uses OpenCV to "rasterize" them.
1. The polygon points are scaled to the actual image dimensions.
2. `cv2.fillPoly` draws this polygon onto a blank canvas, filling the inside with white pixels (255) and leaving the outside black (0).
3. This creates the "Ground Truth" (GT) binary mask that we will train the model to replicate.

### Tensors and Normalization
Before the image and mask are fed to the model:
- **Resizing:** The images are resized to `256x256` (`IMG_SIZE`).
- **Normalization:** The RGB pixels (0-255) are scaled to [0, 1] and then normalized using ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. This centers the data around zero, which helps the neural network's gradients flow smoothly during training.
- **Tensors:** The data is converted to PyTorch Tensors `(1, 3, 256, 256)` where 3 represents the RGB channels.

---

## 3. The Model: U-Net (`src/models/unet.py`)

To solve the semantic segmentation problem, we use a Convolutional Neural Network (CNN) architecture called **U-Net**.

### Why U-Net?
Originally designed for biomedical image segmentation, U-Net is extremely effective at finding precise boundaries even with a small amount of training data. 

### The Architecture
It is called U-Net because it has a symmetric "U" shape consisting of two parts:

#### A. The Encoder (Contracting Path)
- **Goal:** To understand "WHAT" is in the image (context).
- **How it works:** It applies sequences of 3x3 Convolutions (`DoubleConv`) followed by MaxPooling. 
- **The Math:** Each MaxPooling operation halves the spatial dimensions (e.g., 256 -> 128 -> 64) but doubles the number of feature channels (e.g., 32 -> 64 -> 128). The network loses exact spatial resolution but gains deep semantic understanding of the document's features (corners, text, edges, shadows).

#### B. The Decoder (Expanding Path)
- **Goal:** To understand "WHERE" the document is (localization).
- **How it works:** It uses `ConvTranspose2d` to upsample the tiny, deep feature maps back to the original `256x256` resolution.
- **The "Skip Connections" (The Magic):** As the decoder upsamples, it **concatenates** the feature map from the corresponding level of the Encoder. 
  - *Math:* `torch.cat([upsampled_features, encoder_features], dim=1)`
  - This allows the network to combine the deep semantic context from the bottleneck with the precise spatial details that the encoder saw before pooling, resulting in highly accurate boundary predictions.

### The Output
The final layer is a 1x1 Convolution (`nn.Conv2d(base, 1)`) that collapses the feature channels down to a single channel. This outputs a "logit" for every pixel. 

---

## 4. The Loss Function: Soft Dice Loss (`src/training/losses.py`)

To train the network, we need to mathematically quantify how "wrong" its prediction is compared to the ground truth mask. We use the **Dice Loss**.

### The Dice Coefficient (The Metric)
The Dice Coefficient measures the overlap between two samples:
> **Dice = 2 * |Intersection| / (|Prediction| + |Ground Truth|)**

- If the prediction perfectly matches the ground truth, Dice = 1.0.
- If there is no overlap, Dice = 0.0.

### Differentiable Soft Dice (The Loss)
Because neural networks train using backpropagation and gradient descent, the loss function must be differentiable. A hard binary threshold (0 or 1) has no gradient, so we use a "Soft" Dice:

1. **Sigmoid:** We apply a Sigmoid function `torch.sigmoid(logits)` to convert the model's raw logits into probabilities between [0, 1].
2. **Soft Intersection:** Instead of counting overlapping pixels, we multiply the probability array by the ground truth array and sum them.
3. **Loss:** We subtract the result from 1. 
  - `Loss = 1 - Soft_Dice`
  - The optimizer will update the U-Net weights to minimize this loss (push it towards 0).

---

## 5. Post Processing: Masks to Polygons (`src/postprocessing/polygon.py`)

Once the model predicts a mask, we must convert it back to polygon coordinates to satisfy the downstream Digital KYC pipeline.

1. **Thresholding:** We convert the probabilities back to binary `(prob > 0.5)`.
2. **Contour Detection:** We use OpenCV's `cv2.findContours` algorithm (specifically Suzuki's algorithm for topological structural analysis) to find the outer boundary of the white blob.
3. **Approximation:** Real boundaries are jagged. We use `cv2.approxPolyDP` (the Ramer-Douglas-Peucker algorithm) to simplify the curve. It drops points that lie close to a straight line segment, leaving us with a clean polygon (ideally 4 corners) that outlines the document.
