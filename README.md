# 🫀 Liver and Tumor Segmentation (LiTS) via 2D Cascade U-Net

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

## 📌 Project Overview
This project tackles the complex medical imaging task of automatically segmenting the liver and hepatocellular carcinomas (liver tumors) from 3D abdominal CT scans. Using the **LiTS (Liver Tumor Segmentation Challenge)** dataset, this repository explores and compares two deep learning paradigms:
1.  A standard **Single-Stage** multi-class segmentation approach.
2.  A highly optimized **Two-Stage Cascade** approach designed to combat the severe foreground-background class imbalance inherent in medical imaging.

By cropping the region of interest (ROI) around the liver before searching for tumors, the cascade model significantly improves the signal-to-noise ratio, leading to fewer false positives and better detection of small lesions.

---

## 💾 Dataset & Preprocessing

The raw data consists of heavy 3D NIfTI (`.nii`) volumes. To process these efficiently in a 2D network, the following preprocessing pipeline is applied slice-by-slice:

* **Foreground Filtering:** To prevent the model from collapsing by looking at empty space, only slices containing the liver (Class 1) or tumors (Class 2) are fed into the network.
* **Voxel Resampling:** CT scans vary in physical resolution depending on the scanner. We use OpenCV to resample all slices to a uniform **1.0 x 1.0 mm** pixel spacing, ensuring anatomical consistency.
* **CT Windowing:** Soft tissues are difficult to differentiate in raw CT values. We apply a specific abdominal Hounsfield Unit (HU) window (**Width = 350, Level = 40**). This normalizes the data, clamping out dense bone and air to visually isolate the liver parenchyma and lesions.
* **Memory Management:** Implements an on-the-fly caching mechanism to load massive `.nii` files without triggering Out-Of-Memory (OOM) errors on Kaggle.

---

## 🧠 Model Architectures

Two custom convolutional architectures are implemented from scratch:

* **U-Net:** A classic symmetric encoder-decoder network. The encoder captures contextual features via max-pooling, while the decoder uses transposed convolutions and skip-connections to recover spatial resolution.
* **U-Net++ (Variant):** An adjusted architecture utilizing modified blocks with localized skip-connections for deeper feature extraction.

Both networks output raw logits across 3 channels (Background, Liver, Tumor).

---

## 🚀 Training Strategy: The Two-Stage Cascade

Detecting a tiny tumor in a massive abdominal scan is like finding a needle in a haystack. This project solves this using a "zoom-in" cascade approach:

### Stage 1: Liver Localization
* **Goal:** Differentiate Liver vs. Background.
* **Dataset:** Slices containing the liver (`LiverDataset`).
* **Output:** A binary mask of the liver. We extract a bounding box (with a 10-pixel padding) around this predicted mask to create our ROI.

### Stage 2: Tumor Segmentation
* **Goal:** Differentiate Tumor vs. Background *strictly inside the liver ROI*.
* **Dataset:** Cropped slices containing tumors (`TumourDataset`).
* **Output:** By forcing the network to only look inside the cropped liver area, it learns the subtle texture differences between healthy tissue and tumors without being distracted by surrounding abdominal organs.

---

## ⚖️ Loss Formulation & Optimization

### Handling Class Imbalance
The background accounts for >90% of the pixels, the liver ~8%, and tumors often <1%. To prevent the model from simply predicting "Background" everywhere, we use a custom composite loss function:

$$Loss = 0.5 \cdot \text{CE}_{weighted} + 0.5 \cdot \text{Dice}$$

* **Weighted Cross-Entropy (CE):** Applies heavy penalties for missing the foreground. The weights are set to **Background: 0.05, Liver: 1.0, Tumor: 3.0**.
* **Dice Loss:** Provided by `segmentation_models_pytorch`, this metric optimizes directly for spatial overlap (Intersection over Union).
* **Optimizer & Scheduler:** Trained using the **Adam optimizer** (`lr=1e-4`) and a **ReduceLROnPlateau** scheduler that halves the learning rate if the validation F1 score stagnates.

---

## 🌪️ Data Augmentation
To simulate natural anatomical variations and prevent overfitting, the `albumentations` library applies robust transformations during training:
* **Spatial:** Horizontal/Vertical Flips, Random 90° Rotations, Affine transforms (translation, scaling, rotation).
* **Pixel-level:** Elastic Transformations (vital for simulating soft-tissue deformation) and Random Brightness/Contrast shifts to simulate different CT scanner calibrations.

---

## 🧹 Post-Processing

Raw neural network predictions often contain artifacts. We apply classical morphological operations (`skimage.morphology`, `scipy.ndimage`) to clean the masks:
* **Liver (Class 1):** Removes objects smaller than 500 pixels. Isolates the largest connected component to drop scattered false positives.
* **Tumor (Class 2):** Applies binary closing (with a disk structural element) to bridge gaps within fragmented predictions. Removes noise/holes smaller than 30 pixels.

---

## 📊 Evaluation
Models are evaluated on a held-out test set. The pipeline tracks:
1.  **F1-Score:** Calculated independently for Liver and Tumor classes.
2.  **Confusion Matrices:** Generated before and after post-processing to visualize the reduction in false positives and false negatives.
3.  **Visual Overlays:** Generates side-by-side plots of the raw CT, Ground Truth, Raw Prediction, Post-processed Prediction, and Difference map.


### Run on Kaggle
You can view the full code, including the preprocessing pipeline and the model architecture, directly on Kaggle. No local installation is required!

👉 **[View and run the Expression-DCGAN notebook here](https://www.kaggle.com/code/zacxxx/liver-tumor-segmentation-cascade-postprocessing)**
