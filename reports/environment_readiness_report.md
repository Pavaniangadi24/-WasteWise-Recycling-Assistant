# Phase 3A: Environment and Model Readiness Report

**Project:** WasteWise Recycling Assistant  
**Date of Inspection:** November 1, 2026  
**Git Commit:** `fac3b9e` (Phase 1 and 2: Dataset inspection and preprocessing)  
**Status:** Inspection Complete — Ready for MobileNetV2 Implementation  

---

## 1. Environment & Hardware Resource Inspection

### 1.1 Operating System & Python Environment
- **Operating System:** Windows 11 (OS Build 10.0.26300, 64-bit AMD64)
- **Python Executable:** `C:\Users\pavan\AppData\Local\Programs\Python\Python312\python.exe`
- **Python Version:** 3.12.6 (tags/v3.12.6:a4a2d2b, MSC v.1940 64 bit)

### 1.2 CPU & System Memory
- **Processor:** Intel Core Processor (Family 6 Model 183, 13th/14th Gen architecture)
- **Logical Cores:** 20 CPU threads
- **Instruction Accelerators:** SSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX_VNNI, FMA, and Intel oneDNN custom operations enabled
- **Total Physical RAM:** 15.71 GB
- **Available Physical RAM:** ~4.04 GB (Memory load ~74%)

### 1.3 GPU Hardware & TensorFlow Detection
- **Physical Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MiB VRAM / 6 GB)
- **NVIDIA Driver:** 581.86
- **System CUDA Support:** CUDA 13.0
- **TensorFlow GPU Visibility:** `[]` (0 GPUs visible)
- **Built with CUDA:** `False`
- **Technical Root Cause:** Starting with TensorFlow 2.11+, official TensorFlow wheels for native Windows are built strictly for CPU. Native Windows GPU acceleration is not supported by upstream TensorFlow without WSL2 (Windows Subsystem for Linux) or DirectML. 
- **Performance Mitigation:** The 20-core Intel CPU with oneDNN multi-threading and AVX2 vector extensions provides robust throughput for lightweight architectures like MobileNetV2.

---

## 2. Dependency Status

All required machine learning, computer vision, data manipulation, and visualization libraries are already installed and verified:

| Package | Installed Version | Status | Purpose |
| :--- | :---: | :---: | :--- |
| `tensorflow` | **2.20.0** | Verified | Model building, training, evaluation, graph operations |
| `keras` | **3.12.0** | Verified | High-level deep learning API |
| `numpy` | **1.26.4** | Verified | Numerical operations and array manipulation |
| `scikit-learn` | **1.5.1** | Verified | Stratified sampling, metrics (confusion matrix, ROC) |
| `pillow` | **12.0.0** | Verified | Image loading, verification, and RGB conversions |
| `pandas` | **2.2.2** | Verified | Data summaries, reporting, and tabular metrics |
| `matplotlib` | **3.11.0** | Verified | Loss/accuracy curves and confusion matrix plotting |
| `seaborn` | **0.13.2** | Verified | Statistical chart rendering |

No package installations are required.

---

## 3. Preprocessing Compatibility & Double-Scaling Prevention

### 3.1 Pipeline Audit (`src/data_loader.py`)
- Images are resized to **224 × 224 pixels**.
- Explicit RGB conversion safely handles both standard JPEGs and all 372 palette-encoded PNGs.
- `src/data_loader.py` applies `preprocess_input` (scaling raw $[0, 255]$ pixels to $[-1.0, 1.0]$) inside `preprocess_image_tf` and `load_and_preprocess_image_pil`.

> [!WARNING]
> **Double Preprocessing Hazard:** Because `src/data_loader.py` already outputs tensors in the range `[-1.0, 1.0]`, the MobileNetV2 model architecture **MUST NOT** include another `preprocess_input` layer or a `Rescaling(1./127.5, offset=-1.0)` layer. Applying normalization twice would collapse pixel intensities into approximately `[-1.008, -0.992]`, destroying input signal fidelity.

### 3.2 Dataset Integrity & Isolation
- **Train subset:** 18,051 images (Organic: 10,052 [55.69%], Recyclable: 7,999 [44.31%])
- **Validation subset:** 4,513 images (Organic: 2,513 [55.68%], Recyclable: 2,000 [44.32%])
- **Test set:** 2,513 images (Organic: 1,401 [55.75%], Recyclable: 1,112 [44.25%])
- Total verified: 25,077 images. Test set remains strictly isolated.

---

## 4. MobileNetV2 ImageNet Pretrained Weights Status

- When instantiating `tf.keras.applications.MobileNetV2(weights="imagenet")`, Keras searches for `mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5` in `.keras/models/`.
- If absent, it attempts to download the weights file from `https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/`.
- Under standard sandbox execution without outbound network access, downloading is blocked.
- In Phase 3B, downloading this weights file once via network bypass or placing it in `.keras/models/` will enable the ImageNet pretrained weights.

---

## 5. Recommended MobileNetV2 Training Architecture

### 5.1 Model Architecture
```
Input (224 × 224 × 3, values in [-1.0, 1.0])
       │
       ▼
MobileNetV2 Base (weights='imagenet', include_top=False, input_shape=(224, 224, 3))
[Frozen during Phase 1 of training: base_model.trainable = False]
       │
       ▼
GlobalAveragePooling2D() -> 1280 dimensions
       │
       ▼
BatchNormalization()
       │
       ▼
Dropout(0.3)
       │
       ▼
Dense(128, activation='relu')
       │
       ▼
Dropout(0.2)
       │
       ▼
Dense(1, activation='sigmoid') -> Output probability P(Recyclable)
```

### 5.2 Training Parameters & Policy
- **Loss Function:** `tf.keras.losses.BinaryCrossentropy(from_logits=False)`
- **Optimizer:** `Adam(learning_rate=1e-3)` for head training; `Adam(learning_rate=1e-5)` for subsequent fine-tuning.
- **Batch Size:** `32` (balanced for 20-core CPU cache and vector throughput).
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - Area Under the ROC Curve (AUC)
- **Callbacks:**
  - `ModelCheckpoint('models/best_mobilenetv2.keras', monitor='val_auc', mode='max', save_best_only=True)`
  - `EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)`
