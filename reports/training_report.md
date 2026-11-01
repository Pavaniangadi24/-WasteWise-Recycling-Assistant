# Phase 4: Model Training Report

**Project:** WasteWise Recycling Assistant  
**Model Architecture:** `WasteWise_MobileNetV2` (Transfer Learning, Frozen Base)  
**Pretrained Weights:** Official ImageNet MobileNetV2 weights (9.4 MB)  
**Date:** November 1, 2026  
**Status:** Training Completed Successfully (Early Stopping at Epoch 10, Best Epoch 7)  

---

## 1. Executive Summary

Phase 4 executed the initial transfer learning stage for the WasteWise Recycling Assistant by training the custom classification head on top of the frozen MobileNetV2 base feature extractor. The training used the verified 80/20 stratified split (18,051 training images, 4,513 validation images) with `random_state=42`. The test set of 2,513 images was strictly preserved and never loaded or evaluated.

Training ran for **10 epochs** (out of 15 maximum). Early stopping was triggered at Epoch 10 after validation loss plateaued past **Epoch 7**. Model weights were automatically restored from the best epoch, achieving:
- **Best Validation Loss:** `0.1139`
- **Best Validation Accuracy:** `96.10%`
- **Best Validation AUC:** `0.9911`
- **Best Validation Precision:** `95.24%`
- **Best Validation Recall:** `96.00%`

The best model checkpoint is preserved in modern Keras format at `models/best_mobilenetv2.keras` (11.64 MB).

---

## 2. Training Configuration

| Parameter | Value | Rationale / Details |
| :--- | :---: | :--- |
| **Base Model** | MobileNetV2 | 154 layers, frozen (`trainable=False`), ImageNet pretrained |
| **Classification Head** | GAP $\rightarrow$ BN $\rightarrow$ Drop(0.3) $\rightarrow$ Dense(128, relu) $\rightarrow$ Drop(0.2) $\rightarrow$ Dense(1, sigmoid) | 166,657 trainable parameters |
| **Input Shape** | 224 × 224 × 3 | Scaled to `[-1.0, 1.0]` (no double normalization) |
| **Batch Size** | 32 | Optimal for 20-core CPU multi-threading |
| **Max Epochs** | 15 | Configured ceiling |
| **Optimizer** | Adam | Initial $\text{LR} = 0.001$ |
| **Loss Function** | `BinaryCrossentropy(from_logits=False)` | Standard for calibrated binary sigmoid output |
| **Learning Rate Schedule** | `ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1e-6)` | Reduced LR to $2 \times 10^{-4}$ at epoch 7, $4 \times 10^{-5}$ at epoch 10 |
| **Early Stopping** | `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` | Halted training at epoch 10, restored epoch 7 weights |
| **Checkpointing** | `ModelCheckpoint('models/best_mobilenetv2.keras', monitor='val_loss', save_best_only=True)` | Preserved best checkpoint |
| **Hardware** | Intel 20-Core CPU with AVX2 and oneDNN | CPU execution (~38.2 minutes total) |

---

## 3. Dataset Split & Test Set Isolation

| Partition | Total Images | Organic (`O`, 0) | Recyclable (`R`, 1) | % Organic | % Recyclable | Training Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | **18,051** | 10,052 | 7,999 | 55.69% | 44.31% | Gradient updates (565 batches/epoch) |
| **Validation Set** | **4,513** | 2,513 | 2,000 | 55.68% | 44.32% | Checkpoint evaluation (142 batches/epoch) |
| **Test Set** | **2,513** | 1,401 | 1,112 | 55.75% | 44.25% | **Strictly preserved & isolated (untouched)** |
| **Total** | **25,077** | **13,966** | **11,111** | **55.69%** | **44.31%** | 100% data accounted for |

---

## 4. Per-Epoch Progression

The following metrics were logged live across the 10 completed epochs:

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val Acc | Val AUC | Val Precision | Val Recall | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.2213 | 91.62% | 0.9709 | 0.1360 | 95.08% | 0.9874 | 93.97% | 95.00% | 0.001 | Checkpoint saved |
| **2** | 0.1454 | 94.36% | 0.9863 | 0.1253 | 95.08% | 0.9902 | 92.95% | 96.20% | 0.001 | Checkpoint saved |
| **3** | 0.1225 | 95.49% | 0.9902 | 0.1202 | 95.86% | 0.9899 | 94.41% | 96.35% | 0.001 | Checkpoint saved |
| **4** | 0.1070 | 95.98% | 0.9924 | 0.1157 | 95.75% | 0.9908 | 95.02% | 95.40% | 0.001 | Checkpoint saved |
| **5** | 0.0861 | 96.67% | 0.9950 | 0.1189 | 95.72% | 0.9905 | 94.05% | 96.45% | 0.001 | No improvement |
| **6** | 0.0805 | 96.89% | 0.9954 | 0.1227 | 95.70% | 0.9893 | 95.70% | 94.55% | 0.001 | LR reduced to 2e-4 |
| **7** ★ | **0.0581** | **98.01%** | **0.9976** | **0.1139** | **96.10%** | **0.9911** | **95.24%** | **96.00%** | **0.0002** | **BEST CHECKPOINT** |
| **8** | 0.0520 | 98.16% | 0.9981 | 0.1159 | 96.19% | 0.9908 | 95.70% | 95.70% | 0.0002 | No improvement |
| **9** | 0.0437 | 98.48% | 0.9987 | 0.1181 | 96.17% | 0.9899 | 95.65% | 95.70% | 0.0002 | LR reduced to 4e-5 |
| **10** | 0.0448 | 98.35% | 0.9988 | 0.1175 | 96.30% | 0.9901 | 95.44% | 96.25% | 0.00004 | Early stopped (patience 3) |

★ **Epoch 7** achieved the lowest validation loss (`0.1139`) and highest validation AUC (`0.9911`). Weights from this epoch were restored to `models/best_mobilenetv2.keras`.

---

## 5. Artifacts and Plots Generated

All generated artifacts have been persisted:
1. **Best Model Checkpoint:** `models/best_mobilenetv2.keras` (11.64 MB, ignored by Git).
2. **Epoch Metrics CSV:** `outputs/training_history.csv` (complete per-epoch metrics).
3. **Training Summary JSON:** `outputs/training_summary.json` (machine-readable metrics and best epoch record).
4. **Accuracy Curve:** `outputs/accuracy_curve.png` (training vs. validation accuracy across epochs).
5. **Loss Curve:** `outputs/loss_curve.png` (training vs. validation binary cross-entropy loss).
6. **Combined Figure:** `outputs/training_curves.png` (side-by-side accuracy and loss plots).

---

## 6. Unit Test Verification

The full test suite was executed across all components:

```
Ran 21 tests in 23.177s

OK
```

- `tests/test_data_loader.py`: 9 / 9 passed (data integrity, 224x224 RGB conversion, stratified split, test isolation).
- `tests/test_model.py`: 9 / 9 passed (architecture, frozen base, ImageNet weights, sigmoid output, `.keras` saving).
- `tests/test_train.py`: 3 / 3 passed (callbacks, plot generation, test set isolation during training).

---

## 7. Observations and Next Phase Considerations

1. **Convergence Behavior:**
   - Training converged rapidly from Epoch 1 (91.62% train, 95.08% val) to Epoch 7 (98.01% train, 96.10% val).
   - High validation AUC (0.9911) demonstrates exceptional discriminative capability between Organic and Recyclable waste.
2. **Overfitting Control:**
   - Dropout (0.3 and 0.2) and BatchNormalization prevented overfitting; validation loss remained low and steady around 0.114–0.118 without diverging.
3. **Next Steps:**
   - Phase 5: Model Evaluation & Benchmarking on the held-out Test dataset (`TEST/`, 2,513 images), generating confusion matrices, ROC curves, and class-wise classification reports.
   - Optional fine-tuning (unfreezing top MobileNetV2 layers at a low learning rate) can be explored if further gains are desired before deployment.
