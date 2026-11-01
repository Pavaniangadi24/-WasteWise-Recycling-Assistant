# Phase 5: Test Set Evaluation and Model Benchmarking Report

**Project:** WasteWise Recycling Assistant  
**Model Checkpoint:** `models/best_mobilenetv2.keras` (11.64 MB)  
**Evaluation Script:** `src/evaluate.py`  
**Test Suite:** `tests/test_evaluate.py`  
**Date:** November 1, 2026  
**Status:** Evaluation Completed & Benchmarked on Held-Out Test Set  

---

## 1. Executive Summary

Phase 5 performed the definitive, unbiased benchmark evaluation of the trained `WasteWise_MobileNetV2` model against the previously isolated test dataset of 2,513 images (1,401 Organic, 1,112 Recyclable). In strict adherence to scientific evaluation methodology, the test set was never seen during data preprocessing, architecture design, training, or hyperparameter selection. The model was evaluated using standard inference mode at the canonical 0.5 decision threshold without post-hoc threshold tuning or data modification.

The model achieved an **overall test accuracy of 90.65%**, a **ROC-AUC of 0.9866**, and an **Average Precision of 0.9824**, demonstrating exceptional generalization on real-world waste imagery.

---

## 2. Model and Checkpoint Details

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | `WasteWise_MobileNetV2` |
| **Model Architecture** | MobileNetV2 base (154 layers, frozen) + Custom Classification Head |
| **Pretrained Weights** | Official ImageNet MobileNetV2 weights (9.4 MB) |
| **Checkpoint Path** | `models/best_mobilenetv2.keras` |
| **Checkpoint Size** | 11.64 MB |
| **Checkpoint Origin** | Restored from best validation loss epoch (Epoch 7) during Phase 4 training |
| **Total Parameters** | 2,760,517 (Base: 2,257,984; Head: 169,217; Optimizer states: 333,316) |
| **Input Shape & Preprocessing** | 224 × 224 × 3 RGB, scaled to `[-1.0, 1.0]` |
| **Decision Threshold** | 0.50 (Unmodified canonical threshold) |

---

## 3. Test Dataset Distribution

The test set was partitioned prior to Phase 2 and kept completely untouched until this evaluation:

| Class Label | Class Name | Test Sample Count | Proportion | Verification Status |
| :---: | :--- | :---: | :---: | :---: |
| `0` | **Organic (`O`)** | 1,401 | 55.75% | Verified exact match |
| `1` | **Recyclable (`R`)** | 1,112 | 44.25% | Verified exact match |
| **Total** | — | **2,513** | **100.00%** | **100% accounted for** |

---

## 4. Comprehensive Evaluation Metrics

### 4.1 Overall Performance Summary

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Loss (Binary Cross-Entropy)** | `0.2604` | — |
| **Overall Accuracy** | `0.9065` | **90.65%** |
| **Precision (Recyclable)** | `0.9813` | **98.13%** |
| **Recall (Recyclable)** | `0.8040` | **80.40%** |
| **F1-Score (Recyclable)** | `0.8838` | **88.38%** |
| **Specificity (Organic Recall)** | `0.9879` | **98.79%** |
| **ROC-AUC** | `0.9866` | **98.66%** |
| **Average Precision (PR-AUC)** | `0.9824` | **98.24%** |
| **Macro Average F1** | `0.9028` | **90.28%** |
| **Weighted Average F1** | `0.9050` | **90.50%** |

---

### 4.2 Confusion Matrix Analysis

The confusion matrix on the 2,513 test samples is:

$$\begin{pmatrix} \text{TN} = 1,384 & \text{FP} = 17 \\ \text{FN} = 218 & \text{TP} = 894 \end{pmatrix}$$

| Metric | True Organic (`O`, 0) | True Recyclable (`R`, 1) | Total Predicted |
| :--- | :---: | :---: | :---: |
| **Predicted Organic (`0`)** | **1,384** (True Negative) | **218** (False Negative) | 1,602 |
| **Predicted Recyclable (`1`)** | **17** (False Positive) | **894** (True Positive) | 911 |
| **Total Ground Truth** | **1,401** | **1,112** | **2,513** |
| **Class Recall Rate** | **98.79%** | **80.40%** | Overall: **90.65%** |

- **True Negatives (TN):** **1,384** / 1,401 Organic samples correctly recognized (**98.79%** accuracy on Organic waste).
- **False Positives (FP):** **17** / 1,401 Organic samples misclassified as Recyclable (**1.21%** error rate).
- **False Negatives (FN):** **218** / 1,112 Recyclable samples misclassified as Organic (**19.60%** error rate).
- **True Positives (TP):** **894** / 1,112 Recyclable samples correctly recognized (**80.40%** recall).

---

### 4.3 Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **Organic (`0`)** | **86.39%** | **98.79%** | **92.17%** | 1,401 |
| **Recyclable (`1`)** | **98.13%** | **80.40%** | **88.38%** | 1,112 |
| **Macro Average** | **92.26%** | **89.59%** | **90.28%** | 2,513 |
| **Weighted Average** | **91.59%** | **90.65%** | **90.50%** | 2,513 |

---

## 5. Visual Artifacts Generated

The following high-resolution evaluation figures and output files were generated and saved in `outputs/`:
1. **Confusion Matrix Heatmap:** `outputs/confusion_matrix.png` (Annotated with raw counts and normalized percentages).
2. **ROC Curve:** `outputs/roc_curve.png` (ROC curve showing $\text{AUC} = 0.9866$ against the random baseline).
3. **Precision-Recall Curve:** `outputs/precision_recall_curve.png` (PR curve showing $\text{Average Precision} = 0.9824$).
4. **Predictions Table:** `outputs/test_predictions.csv` (2,513 rows containing filepath, true label, predicted probability, predicted label, and classification correctness).
5. **Evaluation Summary JSON:** `outputs/test_evaluation_summary.json` (Structured machine-readable metrics).

---

## 6. Interpretation of Results & Real-World Utility

### 6.1 Extremely High Purity for Recyclable Streams
When the model predicts an item is **Recyclable**, it is correct **98.13% of the time** (only 17 non-recyclables misclassified). In industrial recycling facility operations, **cross-contamination of recyclable materials by wet/organic waste** is the single costliest failure mode, as organic contamination can ruin an entire batch of paper, cardboard, or plastic. A precision of 98.13% provides high operational confidence that items routed to the recycling stream are genuinely recyclable.

### 6.2 Near-Perfect Organic Waste Retention
The model captures **98.79% of all Organic waste** (1,384 out of 1,401 items). This ensures that biological compostable waste is almost completely diverted from landfills into appropriate composting streams.

### 6.3 Conservative Recycling Attribution
The 218 False Negatives represent recyclable items (e.g., crushed, soiled, or translucent plastic/glass) that the model cautiously classified as Organic. Because the model's head was trained with frozen ImageNet feature weights, fine-tuning top convolutional layers in future phases can provide the specialized edge and texture representations necessary to boost recyclable recall even further.

---

## 7. Limitations and Observations

1. **Frozen Base Trade-Off:** The feature extractor was completely frozen throughout training. While this achieved an outstanding 90.65% accuracy and 0.9866 AUC, unfreezing top inverted residual blocks during a subsequent fine-tuning stage could further reduce False Negatives.
2. **Complex / Soiled Items:** Manual inspection of misclassified samples in `outputs/test_predictions.csv` indicates that edge cases primarily involve food-stained containers, transparent plastic wrapping, or items with mixed organic and recyclable parts.
3. **Hardware Execution:** The evaluation ran across all 2,513 images in approximately 44 seconds on the 20-core Intel CPU, proving that MobileNetV2 is lightweight and well-suited for edge and local web deployment without requiring expensive GPU infrastructure.

---

## 8. Unit Test Suite Status

Executed `python -m unittest discover tests`:

```text
Ran 25 tests in 32.364s

OK
```

- `tests/test_data_loader.py`: **9 / 9 Passed**
- `tests/test_model.py`: **9 / 9 Passed**
- `tests/test_train.py`: **3 / 3 Passed**
- `tests/test_evaluate.py`: **4 / 4 Passed**
- **Total: 25 / 25 tests passing.**
