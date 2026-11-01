# Phase 9: Final Project Report — WasteWise Recycling Assistant

- **Project:** WasteWise Recycling Assistant
- **Repository:** `Pavaniangadi24/-WasteWise-Recycling-Assistant`
- **Location:** `C:\Users\pavan\WasteWise-Recycling-Assistant`
- **Date:** November 1, 2026
- **Status:** All 9 Phases Completed, 100% Tested & Submission-Ready

---

## 1. Project Overview

The **WasteWise Recycling Assistant** is an edge-first, AI-driven sustainability application engineered to assist in household and municipal waste segregation. By integrating deep transfer learning via **MobileNetV2** with structured generative reasoning via **Google Gemini AI** and a deterministic **Rule-Based Fallback Knowledge Engine**, WasteWise provides real-time, highly accurate binary classification (**Organic** vs. **Recyclable**) coupled with beginner-friendly disposal, composting, upcycling, and safety recommendations.

### Core Architectural Pillars:
1. **Local Vision Inference & Controlled Metadata Transmission:** All image classification inference runs 100% locally on the user's CPU using a pre-cached Keras model (`models/best_mobilenetv2.keras`). Raw image files or pixel arrays are **never** transmitted to remote servers. When the Google Gemini API integration is enabled with an active API key, text-only metadata (the predicted waste category, numerical confidence score, and any optional user-entered item description text) is transmitted over HTTPS to generate customized guidance.
2. **Resilient Fallback Availability:** If the external Gemini API is unreachable, rate-limited, or unconfigured, the application falls back to an internal expert-curated sustainability knowledge base, providing structured guidance without external network dependencies, provided the local application environment is operational.
3. **Verified Empirical Rigor:** Model performance was evaluated on a strictly isolated test set of 2,513 images with zero data leakage, achieving **90.65% overall test accuracy**, **0.9866 ROC-AUC**, and **98.13% Recyclable precision**.

---

## 2. Completed Phases Summary

| Phase | Title | Core Deliverables & Achievements | Key Verification Output |
| :---: | :--- | :--- | :--- |
| **1** | **Dataset Inspection** | Verified 25,077 total images across all 4 directories; confirmed label mapping (0=Organic, 1=Recyclable); discovered and documented 372 palette-encoded PNGs with `.jpg` extensions. | `reports/dataset_inspection_report.md` |
| **2** | **Data Loading & Preprocessing** | Built modular `src/data_loader.py`; enforced explicit RGB conversion, $224 \times 224$ bilinear resizing, and $[-1, 1]$ normalization; created stratified 80/20 train/val split (18,051 train, 4,513 val); isolated 2,513 test samples. | `reports/data_preprocessing_report.md` |
| **3A** | **Environment Readiness** | Audited Python 3.12, TensorFlow 2.20, and Keras 3.12 environment; verified hardware compute and memory; established `.keras` local workspace caching. | `reports/environment_readiness_report.md` |
| **3B** | **Model Architecture** | Constructed `WasteWise_MobileNetV2` in `src/model.py` with frozen MobileNetV2 base (154 layers) and custom classification head (166,657 params); verified ImageNet pretrained weights (9.4 MB). | `reports/model_architecture_report.md` |
| **4** | **Model Training** | Built `src/train.py`; trained classification head using Adam ($lr=0.001$), `ReduceLROnPlateau`, and `EarlyStopping` (patience: 3); achieved best validation accuracy of **96.10%** at Epoch 7; saved checkpoint `models/best_mobilenetv2.keras` (11.64 MB). | `reports/training_report.md` |
| **5** | **Test Set Evaluation** | Built `src/evaluate.py`; evaluated best checkpoint on the 2,513 held-out test images; generated confusion matrix, ROC curve, and PR curve in `outputs/`; verified **90.65% test accuracy** and **0.9866 ROC-AUC**. | `reports/evaluation_report.md` |
| **6** | **GenAI Integration & Audit** | Built `src/genai_assistant.py` and `src/recycling_rules.py`; implemented dual-dispatch (`google-genai` SDK and direct REST requests); configured `gemini-3.8-flash` default; established text-only metadata transmission and safe API key masking (`mask_api_key`). | `reports/genai_integration_report.md` |
| **7** | **Streamlit Application** | Developed complete interactive web app `app.py` covering Home, Classifier, Dataset Explorer, Model Performance, and About sections; implemented model caching (`@st.cache_resource`) and explicit button gating for AI advice. | `reports/streamlit_ui_report.md` |
| **8** | **Integration & E2E Testing** | Built `tests/test_integration_e2e.py` (15 E2E tests); verified end-to-end vision inference on real test samples; confirmed exact mathematical preprocessing parity; eliminated Windows file descriptor resource leaks; passed all 67 tests. | `reports/integration_testing_report.md` |
| **9** | **Final Documentation & Cleanup** | Repository cleanup (removed unused empty directory); created comprehensive 21-section `README.md`; updated `.gitignore` for full security; finalized project report. | `reports/final_project_report.md` |

---

## 3. Machine Learning Model Architecture

```
Layer (type)                          Output Shape              Param #   Trainable
===================================================================================
InputLayer                            [(None, 224, 224, 3)]           0   False
MobileNetV2 (Pretrained Base)         (None, 7, 7, 1280)      2,257,984   False
GlobalAveragePooling2D                (None, 1280)                    0   False
BatchNormalization                    (None, 1280)                5,120   True
Dropout (rate=0.3)                    (None, 1280)                    0   False
Dense (128 units, activation='relu')  (None, 128)               163,968   True
Dropout (rate=0.2)                    (None, 128)                     0   False
Dense (1 unit, activation='sigmoid')  (None, 1)                     129   True
===================================================================================
Total parameters: 2,427,201 (9.26 MB)
Trainable parameters: 166,657 (651.00 KB)
Non-trainable parameters: 2,260,544 (8.62 MB)
Saved Checkpoint: models/best_mobilenetv2.keras (11.64 MB)
```

---

## 4. Dataset Summary & Partitioning

- **Source:** Sashaank Sekar, *Waste Classification data* (Kaggle).
- **Total Dataset Size:** 25,077 verified images.
- **Class Balance:** Organic = 13,966 (55.70%), Recyclable = 11,111 (44.30%).

| Partition | Organic (`O`, 0) | Recyclable (`R`, 1) | Total Images | Percentage | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 10,052 | 7,999 | **18,051** | 72.0% | Head parameter optimization |
| **Validation Set** | 2,513 | 2,000 | **4,513** | 18.0% | Checkpointing & EarlyStopping |
| **Isolated Test Set** | 1,401 | 1,112 | **2,513** | 10.0% | Unbiased final benchmarking |
| **Total** | **13,966** | **11,111** | **25,077** | **100.0%** | Full repository verification |

---

## 5. Final Evaluation & Benchmarking Results

### 5.1 Validation vs. Test Metrics Separation

| Metric | Best Validation (Epoch 7 Checkpoint) | Final Test (Held-Out Split) | Evaluation Scope |
| :--- | :---: | :---: | :--- |
| **Loss** | `0.1139` | `0.2604` | Binary Cross-Entropy |
| **Accuracy** | **96.10%** | **90.65%** | Overall Correct Classification |
| **ROC-AUC** | **0.9911** | **0.9866** | Area Under Receiver Operating Characteristic |
| **Average Precision (PR-AUC)** | — | **0.9824** | Area Under Precision-Recall Curve |
| **Macro F1-Score** | — | **90.28%** | Unweighted Class Balance |
| **Weighted F1-Score** | — | **90.50%** | Support-Weighted Metric |

### 5.2 Per-Class Breakdown (2,513 Test Samples)

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Organic (0)** | 86.39% | **98.79%** | 92.17% | 1,401 |
| **Recyclable (1)** | **98.13%** | 80.40% | 88.38% | 1,112 |
| **Macro Average** | 92.26% | 89.59% | 90.28% | 2,513 |
| **Weighted Average** | 91.59% | 90.65% | 90.50% | 2,513 |

### 5.3 Confusion Matrix
$$\begin{pmatrix} \text{TN} = 1,384 & \text{FP} = 17 \\ \text{FN} = 218 & \text{TP} = 894 \end{pmatrix}$$

- **17 Organic samples predicted as Recyclable:** Represents a **1.21% false-positive rate among Organic samples** (17 / 1,401).
- **218 Recyclable samples predicted as Organic:** Represents a **19.60% false-negative rate among Recyclable samples** (218 / 1,112).
- **True Organics (TN):** 1,384 (98.79% Organic recall / specificity).
- **True Recyclables (TP):** 894 (80.40% Recyclable recall).

---

## 6. Generative AI & Fallback Integration

- **Target Model:** `gemini-3.8-flash` (default; configurable via `GEMINI_MODEL` in `.env`).
- **Dual-Dispatch Protocol:** Automatically attempts the official `google-genai` SDK first (`from google import genai`), falling back smoothly to direct REST requests via `requests` (`https://generativelanguage.googleapis.com/v1beta/models/...:generateContent`).
- **Offline Rule-Based System:** Built in `src/recycling_rules.py`. Engages if `GEMINI_API_KEY` is missing, unconfigured, or on network partitions, returning identical 6-section structured guidance, provided the local application is operational.
- **Testing Scope:** Automated test suites use mocked API responses (`unittest.mock`) to verify Gemini error handling, quota handling, and prompt formatting without making live API calls. An optional manual script (`scripts/verify_live_gemini.py --live`) is provided for users who wish to test their live API key.
- **Structured Sections Delivered:**
  1. Category Overview & Explanation
  2. Disposal Instructions & Bin Color
  3. Composting / Recycling Recommendations
  4. Practical Upcycling & Reuse Ideas
  5. Environmental Impact (Methane Prevention / Energy Savings)
  6. Safe Handling & Contamination Precautions
  - Mandatory municipal variance legal disclaimer.

---

## 7. Streamlit Web Application Summary

- **Entrypoint:** `app.py`
- **Execution Command:** `streamlit run app.py` (Local address: `http://localhost:8501`)
- **Deployment Status:** The project is a **locally runnable Streamlit web application**; it is not deployed to external cloud hosting.
- **Key Modules & Pages:**
  - `🏠 Home & Overview`: Waste stream definitions, 3-step pipeline, and local vision privacy notes.
  - `♻️ Waste Classifier & AI Assistant`: Live local vision classification, confidence score, raw probability, optional item description input, and explicit user-triggered GenAI advice button.
  - `📊 Dataset Explorer`: Interactive distribution charts, sample counts, and inspection findings without filesystem leakage.
  - `📈 Model Performance`: Complete metric dashboards, confusion matrix display, and diagnostic curves (`outputs/`).
  - `ℹ️ About`: Architectural breakdown, transfer learning details, and real-world sorting limitations.
- **Model Caching & Latency:** `@st.cache_resource` guarantees that the 11.64 MB model is loaded into memory once on startup. Warm single-batch forward passes were benchmarked at approximately 12–25 ms on the test CPU (Intel Core i5-12450HX), while initial cold-start loading requires several seconds. Performance on other hardware or constrained environments will vary.

---

## 8. Test Suite Execution & Verification

Executed complete project test discovery across `tests/`:

```powershell
python -m unittest discover tests
```

### Official Test Results:
```text
Ran 67 tests in 29.592s

OK
```

*Note on Testing Scope: Automated tests execute programmatically and headlessly using mocked external API responses to avoid quota consumption and network dependency. Live Gemini API calls and manual browser testing are not part of the automated test suite.*

### Module Breakdown:
1. `tests/test_data_loader.py`: **9 passed** (RGB conversion, split counts, stratified 80/20 partitioning, TF dataset batching)
2. `tests/test_model.py`: **9 passed** (MobileNetV2 base layer freezing, head parameters, output shape, ImageNet weights)
3. `tests/test_train.py`: **3 passed** (Loss function, Adam optimizer, checkpointing, early stopping)
4. `tests/test_evaluate.py`: **4 passed** (Metric calculations, confusion matrix, ROC-AUC)
5. `tests/test_recycling_rules.py`: **4 passed** (6-section knowledge engine, category normalization, disclaimers)
6. `tests/test_genai_assistant.py`: **12 passed** (Prompt builder, key masking, mocked API responses, HTTP 429 quota handling, network failure recovery, backend tracking)
7. `tests/test_predict.py`: **11 passed** (Image validation, extension checking, empty/corrupt byte handling, pixel scaling bounds, calibrated confidence)
8. `tests/test_integration_e2e.py`: **15 passed** (Real checkpoint inference on held-out test samples, preprocessing parity, Streamlit UI section execution)

---

## 9. Known Real-World Limitations

1. **Strict Binary Classification Scope:** The model was trained strictly as a binary classifier: it only categorizes items as **Organic (0)** or **Recyclable (1)**. It has **no separate Non-Recyclable, Landfill, Hazardous, or Electronic Waste category**. Any non-recyclable or hazardous item (e.g. Styrofoam, ceramic, batteries, lightbulbs, composite pouches) will inevitably be classified as either Organic or Recyclable.
2. **No Guarantee of Municipal Recycling Acceptance:** A model prediction of "Recyclable" is a statistical classification based on image features and does **not** guarantee acceptance at any specific municipal recycling facility. Local sorting infrastructure, contamination standards, and accepted plastic resin codes vary by jurisdiction.
3. **Statistical Prediction Nature:** High model confidence reflects resemblance to training distributions; it does not constitute a legal or municipal sorting guarantee.
4. **Visual Ambiguity of Composite Materials:** Multilayer or composite packaging (e.g., aseptic cartons, foil-lined chip bags, plastic-lined paper coffee cups) may visually resemble clean cardboard or paper but cannot be processed through standard curbside mechanical recycling.
5. **Soiled Recyclables:** Recyclable containers contaminated with grease, sauce, or food residue (such as greasy pizza boxes) cannot be recycled and must be diverted to compost or landfill.

---

## 10. Future Enhancements

- **Multi-Class Material Segmentation:** Extend classification from binary sorting to multi-class identification (e.g., separate classes for Glass, Aluminum, Cardboard, PETE #1, HDPE #2, and Hazardous Waste).
- **Object Detection & Localization:** Integrate YOLOv8 or EfficientDet to detect and segment multiple waste items within a single cluttered photograph.
- **Edge Mobile Deployment:** Export the MobileNetV2 model to TensorFlow Lite (`.tflite`) with integer quantization for standalone offline execution on iOS and Android devices.
- **Municipal Geo-Targeting:** Incorporate user postal code lookup to automatically retrieve and display specific local municipal recycling bylaws and sorting schedules.

---

## 11. Submission Readiness Checklist

- [x] **Model Preserved:** Trained weights (`models/best_mobilenetv2.keras`) and test results (90.65% accuracy) intact and untouched.
- [x] **Secure Credential Handling:** `.env` is excluded by `.gitignore`; API keys are strictly masked (`mask_api_key`); `.env.example` contains only placeholder tokens.
- [x] **Dataset Excluded from Git:** Raw dataset directory (`data/`) is ignored in `.gitignore`.
- [x] **Models & Heavy Binaries Excluded:** `models/` and `outputs/` are ignored in `.gitignore`.
- [x] **Clean Repository:** Removed unused temporary directory `app`; no leftover cache artifacts committed.
- [x] **Accurate Dependencies:** `requirements.txt` specifies all required packages (`streamlit==1.66.1`, `tensorflow==2.20.0`, `keras==3.12.0`, `google-genai>=1.0.0`).
- [x] **Comprehensive Documentation:** Full 21-section `README.md` and 9 comprehensive reports present in `reports/`.
- [x] **Complete Test Suite Passing:** All 67 automated unit and integration tests passing (`python -m unittest discover tests`).
- [x] **Local Runnability Verified:** Streamlit application is tested and fully runnable locally via `streamlit run app.py` (no cloud deployment claimed).

