# Phase 7: Streamlit Web Application Report

**Project:** WasteWise Recycling Assistant
**Application Entrypoint:** `app.py`
**Prediction & Preprocessing Module:** `src/predict.py`
**Test Suite:** `tests/test_predict.py` & `tests/test_integration_e2e.py`
**Date:** November 1, 2026
**Status:** Completed & 100% Validated

---

## 1. Executive Summary

Phase 7 delivered a professional, responsive, and beginner-friendly **Streamlit web application** for the WasteWise Recycling Assistant. The application provides an interactive, end-to-end user experience spanning:
1. **Home & Overview:** Visual explanation of waste streams (Organic vs. Recyclable) and the local vision workflow.
2. **Waste Classifier & AI Assistant:** Real-time local image classification via MobileNetV2 coupled with contextual Generative AI recycling guidance (with automatic offline rule-based fallback).
3. **Dataset Explorer:** Exploration of the 25,077-image Sashaank Sekar dataset, verification findings, and split distribution.
4. **Model Performance:** Comprehensive benchmarking displaying the exact empirical metrics (90.65% test accuracy, 0.9866 ROC-AUC, 96.10% validation accuracy) with interactive tables and evaluation curves.
5. **About:** Deep dive into system architecture, transfer learning design, technology stack, and real-world edge-deployment considerations.

All vision inference is executed 100% locally on the client machine (image pixels never leave the local environment; only prediction metadata and optional user text descriptions are sent to Google Gemini when the API is enabled). The trained weights (`models/best_mobilenetv2.keras`), evaluation metrics, and preprocessing pipelines were preserved without modification.

---

## 2. Files Created and Modified

| File | Status | Description |
| :--- | :---: | :--- |
| `app.py` | **Created** | Primary Streamlit multi-section web application supporting all 5 pages, custom accessible CSS styling, model caching, and responsive layouts. |
| `src/predict.py` | **Created** | Modular, reusable inference module handling image validation, RGB conversion, $224 \times 224$ resizing, $[-1, 1]$ pixel scaling, prediction calculation, and model loading. |
| `tests/test_predict.py` | **Created** | Comprehensive unit test suite with 11 tests verifying image validation, mode conversions, confidence calculation, and model loading boundaries. |
| `.streamlit/config.toml` | **Created** | Streamlit environment configuration disabling analytics telemetry and optimizing server defaults. |
| `requirements.txt` | **Modified** | Updated to include `streamlit==1.66.1`. |
| `reports/streamlit_ui_report.md` | **Created** | Comprehensive documentation and testing report for Phase 7. |

---

## 3. Application Architecture & Page Descriptions

### A. Home & Overview (`render_home()`)
- **Mission Statement:** Introduces the core challenge of municipal recycling contamination and how WasteWise assists households and organizations.
- **Waste Streams Defined:**
  - **Organic Waste (Class 0, Green Bin):** Biodegradable food scraps, fruit/vegetable peels, coffee grounds, and garden trimmings that decompose naturally.
  - **Recyclable Waste (Class 1, Blue Bin):** Rigid plastics (#1 & #2), clean aluminum/tin cans, dry cardboard, paper packaging, and glass.
- **Workflow Steps:** Illustrates the 3-step pipeline: (1) Image Upload $\rightarrow$ (2) Local MobileNetV2 Classification $\rightarrow$ (3) Actionable Recycling Guidance.
- **Local Privacy Architecture:** Reassures users that vision inference is executed 100% locally with zero image pixels uploaded; only prediction metadata (category label, confidence) and optional user text descriptions are transmitted to Gemini over HTTPS when enabled.

### B. Waste Classifier & AI Recycling Assistant (`render_classifier()`)
- **Dual Input Interface:**
  - *Option 1:* Direct file upload supporting JPG, JPEG, and PNG images.
  - *Option 2:* Curated sample test selector (allowing instant trial with held-out test set items like `O_12568.jpg` or `R_10000.jpg`).
- **Validation & Preprocessing:** Validates file size ($\le 25\text{ MB}$), integrity, and dimension. Converts images to RGB, resizes to $224 \times 224$, and scales pixels to $[-1.0, 1.0]$.
- **Classification Output:**
  - Color-coded prediction badge (Green for Organic, Blue for Recyclable).
  - Explicit class mapping: $0 = \text{Organic}$, $1 = \text{Recyclable}$.
  - Correct confidence calculation: if sigmoid output is $p$, confidence is $p$ for Recyclable and $1 - p$ for Organic.
  - Latency and sigmoid score display ($p \ge 0.50$ boundary).
  - Statistical disclaimer: Clarifies that confidence is a mathematical probability and not an unconditional guarantee of ground truth.
- **AI Recycling Assistant Integration:**
  - Allows an optional text description from the user (e.g., *"crushed soda can"*).
  - Gated behind an explicit action button (`"🌱 Get Guidance"`) to prevent automatic or unwanted API calls on page load.
  - Sends only metadata (predicted category, confidence, optional text description)—never raw image bytes.
  - Transparently labels guidance source: `Google Gemini API (gemini-3.8-flash)` vs. `Rule-Based Fallback System (Offline)`.
  - Displays the 6 structured guidance sections: Category Overview, Disposal Instructions, Recycling/Composting Recommendations, Reuse Ideas, Environmental Impact, and Safety Tips.

### C. Dataset Explorer (`render_dataset_explorer()`)
- **Original Dataset Overview:** Documents the Sashaank Sekar waste dataset.
- **Exact Counts Displayed:**
  - **Total Images:** 25,077
  - **Training Partition:** 22,564 images (Organic: 12,565, Recyclable: 9,999)
  - **Isolated Test Partition:** 2,513 images (Organic: 1,401, Recyclable: 1,112)
- **Stratified Split:** Explains the 80/20 train/validation partitioning (Train: 18,051, Val: 4,513, Test: 2,513).
- **Format Integrity:** Notes the Phase 1 finding of 372 palette-encoded PNGs with `.jpg` extensions safely decoded by the pipeline.
- **Visuals:** Bar charts and breakdown tables without exposing local filesystem paths.

### D. Model Performance (`render_model_performance()`)
- **Distinct Metric Separation:** Explicitly distinguishes validation metrics (achieved during training head tuning) from final test metrics (measured on the held-out test split).
- **Exact Empirical Results:**
  - **Best Validation Accuracy:** **96.10%** (Validation Loss: 0.1139, ROC-AUC: 0.9911 at Epoch 7).
  - **Final Test Accuracy:** **90.65%** (2,278 / 2,513 correct).
  - **Test ROC-AUC:** **0.9866**.
  - **Test Average Precision (PR-AUC):** **0.9824**.
  - **Test Macro F1-Score:** **90.28%**.
  - **Test Weighted F1-Score:** **90.50%**.
- **Per-Class Breakdown:**
  - Organic (Class 0): Precision: 86.39%, Recall: 98.79%, F1-Score: 92.17% (Support: 1,401).
  - Recyclable (Class 1): Precision: 98.13%, Recall: 80.40%, F1-Score: 88.38% (Support: 1,112).
- **Confusion Matrix Analysis:**
  - True Organics: 1,384 correctly classified (Organic Recall / Specificity: 98.79%).
  - False Positives: 17 Organic samples predicted as Recyclable (**1.21% false-positive rate among Organic samples**, 17 / 1,401).
  - False Negatives: 218 Recyclable samples predicted as Organic (**19.60% false-negative rate among Recyclable samples**, 218 / 1,112).
  - True Recyclables: 894 correctly classified (Recyclable Recall: 80.40%).
- **Diagnostic Curves:** Renders actual output plots from `outputs/`:
  - `outputs/confusion_matrix.png`
  - `outputs/roc_curve.png`
  - `outputs/precision_recall_curve.png`
  - `outputs/training_curves.png`

### E. About (`render_about()`)
- **Architecture:** MobileNetV2 base (154 frozen layers) + GlobalAveragePooling2D, BatchNormalization, Dropout(0.3), Dense(128, ReLU), Dropout(0.2), Dense(1, Sigmoid).
- **Technology Stack:** Python 3.12, TensorFlow 2.20, Keras 3.12, Streamlit 1.66, Pillow, Scikit-Learn, Requests.
- **Practical Limitations:** Notes binary-only classification, visual ambiguity of soiled/composite items, and municipal sorting variance.

---

## 4. How to Run the Application

The application is structured as a **locally runnable Streamlit web application** (`streamlit run app.py`), accessible in any standard web browser on the local machine. It is not currently deployed to cloud hosting.

### A. Launch Command
From the project root directory (`C:\Users\pavan\WasteWise-Recycling-Assistant`), run:

```powershell
streamlit run app.py
```

Or explicitly with python module:
```powershell
python -m streamlit run app.py
```

The application will open in your default web browser at:
```text
http://localhost:8501
```

### B. Gemini API Key Configuration (Optional)
The application operates 100% offline out-of-the-box using the built-in rule-based fallback system. To enable live Gemini AI guidance:
1. Create a `.env` file in the project root:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Insert your Google AI Studio API key:
   ```ini
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   GEMINI_MODEL=gemini-3.8-flash
   ```
3. Restart or reload the Streamlit app.

---

## 5. Model Loading, Preprocessing & Caching Details

- **Model Caching & Latency Characteristics:**
  - Model loading is decorated with `@st.cache_resource(show_spinner=False)`. The Keras model is loaded into memory once upon application initialization.
  - **Cold Start vs. Warm Inference:** Initial model loading and TensorFlow graph building take approximately 3–5 seconds on application startup. Once loaded and warmed up, single-image forward-pass CPU inference averages ~12–25 ms on the test benchmark machine (Intel Core i5-12450HX). Note that this is an empirical benchmark under specific test conditions, not a universal latency guarantee across different hardware.
- **Preprocessing Pipeline Consistency:**
  - Image input is validated via `validate_image_file()`.
  - Mode is explicitly forced to RGB: `pil_img.convert("RGB")`.
  - Resize uses bilinear interpolation: `pil_img.resize((224, 224), Image.Resampling.BILINEAR)`.
  - Pixel normalization is mathematically identical to MobileNetV2 `preprocess_input`:
    $$\text{normalized} = \frac{x}{127.5} - 1.0 \quad \in [-1.0, 1.0]$$
- **Confidence Computation:**
  $$\text{Confidence} = \begin{cases} p & \text{if } p \ge 0.50 \quad (\text{Recyclable, label } 1) \\ 1 - p & \text{if } p < 0.50 \quad (\text{Organic, label } 0) \end{cases}$$

---

## 6. Automated Testing & Validation

The project test suite encompasses 67 automated tests across all components (`tests/test_predict.py`, `tests/test_integration_e2e.py`, etc.).

Executed:
```powershell
python -m unittest discover tests
```

### Complete Test Coverage Breakdown (67 Tests Passing):
1. **`tests/test_data_loader.py` (9 tests):** Validates RGB conversion, dataset directory paths, split records, deterministic split isolation, and TensorFlow dataset batching.
2. **`tests/test_model.py` (9 tests):** Validates MobileNetV2 architecture, frozen base layers, trainable classification head, output shape `(None, 1)`, and ImageNet weights.
3. **`tests/test_train.py` (3 tests):** Validates loss function, optimizer setup, checkpoint callbacks, and training convergence.
4. **`tests/test_evaluate.py` (4 tests):** Validates evaluation pipeline, metrics calculation (accuracy, precision, recall, F1, ROC-AUC), and confusion matrix generation.
5. **`tests/test_recycling_rules.py` (4 tests):** Validates 6-section organic/recyclable guidance generation, category normalization, and disclaimer inclusion.
6. **`tests/test_genai_assistant.py` (12 tests):** Validates prompt engineering, key masking, mocked API responses, HTTP 429 quota fallback, network failure fallback, empty response handling, default model selection, and backend tracking.
7. **`tests/test_predict.py` (11 tests):** Validates image file validation (JPG/PNG), rejection of invalid extensions, rejection of corrupted/empty bytes, preprocessing mode conversions (RGB/RGBA/L/P), normalized pixel bounds, confidence calculation for both classes, decision boundaries, and model loading error handling.
8. **`tests/test_integration_e2e.py` (15 tests):** Validates full end-to-end classification pipeline, held-out test sample workflows, preprocessing parity, Streamlit UI section execution, and edge case handling.

---

## 7. Known Real-World Limitations

1. **Strict Binary Classification Scope:** The vision classifier operates strictly as a binary classifier predicting either `Organic` (0) or `Recyclable` (1). It does NOT include separate classes for Non-Recyclable / Landfill waste, hazardous items, or electronic waste. A prediction of "Recyclable" indicates visual similarity to recyclable training distribution items and does not guarantee municipal recycling acceptance.
2. **Statistical Prediction Nature:** High model confidence reflects resemblance to training distributions; it is not an unconditional guarantee of municipal sorting acceptance.
3. **Hidden Composite Materials:** Paper coffee cups with polyethylene liners, composite snack wrappers, and greasy pizza boxes appear clean on one surface but are contaminated for mechanical recycling.
4. **Regional Sorting Variations:** Municipal capabilities differ significantly across regional waste management districts. Users are encouraged to verify local waste collection policies.
