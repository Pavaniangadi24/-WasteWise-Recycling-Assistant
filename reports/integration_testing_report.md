# Phase 8: Integration and End-to-End Testing Report

- **Project:** WasteWise Recycling Assistant
- **Test Suite Directory:** `tests/`
- **E2E Test File:** `tests/test_integration_e2e.py`
- **Application Entrypoint:** `app.py`
- **Date:** November 1, 2026
- **Status:** 100% Validated (67/67 Tests Passing)

---

## 1. Executive Summary

Phase 8 conducted comprehensive, multi-tiered integration and end-to-end (E2E) testing across the complete WasteWise Recycling Assistant project. The objectives were to:
1. Verify the integrity of the end-to-end image classification pipeline from raw file upload through MobileNetV2 inference and confidence calibration.
2. Verify Generative AI assistant integration, explicit button triggering, metadata-only transmission, and deterministic rule-based fallback under network partitions and missing API keys.
3. Validate that all 5 Streamlit application sections execute without runtime crashes or variable errors.
4. Verify that all displayed performance metrics and diagnostic plots align with empirical benchmarking reports.
5. Stress-test file validation against invalid, corrupted, zero-byte, and non-image files.
6. Run the complete automated test suite and eliminate any resource leaks or warnings.

**Results:** All 67 automated tests passed cleanly in under 30 seconds with zero failures and zero errors.

---

## 2. Testing Methodology

The testing protocol employed a multi-layered verification strategy:

```
┌────────────────────────────────────────────────────────┐
│  Tier 1: Unit Testing (44 Tests)                       │
│  - Data loading, model definition, training callbacks,  │
│    evaluation math, rules, prompt builder, predict     │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Tier 2: Pipeline Component Testing (8 Tests)          │
│  - Mathematical parity between data_loader & predict   │
│  - Mode conversions (RGB, RGBA, L, P)                  │
│  - Palette-encoded PNG with .jpg extension handling    │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Tier 3: End-to-End Workflow Testing (8 Tests)         │
│  - Real inference using models/best_mobilenetv2.keras  │
│  - Held-out test samples: O_12568.jpg & R_10000.jpg    │
│  - End-to-end flow from classification to GenAI advice │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Tier 4: UI & Section Integration Testing (7 Tests)    │
│  - Streamlit rendering functions in app.py             │
│  - Navigation destinations, metric displays, curves    │
│  - Session state caching and explicit button gating    │
└────────────────────────────────────────────────────────┘
```

### Key Methodological Safeguards:
- **No Model Modification:** Trained weights (`models/best_mobilenetv2.keras`) and evaluation metrics were evaluated directly without retraining or fine-tuning.
- **Zero API Quota Consumption:** All external Gemini API network calls during automated testing were hermetically mocked via `unittest.mock.patch`, ensuring test repeatability, deterministic behavior, and zero quota usage.
- **Mathematical Parity Verification:** Preprocessing outputs from `src/data_loader.py` and `src/predict.py` were compared with `np.testing.assert_allclose(atol=1e-5)` on identical test images to confirm exact numerical consistency.

---

## 3. Components Tested

| Component | Test File | Tests | Key Validations |
| :--- | :--- | :---: | :--- |
| **Data Loader** | `tests/test_data_loader.py` | 9 | RGB mode conversion, split counts, stratified 80/20 train/val split, test set isolation, TF dataset batching. |
| **Model Architecture** | `tests/test_model.py` | 9 | MobileNetV2 base layer freezing (154 layers), classification head parameters, output shape `(None, 1)`, ImageNet weights. |
| **Training Pipeline** | `tests/test_train.py` | 3 | Binary crossentropy loss, Adam optimizer, checkpoint callback, early stopping configuration. |
| **Evaluation Suite** | `tests/test_evaluate.py` | 4 | Metric computation (accuracy, precision, recall, F1, ROC-AUC), confusion matrix generation. |
| **Rule-Based Engine** | `tests/test_recycling_rules.py` | 4 | 6-section organic/recyclable guidance generation, category normalization, local variance disclaimer. |
| **GenAI Assistant** | `tests/test_genai_assistant.py` | 12 | Prompt tokenization, API key masking, mocked API responses, HTTP 429 quota fallback, network error fallback, default model (`gemini-3.8-flash`), backend tracking. |
| **Predict Helpers** | `tests/test_predict.py` | 11 | Image validation, extension checks, empty/corrupted file rejection, pixel normalization (`x / 127.5 - 1.0`), confidence mapping ($0 = \text{Organic}, 1 = \text{Recyclable}$). |
| **End-to-End Integration** | `tests/test_integration_e2e.py` | 15 | Real checkpoint inference on held-out test images, E2E flow into advice generation, preprocessing parity, Streamlit UI section execution. |

---

## 4. End-to-End Workflow Verification Results

### 4.1 Organic Waste Workflow (Real Held-Out Test Image `O_12568.jpg`)
1. **Input & Validation:** Image validated successfully as a readable JPEG; size $225 \times 225$ pixels.
2. **Preprocessing:** Converted to RGB mode, resized to $224 \times 224$, scaled to $[-1.0, 1.0]$. Shape: `(1, 224, 224, 3)`.
3. **Inference:** Evaluated through `models/best_mobilenetv2.keras`.
   - Raw sigmoid probability: $p = 0.0012$ ($p < 0.50$).
   - Predicted class: **Organic (Label 0)**.
   - Calibrated confidence: $1.0 - 0.0012 = 99.88\%$.
   - Inference latency: ~14 ms on local CPU (warm single-image forward pass on Intel Core i5 test machine; cold start takes several seconds).
4. **Assistant Integration (Offline Mode):**
   - Transmitted metadata (`category='Organic'`, `confidence=0.9988`, `item_description='vegetable scrap'`).
   - Rule-based fallback engaged seamlessly.
   - Formatted output populated with all 6 required sections:
     1. Category Overview (biodegradable organic matter)
     2. Disposal Instructions (green/brown compost cart, no plastic liners)
     3. Composting Recommendations (home & municipal compost, brown/green balance)
     4. Practical Reuse Ideas (vegetable broth, citrus cleaner, coffee grounds)
     5. Environmental Impact (methane prevention)
     6. Safe Handling Tip (hygiene, sealing caddies)
     - Appended local variance disclaimer notice.

### 4.2 Recyclable Waste Workflow (Real Held-Out Test Image `R_10000.jpg`)
1. **Input & Validation:** Image validated successfully as a readable JPEG; size $225 \times 225$ pixels.
2. **Preprocessing:** Converted to RGB mode, resized to $224 \times 224$, scaled to $[-1.0, 1.0]$.
3. **Inference:** Evaluated through `models/best_mobilenetv2.keras`.
   - Raw sigmoid probability: $p = 0.9989$ ($p \ge 0.50$).
   - Predicted class: **Recyclable (Label 1)**.
   - Calibrated confidence: $p = 99.89\%$.
   - Inference latency: ~13 ms on local CPU (warm single-image forward pass on Intel Core i5 test machine; cold start takes several seconds).
4. **Assistant Integration (Mocked Active Gemini API):**
   - Transmitted metadata (`category='Recyclable'`, `confidence=0.9989`, `item_description='plastic water bottle'`).
   - Mocked Gemini returned structured 6-section response.
   - Response tagged as `is_fallback=False`, source `Google Gemini API (gemini-3.8-flash)`.

### 4.3 Preprocessing Parity Test
Compared `load_and_preprocess_image_pil` from `src/data_loader.py` with `preprocess_image_for_model` from `src/predict.py`:
- `np.allclose(arr_loader, arr_predict[0], atol=1e-5)` evaluated to **True**.
- Confirms mathematical equivalence between training/evaluation and real-time inference.

### 4.4 Robustness & Invalid Input Handling
Tested against edge cases:
- Non-image text file (`notes.txt`): Correctly rejected with `Unsupported file extension`.
- PDF document (`doc.pdf`): Correctly rejected with `Unsupported file extension`.
- Zero-byte empty file (`empty.jpg`): Correctly rejected with `File is empty (0 bytes)`.
- Corrupted JPEG bytes (`corrupt.jpg`): Correctly rejected with `Invalid or corrupted image`.
- Palette-encoded PNG with `.jpg` extension (Phase 1 finding): Safely decoded to RGB, resized, and classified without warnings or errors.

---

## 5. Streamlit UI Sections Verification

Each section of `app.py` was executed and tested within a programmatic test environment:

1. **`render_home()`:** Verified that welcome headers, purpose descriptions, the 3-step workflow diagram, two stream explanation cards, and local privacy notices render correctly.
2. **`render_sidebar()`:** Verified that all 5 navigation options (`🏠 Home & Overview`, `♻️ Waste Classifier & AI Assistant`, `📊 Dataset Explorer`, `📈 Model Performance`, `ℹ️ About`) are accessible and that system status cards report model and GenAI state correctly.
3. **`render_classifier()`:**
   - Input selector functions for both upload and sample image dropdown.
   - Classification displays prediction badge, confidence bar, raw probability, and statistical probability disclaimer.
   - Guidance generation is strictly gated behind the explicit `"🌱 Get Guidance"` button.
   - Advice is cached in `st.session_state` to prevent recomputation on minor UI redraws.
4. **`render_dataset_explorer()`:** Verified that verified dataset metrics (25,077 total; 22,564 train; 2,513 test), class share breakdowns, and bar charts display accurately without exposing local filesystem paths.
5. **`render_model_performance()`:**
   - Verified that validation accuracy (96.10%) is clearly distinguished from final test accuracy (90.65%).
   - Verified per-class table and confusion matrix table:
     - Organic Recall / Specificity: 98.79% (1,384 / 1,401 correct; 17 Organic predicted as Recyclable = 1.21% FPR among Organic samples).
     - Recyclable Recall: 80.40% (894 / 1,112 correct; 218 Recyclable predicted as Organic = 19.60% FNR among Recyclable samples).
     - Recyclable Precision: 98.13% (894 / 911).
   - Verified that all 4 diagnostic plots (`outputs/confusion_matrix.png`, `outputs/roc_curve.png`, `outputs/precision_recall_curve.png`, `outputs/training_curves.png`) are loaded and displayed.
6. **`render_about()`:** Verified complete documentation of architecture, transfer learning, GenAI integration, rule-based fallback, and practical sorting limitations.

---

## 6. Automated Test Suite Execution Results

Ran complete discovery suite across `tests/`:

```powershell
python -m unittest discover tests
```

### Execution Log:
```text
Ran 67 tests in 29.569s

OK
```

### Complete Test Breakdown (67 Tests):
- `tests/test_data_loader.py`: **9 passed**
- `tests/test_model.py`: **9 passed**
- `tests/test_train.py`: **3 passed**
- `tests/test_evaluate.py`: **4 passed**
- `tests/test_recycling_rules.py`: **4 passed**
- `tests/test_genai_assistant.py`: **12 passed**
- `tests/test_predict.py`: **11 passed**
- `tests/test_integration_e2e.py`: **15 passed**

---

## 7. Issues Identified and Fixes Implemented

### Issue 1: Lingering File Descriptors on Windows in Image Validation
- **Diagnosis:** In `validate_image_file()`, calling `Image.open(p)` kept the underlying OS file handle open until garbage collection occurred, triggering Python `ResourceWarning: unclosed file` during rapid batch testing.
- **Fix:** Refactored `validate_image_file()` in `src/predict.py` to read bytes using `p.read_bytes()`, immediately closing the operating system file descriptor. Image verification and loading now use in-memory `io.BytesIO(file_bytes)` followed by `img.load()`, ensuring zero open file handles.

### Issue 2: Streamlit Metric Testing Inconsistency
- **Diagnosis:** In `app.py`, metrics are called as methods on column objects (`col.metric(...)`) rather than top-level `st.metric(...)`. Initial mock assertions checked `mock_st.metric.assert_called()`, causing false assertion failures.
- **Fix:** Updated `tests/test_integration_e2e.py` to assert that columns were constructed and children properly called.

---

## 8. Remaining Limitations

1. **Strict Binary Classification Scope:** The vision classifier operates strictly as a binary classifier predicting either `Organic` (0) or `Recyclable` (1). It does NOT include separate categories for Non-Recyclable / Landfill waste, hazardous items, or electronic waste. A prediction of "Recyclable" indicates visual similarity to recyclable training distribution items, not a guarantee of municipal recycling acceptance.
2. **Local Vision Privacy & Metadata Transmission:** Image pixels are processed 100% locally on the user's device and never uploaded to cloud servers. Only prediction metadata (category label, confidence) and optional user text descriptions are transmitted over HTTPS to Gemini when API access is configured.
3. **Visual Ambiguity of Composite Materials:** Multilayer or composite packaging (e.g. aseptic cartons or plastic-lined coffee cups) cannot be reliably separated by surface appearance alone.
4. **Municipal Sorting Differences:** Waste management guidelines vary across jurisdictions. The application includes explicit disclaimers advising users to verify local recycling capabilities.
5. **Deployment & Browser Testing Scope:** Automated testing validated full programmatic and headless execution of the application pipeline using mocked API dependencies. Interactive browser-based testing can be performed by the user by launching `streamlit run app.py`. The application is structured as a locally runnable tool, not deployed to cloud hosting.
