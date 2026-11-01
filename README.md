# 🌱 WasteWise Recycling Assistant

> **Intelligent Edge Waste Classification & Contextual Sustainability Guidance**<br>
> Powered by MobileNetV2 Transfer Learning and Google Gemini AI (with Deterministic Offline Fallback)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Keras 3.12](https://img.shields.io/badge/Keras-3.12.0-red.svg)](https://keras.io/)
[![Streamlit 1.66](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/Tests-67%2F67%20Passed-brightgreen.svg)]()

---

## 1. Project Title & Introduction

**WasteWise Recycling Assistant** is an end-to-end computer vision and Generative AI application engineered to assist in waste segregation and promote circular economy practices. By combining a lightweight deep learning vision model (**MobileNetV2**) with structured generative reasoning (**Google Gemini AI** and an offline rule-based knowledge engine), WasteWise provides real-time binary classification of waste items and actionable, beginner-friendly disposal guidance.

The application operates with a **local vision inference design**: all computer vision inference executes 100% locally on the user's machine using a pre-cached model. Raw image files and pixel arrays are **never** uploaded or transmitted over the network. However, when the optional Google Gemini API integration is enabled with an active API key, prediction metadata (predicted category label, numerical confidence score) and any optional user-entered item description text are transmitted to Google Gemini over HTTPS to generate customized guidance. Users who require full offline operation can leave the API key unconfigured to use the built-in rule-based fallback engine exclusively.

---

## 2. Problem Statement

Improper waste disposal and contamination in municipal recycling streams are critical environmental challenges. When non-recyclable or food-soiled organic waste enters recycling bins, entire batches of paper, plastic, and cardboard are rejected and diverted to landfills. Conversely, when organic waste is dumped in landfills rather than composted, anaerobic decomposition produces large quantities of **methane gas**—a greenhouse gas over 25 times more potent than carbon dioxide.

Municipal recycling rules are often confusing, and consumers frequently struggle to determine whether an item is compostable or recyclable. WasteWise bridges this knowledge gap through rapid visual classification and structured, contextual sustainability advice.

---

## 3. Project Objectives

1. **High-Accuracy Edge Vision Classification:** Implement and fine-tune a lightweight MobileNetV2 model to classify waste into two fundamental streams: **Organic (0)** and **Recyclable (1)** with $>90\%$ test accuracy.
2. **Local Edge Vision Inference:** Ensure all image processing and neural network inference occur locally without transmitting image bytes over the network.
3. **Generative AI Integration:** Contextualize predictions using Google Gemini API (`gemini-3.8-flash`) to generate structured, beginner-friendly waste management advice covering 6 distinct sustainability domains.
4. **Rule-Based Fallback Architecture:** Implement an expert-curated rule-based fallback system (`src/recycling_rules.py`) that provides structured recycling guidance when external Gemini APIs are unconfigured, rate-limited, or unreachable, provided the local application environment is operational.
5. **Interactive Web Application:** Deliver an intuitive, accessible Streamlit web application featuring a classifier, dataset explorer, empirical performance dashboard, and educational guides.
6. **Rigorous Scientific Verification:** Implement a comprehensive automated test suite (67 unit and end-to-end integration tests) verifying every step from image validation to inference and reporting.

---

## 4. Key Features

- **⚡ Local CPU Inference:** Pre-cached MobileNetV2 model checkpoint provides responsive on-device binary classification on standard CPU hardware (benchmarked at ~12–25 ms single-batch forward pass after initial model loading and warm-up on an Intel Core i5 test system; initial cold-start model initialization requires several seconds).
- **🛡️ Offline Fallback Guidance:** Automatically routes queries to an expert-curated rule-based knowledge base if the Gemini API key is missing, invalid, rate-limited, or offline, provided the local application is functioning.
- **🔒 Local Image Processing with Optional Text Metadata:** Raw image pixels never leave the device. When Gemini is enabled, only predicted category, confidence score, and optional user text are transmitted to the API over HTTPS.
- **📋 6-Section Structured Recommendations:** Every recommendation covers:
  1. *Category Overview & Material Breakdown*
  2. *Disposal Instructions & Bin Placement*
  3. *Recycling / Composting Recommendations*
  4. *Practical Reuse & Upcycling Ideas*
  5. *Environmental Benefits (Methane Reduction / Energy Conservation)*
  6. *Safe Handling & Contamination Precautions*
- **📊 Interactive Streamlit Dashboard:** 5 comprehensive application sections including a live classifier, dataset explorer, model benchmarking suite, and project overview.
- **🧪 Complete Automated Test Suite:** 67 tests covering data loading, model architecture, training routines, evaluation metrics, image validation, GenAI fallback, and end-to-end workflows.

---

## 5. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Runtime** | Python 3.12.6 (Windows 11) |
| **Deep Learning Framework** | TensorFlow 2.20.0, Keras 3.12.0 |
| **Model Architecture** | MobileNetV2 (ImageNet Pretrained) |
| **Data & Image Processing** | Pillow 12.0.0, NumPy 1.26.4, Pandas 2.2.2, Scikit-Learn 1.5.1 |
| **Generative AI Integration** | Google Gemini API (`gemini-3.8-flash`), `google-genai` SDK, `requests 2.32.5` |
| **Web Interface** | Streamlit 1.66.1 (Locally runnable web application) |
| **Visualization & Reporting** | Matplotlib 3.11.0, Seaborn 0.13.2 |
| **Testing & CI/CD** | Python `unittest`, `unittest.mock` |

---

## 6. Dataset Details & Attribution

The model was trained and evaluated on the **Waste Classification Dataset** created by **Sashaank Sekar** (hosted on [Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)):

- **Dataset Source:** Sashaank Sekar, *Waste Classification data*.
- **Total Verified Images:** **25,077 images**
- **Original Split Structure:**
  - `DATASET/TRAIN/O`: 12,565 images
  - `DATASET/TRAIN/R`: 9,999 images
  - `DATASET/TEST/O`: 1,401 images
  - `DATASET/TEST/R`: 1,112 images
- **Inspection Findings:** Phase 1 dataset inspection revealed 372 files with `.jpg` extensions that are internally encoded as palette-based PNGs (mode `'P'`). The preprocessing pipeline handles these transparently via explicit PIL RGB conversion.
- **Dataset Licensing Note:** Dataset licensing and terms of use are governed by the original publisher on Kaggle.

---

## 7. Dataset Distribution & Partitioning

To avoid data leakage, the original `TEST` folder (2,513 images) was isolated from the start. The `TRAIN` partition (22,564 images) was split into training and validation sets using a stratified 80/20 split (`random_state=42`):

| Partition | Organic (`O`, 0) | Recyclable (`R`, 1) | Total | Proportion | Usage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 10,052 | 7,999 | **18,051** | 72.0% | Head training with Adam ($lr=0.001$) |
| **Validation Set** | 2,513 | 2,000 | **4,513** | 18.0% | Early stopping & checkpoint selection |
| **Isolated Test Set** | 1,401 | 1,112 | **2,513** | 10.0% | Unbiased final benchmarking |
| **Total Verified** | **13,966** | **11,111** | **25,077** | **100.0%** | Full Sashaank Sekar dataset |

---

## 8. MobileNetV2 Model Architecture

The model uses transfer learning based on MobileNetV2:

```
Input (224 × 224 × 3 RGB, scaled to [-1.0, 1.0])
   │
   ▼
MobileNetV2 Base Backbone (Pretrained on ImageNet, 154 layers, FROZEN)
   │ Output shape: (None, 7, 7, 1280)
   ▼
GlobalAveragePooling2D
   │ Output shape: (None, 1280)
   ▼
BatchNormalization
   │ Output shape: (None, 1280)
   ▼
Dropout (rate = 0.3)
   ▼
Dense (128 units, activation = 'relu')
   │ Output shape: (None, 128)
   ▼
Dropout (rate = 0.2)
   ▼
Dense (1 unit, activation = 'sigmoid')
   │ Output shape: (None, 1) -> Probability p ∈ [0.0, 1.0]
```

### Parameter Breakdown:
- **Total Parameters:** 2,424,641
- **Trainable Parameters:** 166,657 (Classification Head)
- **Non-Trainable Parameters:** 2,257,984 (Frozen MobileNetV2 Base)
- **Saved Checkpoint:** `models/best_mobilenetv2.keras` (11.64 MB)

---

## 9. Training Methodology

- **Training Strategy:** Frozen base feature extraction. Only the custom classification head was trained.
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam (Initial learning rate $\eta = 0.001$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (monitor: `val_loss`, factor: 0.2, patience: 2, min_lr: 1e-5)
- **Regularization:** Batch Normalization, Dropout ($0.3$ and $0.2$)
- **Callbacks:** `ModelCheckpoint` (monitored `val_loss`, saved best checkpoint) and `EarlyStopping` (patience: 3, `restore_best_weights=True`)
- **Execution:** Early stopped at Epoch 10; best weights restored from **Epoch 7**.

---

## 10. Actual Validation and Test Results

### 10.1 Best Validation Performance (Epoch 7 Checkpoint on Validation Split: 4,513 Images)
- **Validation Loss:** `0.1139`
- **Validation Accuracy:** `96.10%`
- **Validation ROC-AUC:** `0.9911`
- **Validation Precision:** `95.24%`
- **Validation Recall:** `96.00%`

### 10.2 Final Test Performance (Held-Out Test Set: 2,513 Images)
Evaluated at the standard decision boundary ($p = 0.50$):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Loss** | `0.2604` | — |
| **Test Accuracy** | `0.9065` | **90.65%** (2,278 / 2,513 correct) |
| **Test ROC-AUC** | `0.9866` | **98.66%** |
| **Average Precision (PR-AUC)** | `0.9824` | **98.24%** |
| **Macro Average F1-Score** | `0.9028` | **90.28%** |
| **Weighted Average F1-Score** | `0.9050` | **90.50%** |

### Per-Class Test Breakdown:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Organic (`0`)** | **86.39%** | **98.79%** | **92.17%** | 1,401 |
| **Recyclable (`1`)** | **98.13%** | **80.40%** | **88.38%** | 1,112 |
| **Macro Average** | **92.26%** | **89.59%** | **90.28%** | 2,513 |
| **Weighted Average** | **91.59%** | **90.65%** | **90.50%** | 2,513 |

---

## 11. Confusion Matrix Interpretation

On the 2,513 held-out test samples:

$$\begin{pmatrix} \text{TN} = 1,384 & \text{FP} = 17 \\ \text{FN} = 218 & \text{TP} = 894 \end{pmatrix}$$

- **17 Organic samples predicted as Recyclable:** Represents a **1.21% false-positive rate among Organic samples** (17 / 1,401). This low false-positive rate indicates that very few organic items are misclassified as recyclable, helping to protect recycling batches from organic contamination.
- **218 Recyclable samples predicted as Organic:** Represents a **19.60% false-negative rate among Recyclable samples** (218 / 1,112). These 218 recyclable items were conservatively classified as organic.
- **True Negatives (TN):** **1,384** / 1,401 Organic samples correctly recognized (**98.79%** Organic recall / specificity).
- **True Positives (TP):** **894** / 1,112 Recyclable samples correctly recognized (**80.40%** Recyclable recall).

---

## 12. Gemini Integration & Offline Fallback

The application features a dual-engine architecture:
- **Primary Engine (Google Gemini AI):** Configured for `gemini-3.8-flash` (or `gemini-2.5-flash`). Uses dual-dispatch supporting both the official `google-genai` SDK and direct REST requests (`requests`).
- **Offline Rule-Based Knowledge Engine (`src/recycling_rules.py`):** Provides structured recommendations when the API key is not configured, rate-limited (HTTP 429), or offline, provided the local application is functioning.
- **Privacy Boundary:** Raw image pixels are never uploaded. Only the class label, confidence score, and optional user description are sent to Gemini when enabled.
- **Testing Verification Notice:** Automated test suites use hermetic mocking (`unittest.mock`) to verify Gemini error handling, quota handling, and prompt formatting without making live API calls. An optional manual script (`scripts/verify_live_gemini.py --live`) is provided for users who wish to test their live API key.

---

## 13. System Architecture & Workflow

```
[User Image: JPG/PNG]
         │
         ▼
[Image Validation (src/predict.py)] ──(Invalid)──> [Descriptive Error Message]
         │ (Valid)
         ▼
[Preprocessing: RGB, 224x224, [-1, 1]]
         │
         ▼
[MobileNetV2 Inference (models/best_mobilenetv2.keras)]
         │
         ▼
[Sigmoid Probability p] ──> Class Mapping: 0 = Organic (p < 0.5), 1 = Recyclable (p ≥ 0.5)
         │                   Calibrated Confidence: p if Recyclable else (1 - p)
         ▼
[Streamlit UI Display] ──> Shows Category Badge, Confidence Bar, Metrics
         │
         ▼ (User explicitly clicks "🌱 Get Guidance")
[GenAI Assistant (src/genai_assistant.py)]
     ├── If API Key Configured ──> Google Gemini API (`gemini-3.8-flash`)
     └── If Missing / Error    ──> Offline Rule-Based Fallback (`src/recycling_rules.py`)
         │
         ▼
[6 Structured Sustainability Sections Displayed in Web UI]
```

---

## 14. Installation Instructions (Windows)

### Prerequisites:
- Windows 10/11 (64-bit)
- Python 3.12 (Installed and added to system `PATH`)
- Git

### Step-by-Step Setup:

1. **Clone the Repository:**
   ```powershell
   git clone https://github.com/Pavaniangadi24/-WasteWise-Recycling-Assistant.git
   cd WasteWise-Recycling-Assistant
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 15. Environment Setup (Optional for Live Gemini)

WasteWise functions offline out-of-the-box using the built-in rule-based fallback system.

To enable live Google Gemini AI recommendations:
1. Copy the template `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and add your Google AI Studio API key:
   ```ini
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   GEMINI_MODEL=gemini-3.8-flash
   GEMINI_TIMEOUT_SECONDS=15
   ```
*(Note: `.env` is listed in `.gitignore` and is never committed.)*

---

## 16. How to Run the Streamlit Application

The project is a **locally runnable web application** (not deployed to external cloud hosting).

From the project root directory, launch the Streamlit application:

```powershell
streamlit run app.py
```

*Or run via Python:*
```powershell
python -m streamlit run app.py
```

Open your browser to:
```text
http://localhost:8501
```

---

## 17. How to Run Automated Tests

Execute the complete 67-test automated test discovery suite:

```powershell
python -m unittest discover tests
```

*Note on Testing Scope: Automated tests execute programmatically and headlessly using mocked external API responses to avoid quota consumption and network dependency. Live Gemini API calls and manual browser testing are not part of the automated test suite.*

To run a specific test module:
```powershell
# Run end-to-end integration tests
python -m unittest tests/test_integration_e2e.py

# Run prediction helper tests
python -m unittest tests/test_predict.py

# Run GenAI assistant tests
python -m unittest tests/test_genai_assistant.py
```

---

## 18. Project Directory Structure

```text
WasteWise-Recycling-Assistant/
├── .env.example                  # Template configuration for Gemini API
├── .gitignore                    # Git ignore file (excludes dataset, models, .env)
├── README.md                     # Comprehensive project documentation
├── requirements.txt              # Production dependency specifications
├── app.py                        # Streamlit multi-page web application entrypoint
│
├── .streamlit/
│   └── config.toml               # Streamlit runtime and server configuration
│
├── models/
│   └── best_mobilenetv2.keras    # Saved trained Keras model checkpoint (11.64 MB)
│
├── outputs/                      # Evaluation figures and metrics JSON/CSV files
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── test_evaluation_summary.json
│   ├── test_predictions.csv
│   ├── training_curves.png
│   ├── training_history.csv
│   └── training_summary.json
│
├── reports/                      # Detailed technical audit reports (Phases 1–9)
│   ├── dataset_inspection_report.md
│   ├── data_preprocessing_report.md
│   ├── environment_readiness_report.md
│   ├── model_architecture_report.md
│   ├── training_report.md
│   ├── evaluation_report.md
│   ├── genai_integration_report.md
│   ├── streamlit_ui_report.md
│   ├── integration_testing_report.md
│   └── final_project_report.md
│
├── scripts/
│   └── verify_live_gemini.py     # Standalone manual live verification tool
│
├── src/                          # Core application source modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading, validation, and TF dataset pipeline
│   ├── evaluate.py               # Test split evaluation and plot generation
│   ├── genai_assistant.py        # Gemini AI orchestrator and dual-dispatch client
│   ├── model.py                  # MobileNetV2 architecture builder
│   ├── predict.py                # Reusable image validation and inference helpers
│   ├── recycling_rules.py        # Expert-curated 6-section rule-based fallback
│   └── train.py                  # Model training pipeline and callbacks
│
└── tests/                        # Automated unit and integration test suite (67 tests)
    ├── test_data_loader.py       # 9 data loading and split tests
    ├── test_evaluate.py          # 4 evaluation metric tests
    ├── test_genai_assistant.py   # 12 GenAI prompt, fallback, and masking tests
    ├── test_integration_e2e.py   # 15 end-to-end integration & UI tests
    ├── test_model.py             # 9 architecture and weight loading tests
    ├── test_predict.py           # 11 validation and inference helper tests
    ├── test_recycling_rules.py   # 4 rule-based knowledge engine tests
    └── test_train.py             # 3 training callback and optimizer tests
```

---

## 19. Practical Limitations

1. **Strict Binary Classification Scope:** The model was trained strictly as a binary classifier: it only categorizes items as **Organic (0)** or **Recyclable (1)**. It has **no separate Non-Recyclable, Landfill, Hazardous, or Electronic Waste category**. Any non-recyclable or hazardous item (e.g. Styrofoam, ceramic, batteries, lightbulbs, composite pouches) will inevitably be classified as either Organic or Recyclable.
2. **No Guarantee of Municipal Recycling Acceptance:** A model prediction of "Recyclable" is a statistical classification based on image features and does **not** guarantee acceptance at any specific municipal recycling facility. Local sorting infrastructure, contamination standards, and accepted plastic resin codes vary by jurisdiction.
3. **Statistical Prediction Nature:** High model confidence reflects resemblance to training distributions; it does not constitute a legal or municipal sorting guarantee.
4. **Visual Ambiguity of Composite Materials:** Multilayer or composite packaging (e.g., aseptic cartons, foil-lined chip bags, plastic-lined paper coffee cups) may visually resemble clean cardboard or paper but cannot be processed through standard curbside mechanical recycling.
5. **Soiled Recyclables:** Recyclable containers contaminated with grease, sauce, or food residue (such as greasy pizza boxes) cannot be recycled and must be diverted to compost or landfill.

---

## 20. Future Enhancements

- **Multi-Class Material Segmentation:** Extend classification from binary sorting to multi-class identification (e.g., separate classes for Glass, Aluminum, Cardboard, PETE #1, HDPE #2, and Hazardous Waste).
- **Object Detection & Localization:** Integrate YOLOv8 or EfficientDet to detect and segment multiple waste items within a single cluttered photograph.
- **Edge Mobile Deployment:** Export the MobileNetV2 model to TensorFlow Lite (`.tflite`) with integer quantization for standalone offline execution on iOS and Android devices.
- **Municipal Geo-Targeting:** Incorporate user postal code lookup to automatically retrieve and display specific local municipal recycling bylaws and sorting schedules.

---

## 21. Author & Contribution Information

- **Developer:** Built as part of the WasteWise Recycling Assistant project.
- **Dataset Attribution:** Sashaank Sekar (*Waste Classification data*, Kaggle).
- **Backbone Model:** Google MobileNetV2 pretrained on ImageNet.
- **Generative AI:** Google Gemini API (`gemini-3.8-flash`).

