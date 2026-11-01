# Phase 6: GenAI Integration and Audit Report

**Project:** WasteWise Recycling Assistant  
**Assistant Source:** `src/genai_assistant.py`  
**Fallback Knowledge Source:** `src/recycling_rules.py`  
**Configuration Template:** `.env.example`  
**Verification Script:** `scripts/verify_live_gemini.py`  
**Date:** November 1, 2026  
**Status:** Audit Completed & 100% Validated  

---

## 1. Executive Summary & Audit Purpose

This audit report documents the finalized Phase 6 GenAI Integration for the WasteWise Recycling Assistant prior to commencing Phase 7 (Streamlit UI). 

The assistant translates MobileNetV2 image classification predictions (`Organic` vs `Recyclable` with associated confidence scores) into actionable, structured recycling guidance. The architecture guarantees resilience and deterministic availability through a dual-mode engine:
1. **Generative AI Assistant:** Leverages Google Gemini with prompt engineering across 6 structured sustainability sections.
2. **Deterministic Rule-Based Fallback:** Instantly and seamlessly engages when the Gemini API key is unconfigured, rate-limited, blocked by network partitions, or encounters API errors.

---

## 2. Integration Architecture & Data Flow

```
┌────────────────────────────────────────────────────────┐
│  Vision Perception Component (MobileNetV2)            │
│  - Input: 224x224x3 RGB Waste Image                    │
│  - Local Inference on Client Machine                   │
│  - Output: Category ('Organic'/'Recyclable')           │
│    + Confidence (e.g. 0.96)                            │
└──────────────────────────┬─────────────────────────────┘
                           │ Category + Confidence
                           ▼
┌────────────────────────────────────────────────────────┐
│  GenAI Assistant Orchestrator (src/genai_assistant.py) │
│  - Validates confidence score in [0.0, 1.0]            │
│  - Checks API key presence & masks credentials         │
│  - Formats structured 6-section prompt                 │
└──────────────┬───────────────────────────┬─────────────┘
               │ If API Key Active         │ If Missing / Quota / Error
               ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│  Google Gemini Engine        │ │  Rule-Based Fallback Engine  │
│  - Model: gemini-3.8-flash   │ │  (src/recycling_rules.py)    │
│  - Backend: google-genai SDK │ │  - Deterministic knowledge   │
│    or Gemini REST client     │ │  - 6 structured sections     │
│  - Source: "Google Gemini"   │ │  - Source: "Rule-Based"      │
└──────────────────────────────┘ └──────────────────────────────┘
```

---

## 3. SDK Implementation & Backend Confirmation

### 3.1 Dual-Backend Dispatch Architecture
The system supports both the modern unified `google-genai` SDK and direct REST requests without requiring code modifications:
- **Preferred Backend (`google-genai` SDK):**  
  `requirements.txt` specifies `google-genai>=1.0.0`. When this package is installed, `src/genai_assistant.py` instantiates `genai.Client(api_key=...)` and executes `client.models.generate_content(...)`.
- **Graceful REST Backend (`requests`):**  
  If the `google-genai` SDK is not installed in the active Python environment, `src/genai_assistant.py` seamlessly dispatches requests via HTTP POST to the official Gemini endpoint:
  ```
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
  ```
- **Fallback Backend (`src/recycling_rules.py`):**  
  If no key is configured or if an error (such as HTTP 429 Resource Exhausted or connection loss) occurs, the system transparently routes the request to the offline rule-based knowledge base.

Both the response metadata (`backend` field) and logging transparently record which backend served the guidance (`"google-genai SDK"`, `"Gemini REST API (requests)"`, or `"Rule-Based Fallback Engine"`).

---

## 4. Default Model Selection & Configurable Options

- **Default Model:** `gemini-3.8-flash`  
  Configured in `src/genai_assistant.py` as `DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"` and in `.env.example`.
- **Configurability via `.env`:**  
  The target model is dynamically configurable via the `GEMINI_MODEL` environment variable. Users can switch models without touching application code.
- **API Availability & Alternative Models:**  
  Because Google AI Studio regularly updates model tiers and regional availability, the assistant is compatible with all Gemini Flash and Pro model identifiers:
  - `gemini-3.8-flash` (Configured default)
  - `gemini-2.5-flash` (Recommended high-throughput baseline)
  - `gemini-2.0-flash`
  - `gemini-1.5-flash`

If a specific model identifier is not active on a user's API key tier (resulting in HTTP 404), the assistant automatically catches the exception, serves the rule-based fallback guidance, and prompts the user to select an alternate model in `.env`.

---

## 5. Security & API Key Protection

1. **Zero Key Exposure in Logs or UI:**  
   The `mask_api_key()` utility truncates all API keys (e.g. `AIza...ef` or `[NOT_SET]`). Raw API keys are never printed, written to logs, or displayed in UI or reports.
2. **Environment Variable Loading:**  
   API keys are loaded from local environment variables or an uncommitted `.env` file via `python-dotenv` (with an automatic lightweight line-parser fallback).
3. **Repository Cleanliness:**  
   The `.gitignore` strictly excludes `.env` to prevent credential commits. Only `.env.example` containing placeholder tokens is tracked in version control.

---

## 6. Data Privacy & Network Transmission Clarification

> [!IMPORTANT]
> **Data Transmission Clarification:**  
> The assertion that data handling is private must be stated with precision rather than as an unconditional privacy guarantee:
> - **What stays 100% local:** Image pixels, image files, and MobileNetV2 vision inference run entirely on the local machine. **Raw image bytes are NEVER uploaded or transmitted over the network.**
> - **What is transmitted when Gemini is enabled:** If and only if a valid `GEMINI_API_KEY` is provided, text metadata—specifically the classifier's predicted category (`"Organic"` or `"Recyclable"`), the numeric confidence score (e.g. `0.95`), and any optional user-entered item text—is transmitted over HTTPS to Google's Gemini API to generate the customized recommendations.
> - **Complete Air-Gap Option:** Users requiring zero external network transmissions can simply leave `GEMINI_API_KEY` blank or unset. The application runs completely offline using the built-in rule-based expert system.

---

## 7. Manual Live API Verification Procedure

To prevent accidental API quota consumption, test flakiness, and dependency on live network connections during CI/CD, live API tests are strictly separated from automated unit tests:

1. **Location:** `scripts/verify_live_gemini.py` (located in `scripts/`, excluded from `unittest` discovery).
2. **Offline Mode (Default):**  
   Running without flags tests the local fallback system:
   ```powershell
   python scripts/verify_live_gemini.py
   ```
3. **Live Verification (Opt-In with `--live`):**  
   To test real connectivity with an active Gemini API key:
   ```powershell
   # Standard live verification using default model (gemini-3.8-flash)
   python scripts/verify_live_gemini.py --live

   # Live verification with specific model override
   python scripts/verify_live_gemini.py --live --model gemini-2.5-flash
   ```
4. **Behavior:**
   - Displays masked key status (`AIza...78`).
   - Measures round-trip API latency.
   - Prints formatted preview of received advice.
   - If the API call fails, catches the error and demonstrates instant fallback.

---

## 8. Test Suite Verification

Automated unit tests use `unittest.mock` to ensure 100% deterministic, offline execution with zero network dependency.

### Test Coverage Breakdown:
- `tests/test_data_loader.py`: 9 tests (RGB conversion, sizing, split isolation)
- `tests/test_model.py`: 9 tests (MobileNetV2 base, frozen layers, classification head)
- `tests/test_train.py`: 3 tests (optimizer, metrics, checkpointing)
- `tests/test_evaluate.py`: 4 tests (metrics, confusion matrix, ROC-AUC)
- `tests/test_recycling_rules.py`: 4 tests (organic & recyclable rule knowledge)
- `tests/test_genai_assistant.py`: 12 tests (mocked API responses, timeout, quota handling, key masking, prompt engineering, default model configuration, and backend tracking)

**Total Test Suite:** 41 tests passed (0 failures, 0 errors).
