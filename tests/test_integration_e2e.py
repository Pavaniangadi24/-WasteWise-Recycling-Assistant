"""
End-to-End Integration Tests for WasteWise Recycling Assistant.

Validates the full workflow from raw image input, validation, preprocessing,
model inference, confidence calibration, and GenAI / offline rule-based fallback
guidance without requiring a live external Gemini network connection.
"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

# Ensure project root in sys.path
_project_root = Path(__file__).resolve().parent.parent

from src.predict import (
    validate_image_file,
    preprocess_image_for_model,
    calculate_prediction_and_confidence,
    predict_image,
    load_classifier_model,
    DEFAULT_MODEL_PATH,
)
from src.data_loader import load_and_preprocess_image_pil
from src.genai_assistant import (
    GenAIRecyclingAssistant,
    get_recycling_advice,
    DEFAULT_GEMINI_MODEL,
)


class TestEndToEndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load real model checkpoint once for the entire test suite."""
        cls.model_path = _project_root / DEFAULT_MODEL_PATH
        cls.assertTrue(cls.model_path.is_file(), f"Model checkpoint missing: {cls.model_path}")
        cls.model = load_classifier_model(str(cls.model_path))

        # Sample test images
        cls.organic_sample_path = _project_root / "data" / "raw" / "DATASET" / "TEST" / "O" / "O_12568.jpg"
        cls.recyclable_sample_path = _project_root / "data" / "raw" / "DATASET" / "TEST" / "R" / "R_10000.jpg"

    def test_e2e_real_model_classification_organic(self):
        """Test complete classification pipeline on a real held-out organic test image."""
        self.assertTrue(self.organic_sample_path.is_file(), "Organic test sample not found.")

        # Step 1: Validation
        is_valid, err, pil_img = validate_image_file(self.organic_sample_path)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(pil_img)

        # Step 2 & 3: Preprocessing & Model Prediction
        pred = predict_image(self.model, pil_img)

        # Step 4: Verify prediction structure & class
        self.assertEqual(pred["category"], "Organic")
        self.assertEqual(pred["label"], 0)
        self.assertLess(pred["raw_probability"], 0.5)
        self.assertGreaterEqual(pred["confidence"], 0.5)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertEqual(pred["confidence_percent"], round(pred["confidence"] * 100, 2))
        self.assertEqual(pred["input_shape"], [1, 224, 224, 3])
        self.assertGreater(pred["inference_latency_ms"], 0)

    def test_e2e_real_model_classification_recyclable(self):
        """Test complete classification pipeline on a real held-out recyclable test image."""
        self.assertTrue(self.recyclable_sample_path.is_file(), "Recyclable test sample not found.")

        # Step 1: Validation
        is_valid, err, pil_img = validate_image_file(self.recyclable_sample_path)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(pil_img)

        # Step 2 & 3: Preprocessing & Model Prediction
        pred = predict_image(self.model, pil_img)

        # Step 4: Verify prediction structure & class
        self.assertEqual(pred["category"], "Recyclable")
        self.assertEqual(pred["label"], 1)
        self.assertGreaterEqual(pred["raw_probability"], 0.5)
        self.assertGreaterEqual(pred["confidence"], 0.5)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertEqual(pred["confidence_percent"], round(pred["confidence"] * 100, 2))
        self.assertEqual(pred["input_shape"], [1, 224, 224, 3])
        self.assertGreater(pred["inference_latency_ms"], 0)

    def test_e2e_classification_to_offline_fallback_guidance(self):
        """Test pipeline output flowing directly into offline rule-based fallback assistant."""
        # 1. Classify organic sample
        _, _, pil_img = validate_image_file(self.organic_sample_path)
        pred = predict_image(self.model, pil_img)

        # 2. Query assistant without API key
        res = get_recycling_advice(
            category=pred["category"],
            confidence=pred["confidence"],
            item_description="vegetable scrap",
            api_key="",
        )

        # 3. Assert fallback engagement and structure
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["category"], "Organic")
        self.assertEqual(res["backend"], "Rule-Based Fallback Engine")
        self.assertIn("Rule-Based Fallback", res["source"])

        # 4. Assert all 6 sections are populated
        text = res["formatted_text"]
        self.assertIn("1. Category Overview", text)
        self.assertIn("2. Disposal Instructions", text)
        self.assertIn("3. Composting / Recycling Recommendations", text)
        self.assertIn("4. Practical Reuse Ideas", text)
        self.assertIn("5. Environmental Impact", text)
        self.assertIn("6. Safe Handling Tip", text)
        self.assertIn("Local Variance Notice", text)
        self.assertIn("vegetable scrap", text)

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_e2e_classification_to_mocked_gemini_guidance(self, mock_gemini_call):
        """Test pipeline output flowing into active Gemini assistant."""
        mock_gemini_call.return_value = (
            "1. Category Explanation: Recyclable plastic material.\n"
            "2. Disposal Instructions: Rinse and place in blue bin.\n"
            "3. Recycling Recommendations: Mechanical shredding and pelletizing.\n"
            "4. Practical Reuse Ideas: Planter or storage scoop.\n"
            "5. Environmental Benefits: Saves 95% energy.\n"
            "6. Safety & Handling Tip: Avoid sharp cut edges.\n\n"
            "Note: Municipal recycling capabilities differ."
        )

        # 1. Classify recyclable sample
        _, _, pil_img = validate_image_file(self.recyclable_sample_path)
        pred = predict_image(self.model, pil_img)

        # 2. Query assistant with mocked valid key
        res = get_recycling_advice(
            category=pred["category"],
            confidence=pred["confidence"],
            item_description="plastic water bottle",
            api_key="AIzaSyMockedValidKeyForTesting12345",
        )

        # 3. Assert success
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["category"], "Recyclable")
        self.assertIn("Google Gemini API", res["source"])
        self.assertEqual(res["model_used"], DEFAULT_GEMINI_MODEL)
        self.assertIn("plastic water bottle", res["item_description"])
        self.assertIn("Recyclable plastic material", res["formatted_text"])

    def test_e2e_preprocessing_pipeline_consistency(self):
        """
        Verify that preprocessing in `src/predict.py` matches `src/data_loader.py`
        identically down to floating-point numerical tolerances.
        """
        # Preprocess via data_loader
        tensor_loader = load_and_preprocess_image_pil(self.organic_sample_path)

        # Preprocess via predict helper
        with Image.open(self.organic_sample_path) as img:
            tensor_predict = preprocess_image_for_model(img)

        # Assert shapes
        self.assertEqual(tensor_loader.shape, (224, 224, 3))
        self.assertEqual(tensor_predict.shape, (1, 224, 224, 3))

        # Assert numerical parity
        np.testing.assert_allclose(
            tensor_loader,
            tensor_predict[0],
            atol=1e-5,
            err_msg="Preprocessing discrepancy detected between data_loader.py and predict.py!",
        )

    def test_e2e_palette_mode_p_png_with_jpg_extension(self):
        """
        Replicate the Phase 1 dataset discovery (372 mode 'P' images with .jpg extensions)
        and verify flawless end-to-end processing without errors.
        """
        # Create an in-memory palette (mode 'P') image saved as PNG format with .jpg filename
        img_p = Image.new("P", (150, 150))
        img_p.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] * 85)

        buf = io.BytesIO()
        img_p.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        # Validate
        is_valid, err, loaded = validate_image_file(raw_bytes, filename="palette_item.jpg")
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        # Preprocess & Predict
        pred = predict_image(self.model, loaded)
        self.assertIn(pred["category"], ["Organic", "Recyclable"])
        self.assertIn(pred["label"], [0, 1])
        self.assertEqual(pred["input_shape"], [1, 224, 224, 3])

    def test_e2e_invalid_upload_handling(self):
        """Verify robust rejection of invalid, empty, or corrupted files."""
        # 1. Non-image text file
        v, err, _ = validate_image_file(b"Hello world this is not an image", filename="notes.txt")
        self.assertFalse(v)
        self.assertIn("Unsupported file extension", err)

        # 2. PDF file
        v, err, _ = validate_image_file(b"%PDF-1.5 fake pdf content", filename="doc.pdf")
        self.assertFalse(v)
        self.assertIn("Unsupported file extension", err)

        # 3. Empty file
        v, err, _ = validate_image_file(b"", filename="empty.jpg")
        self.assertFalse(v)
        self.assertIn("empty", err.lower())

        # 4. Corrupted JPEG header
        v, err, _ = validate_image_file(b"\xff\xd8\xff\xe0CORRUPTED_BYTES_HERE", filename="corrupt.jpg")
        self.assertFalse(v)
        self.assertIn("corrupted", err.lower())

    def test_e2e_evaluation_metrics_and_artifacts_integrity(self):
        """Verify that all saved evaluation metrics and plots match reports perfectly."""
        summary_path = _project_root / "outputs" / "test_evaluation_summary.json"
        self.assertTrue(summary_path.is_file(), "Evaluation summary JSON missing.")

        with open(summary_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Assert exact reported benchmark metrics
        self.assertAlmostEqual(metrics["accuracy"], 0.906486, places=4)  # 90.65%
        self.assertAlmostEqual(metrics["roc_auc"], 0.986571, places=4)   # 0.9866
        self.assertAlmostEqual(metrics["macro_avg"]["f1-score"], 0.902790, places=4)  # 90.28%
        self.assertAlmostEqual(metrics["weighted_avg"]["f1-score"], 0.904970, places=4)  # 90.50%

        # Assert confusion matrix counts
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tn"], 1384)
        self.assertEqual(cm["fp"], 17)
        self.assertEqual(cm["fn"], 218)
        self.assertEqual(cm["tp"], 894)
        self.assertEqual(metrics["total_samples"], 2513)

        # Assert all 6 diagnostic plot files exist and are non-empty
        expected_plots = [
            "confusion_matrix.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            "training_curves.png",
            "accuracy_curve.png",
            "loss_curve.png",
        ]
        for plot_name in expected_plots:
            plot_file = _project_root / "outputs" / plot_name
            self.assertTrue(plot_file.is_file(), f"Missing output plot: {plot_name}")
            self.assertGreater(plot_file.stat().st_size, 10000, f"Plot file suspiciously small: {plot_name}")


class TestStreamlitSectionsIntegration(unittest.TestCase):
    """
    Validates that each of the five Streamlit application sections executes
    without runtime exceptions, unhandled errors, or missing variable references.
    """

    def setUp(self):
        # Create a mock streamlit environment with realistic return values
        self.patcher = patch("app.st")
        self.mock_st = self.patcher.start()

        # Configure columns and tabs unpacking
        def mock_columns(spec, **kwargs):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def mock_tabs(tab_names):
            return [MagicMock() for _ in range(len(tab_names))]

        self.mock_st.columns.side_effect = mock_columns
        self.mock_st.tabs.side_effect = mock_tabs
        self.mock_st.session_state = {}

    def tearDown(self):
        self.patcher.stop()

    def test_app_render_home_executes_cleanly(self):
        """Verify render_home() executes and populates page elements without errors."""
        import app
        app.render_home()
        self.mock_st.markdown.assert_called()
        self.mock_st.info.assert_called()

    def test_app_render_dataset_explorer_executes_cleanly(self):
        """Verify render_dataset_explorer() displays counts and distribution table."""
        import app
        app.render_dataset_explorer()
        self.mock_st.columns.assert_called()
        self.mock_st.dataframe.assert_called()
        self.mock_st.bar_chart.assert_called()

    def test_app_render_model_performance_executes_cleanly(self):
        """Verify render_model_performance() renders benchmark highlights and diagnostic plots."""
        import app
        app.render_model_performance()
        self.mock_st.columns.assert_called()
        self.mock_st.dataframe.assert_called()
        self.mock_st.image.assert_called()

    def test_app_render_about_executes_cleanly(self):
        """Verify render_about() displays project architecture and disclaimers."""
        import app
        app.render_about()
        self.mock_st.markdown.assert_called()

    def test_app_render_sidebar_navigation(self):
        """Verify render_sidebar() provides all 5 navigation destinations."""
        import app
        self.mock_st.sidebar.radio.return_value = "🏠 Home & Overview"
        page = app.render_sidebar()
        self.assertEqual(page, "🏠 Home & Overview")
        self.mock_st.sidebar.radio.assert_called_once()
        nav_options = self.mock_st.sidebar.radio.call_args[0][1]
        self.assertEqual(len(nav_options), 5)
        self.assertIn("🏠 Home & Overview", nav_options)
        self.assertIn("♻️ Waste Classifier & AI Assistant", nav_options)
        self.assertIn("📊 Dataset Explorer", nav_options)
        self.assertIn("📈 Model Performance", nav_options)
        self.assertIn("ℹ️ About", nav_options)

    @patch("app.predict_image")
    @patch("app.get_cached_model")
    def test_app_render_classifier_with_sample_image(self, mock_model, mock_predict):
        """Verify render_classifier() executes inference and allows guidance generation."""
        import app
        mock_model.return_value = MagicMock()
        mock_predict.return_value = {
            "category": "Organic",
            "confidence": 0.965,
            "confidence_percent": 96.5,
            "raw_probability": 0.035,
            "inference_latency_ms": 14.2,
        }

        # Simulate choosing sample image
        self.mock_st.radio.return_value = "🖼️ Try a Sample Image from Test Set"
        self.mock_st.selectbox.return_value = "🥬 Sample Organic 1 (O_12568.jpg)"
        self.mock_st.button.return_value = False  # Did not click advice yet

        app.render_classifier()

        # Prediction was executed and rendered
        mock_predict.assert_called_once()
        self.assertEqual(self.mock_st.session_state["last_category"], "Organic")
        self.assertEqual(self.mock_st.session_state["last_confidence"], 0.965)
        self.mock_st.progress.assert_called_once_with(0.965)

    @patch("app.predict_image")
    @patch("app.get_cached_model")
    def test_app_render_classifier_with_guidance_click(self, mock_model, mock_predict):
        """Verify explicit user button click triggers advice generation and caches it."""
        import app
        mock_model.return_value = MagicMock()
        mock_predict.return_value = {
            "category": "Recyclable",
            "confidence": 0.98,
            "confidence_percent": 98.0,
            "raw_probability": 0.98,
            "inference_latency_ms": 12.0,
        }

        self.mock_st.radio.return_value = "🖼️ Try a Sample Image from Test Set"
        self.mock_st.selectbox.return_value = "📦 Sample Recyclable 1 (R_10000.jpg)"
        self.mock_st.text_input.return_value = "clean aluminum can"
        self.mock_st.button.return_value = True  # User explicitly clicked "Get Guidance"

        app.render_classifier()

        # Verify advice is cached in session state
        self.assertIn("cached_advice", self.mock_st.session_state)
        cached = self.mock_st.session_state["cached_advice"]
        self.assertEqual(cached["category"], "Recyclable")
        self.assertTrue(cached["is_fallback"])  # Offline fallback triggered since no key is set


if __name__ == "__main__":
    unittest.main()
