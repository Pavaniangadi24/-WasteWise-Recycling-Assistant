"""
Unit tests for WasteWise prediction and preprocessing helpers (`src/predict.py`).
"""

import io
import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image

from src.predict import (
    validate_image_file,
    preprocess_image_for_model,
    calculate_prediction_and_confidence,
    predict_image,
    load_classifier_model,
)


class TestPredictHelpers(unittest.TestCase):
    def setUp(self):
        # Create a simple valid RGB PIL Image in memory
        self.valid_rgb_img = Image.new("RGB", (300, 200), color=(128, 64, 32))
        img_byte_arr = io.BytesIO()
        self.valid_rgb_img.save(img_byte_arr, format="JPEG")
        self.valid_jpg_bytes = img_byte_arr.getvalue()

    def test_validate_image_valid_bytes(self):
        """Test validation succeeds on valid JPEG bytes."""
        is_valid, err, img = validate_image_file(self.valid_jpg_bytes, filename="sample.jpg")
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (300, 200))

    def test_validate_image_bytesio(self):
        """Test validation succeeds on BytesIO buffer."""
        buf = io.BytesIO(self.valid_jpg_bytes)
        is_valid, err, img = validate_image_file(buf, filename="sample.jpeg")
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(img.size, (300, 200))

    def test_validate_image_invalid_extension(self):
        """Test validation rejects unsupported file extensions."""
        is_valid, err, img = validate_image_file(self.valid_jpg_bytes, filename="report.pdf")
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file extension", err)
        self.assertIsNone(img)

    def test_validate_image_empty_bytes(self):
        """Test validation rejects empty bytes."""
        is_valid, err, img = validate_image_file(b"", filename="empty.png")
        self.assertFalse(is_valid)
        self.assertIn("empty", err.lower())
        self.assertIsNone(img)

    def test_validate_image_corrupted_data(self):
        """Test validation rejects corrupted image data."""
        is_valid, err, img = validate_image_file(b"NOT_AN_IMAGE_DATA_12345", filename="corrupt.jpg")
        self.assertFalse(is_valid)
        self.assertIn("Invalid or corrupted image", err)
        self.assertIsNone(img)

    def test_preprocess_image_modes_and_scaling(self):
        """Test preprocessing converts various PIL modes to shape (1, 224, 224, 3) and range [-1.0, 1.0]."""
        modes = ["RGB", "RGBA", "L", "P"]
        for mode in modes:
            pil_img = Image.new(mode, (100, 100), color=128 if mode == "L" else 0)
            tensor = preprocess_image_for_model(pil_img, target_size=(224, 224))

            # Shape check
            self.assertEqual(tensor.shape, (1, 224, 224, 3))
            self.assertEqual(tensor.dtype, np.float32)

            # Range check [-1.0, 1.0]
            self.assertGreaterEqual(tensor.min(), -1.0)
            self.assertLessEqual(tensor.max(), 1.0)

    def test_confidence_calculation_recyclable(self):
        """Test confidence calculation for Recyclable class (p >= 0.5)."""
        res = calculate_prediction_and_confidence(0.85)
        self.assertEqual(res["label"], 1)
        self.assertEqual(res["category"], "Recyclable")
        self.assertAlmostEqual(res["confidence"], 0.85, places=5)
        self.assertEqual(res["confidence_percent"], 85.0)

    def test_confidence_calculation_organic(self):
        """Test confidence calculation for Organic class (p < 0.5): confidence is 1 - p."""
        res = calculate_prediction_and_confidence(0.12)
        self.assertEqual(res["label"], 0)
        self.assertEqual(res["category"], "Organic")
        self.assertAlmostEqual(res["confidence"], 0.88, places=5)
        self.assertEqual(res["confidence_percent"], 88.0)

    def test_confidence_calculation_boundaries(self):
        """Test edge cases at 0.0, 0.5, and 1.0."""
        # 0.0 -> Organic, confidence = 1.0
        res_0 = calculate_prediction_and_confidence(0.0)
        self.assertEqual(res_0["label"], 0)
        self.assertEqual(res_0["category"], "Organic")
        self.assertEqual(res_0["confidence"], 1.0)

        # 0.5 -> Recyclable, confidence = 0.5
        res_half = calculate_prediction_and_confidence(0.5)
        self.assertEqual(res_half["label"], 1)
        self.assertEqual(res_half["category"], "Recyclable")
        self.assertEqual(res_half["confidence"], 0.5)

        # 1.0 -> Recyclable, confidence = 1.0
        res_1 = calculate_prediction_and_confidence(1.0)
        self.assertEqual(res_1["label"], 1)
        self.assertEqual(res_1["category"], "Recyclable")
        self.assertEqual(res_1["confidence"], 1.0)

    def test_predict_image_mock_model(self):
        """Test end-to-end predict_image helper with a mocked model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.92]], dtype=np.float32)

        res = predict_image(mock_model, self.valid_rgb_img)
        self.assertEqual(res["category"], "Recyclable")
        self.assertEqual(res["label"], 1)
        self.assertEqual(res["confidence_percent"], 92.0)
        self.assertIn("inference_latency_ms", res)
        self.assertEqual(res["input_shape"], [1, 224, 224, 3])
        mock_model.predict.assert_called_once()

    def test_load_classifier_model_not_found(self):
        """Test load_classifier_model raises FileNotFoundError on invalid path."""
        with self.assertRaises(FileNotFoundError):
            load_classifier_model("non_existent_model_checkpoint.keras")


if __name__ == "__main__":
    unittest.main()
