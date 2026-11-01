"""
Unit tests for WasteWise Google Gemini AI Assistant Integration and Fallback Mechanism.

All API interactions are strictly mocked with `unittest.mock` to ensure deterministic,
fast, and quota-free test execution.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from src.genai_assistant import (
    GenAIRecyclingAssistant,
    get_recycling_advice,
    build_gemini_prompt,
    validate_confidence,
    mask_api_key,
)


class TestGenAIAssistant(unittest.TestCase):
    def setUp(self):
        # Ensure fresh assistant instance for each test
        self.assistant = GenAIRecyclingAssistant(api_key="mock_valid_api_key_123456789")

    def test_mask_api_key(self):
        """Test API key masking for safe logging."""
        self.assertEqual(mask_api_key(None), "[NOT_SET]")
        self.assertEqual(mask_api_key(""), "[NOT_SET]")
        self.assertEqual(mask_api_key("short"), "[NOT_SET]")
        masked = mask_api_key("AIzaSyD-1234567890abcdef")
        self.assertTrue(masked.startswith("AIza..."))
        self.assertTrue(masked.endswith("ef"))
        self.assertNotIn("1234567890", masked)

    def test_confidence_validation(self):
        """Test confidence score validation and boundary clamping."""
        self.assertEqual(validate_confidence(0.95), 0.95)
        self.assertEqual(validate_confidence(0.0), 0.0)
        self.assertEqual(validate_confidence(1.0), 1.0)
        self.assertEqual(validate_confidence(1), 1.0)

        with self.assertRaises(ValueError):
            validate_confidence(-0.1)

        with self.assertRaises(ValueError):
            validate_confidence(1.5)

        with self.assertRaises(TypeError):
            validate_confidence("high")

    def test_prompt_construction(self):
        """Test prompt builder contains classifier outputs and guidelines."""
        prompt = build_gemini_prompt("Organic", 0.965, item_description="apple peel")
        self.assertIn("Organic", prompt)
        self.assertIn("96.5%", prompt)
        self.assertIn("apple peel", prompt)
        self.assertIn("6 numbered sections", prompt)
        self.assertIn("local municipal recycling facilities", prompt)

    def test_missing_api_key_triggers_fallback(self):
        """Test that missing or placeholder API keys automatically trigger rule-based fallback."""
        assistant_empty = GenAIRecyclingAssistant(api_key="")
        res = assistant_empty.generate_recommendations("Recyclable", 0.92)

        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["category"], "Recyclable")
        self.assertIn("Rule-Based Fallback", res["source"])
        self.assertIn("error", res)

        # Test placeholder key
        assistant_placeholder = GenAIRecyclingAssistant(api_key="your_gemini_api_key_here")
        res_place = assistant_placeholder.generate_recommendations("Organic", 0.88)
        self.assertTrue(res_place["is_fallback"])

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_successful_mocked_gemini_response(self, mock_call):
        """Test successful Gemini API response processing."""
        mock_call.return_value = (
            "1. Category Explanation: This item is Organic as it is biodegradable.\n"
            "2. Disposal Instructions: Place in green compost cart.\n"
            "3. Composting: Suitable for home and municipal composting.\n"
            "4. Reuse: Use trimmings for stock.\n"
            "5. Environmental Benefits: Reduces methane.\n"
            "6. Safety Tip: Wash hands after handling.\n\n"
            "Note: Local municipal rules may vary."
        )

        res = self.assistant.generate_recommendations("Organic", 0.94, item_description="banana peel")

        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["category"], "Organic")
        self.assertEqual(res["confidence"], 0.94)
        self.assertIn("Google Gemini API", res["source"])
        self.assertIn("Category Explanation", res["formatted_text"])
        mock_call.assert_called_once()

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_api_network_exception_triggers_fallback(self, mock_call):
        """Test that API network / connection exceptions trigger graceful rule-based fallback."""
        mock_call.side_effect = ConnectionError("Could not resolve host: generativelanguage.googleapis.com")

        res = self.assistant.generate_recommendations("Recyclable", 0.85)

        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["category"], "Recyclable")
        self.assertIn("Rule-Based Fallback", res["source"])
        self.assertIn("ConnectionError", res["error"])
        # Fallback text must still be fully populated
        self.assertIn("Recyclable Waste Guidance", res["formatted_text"])

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_api_quota_exhaustion_triggers_fallback(self, mock_call):
        """Test that HTTP 429 quota exhaustion triggers graceful fallback."""
        mock_call.side_effect = RuntimeError("Gemini API quota exceeded (HTTP 429: Resource Exhausted).")

        res = self.assistant.generate_recommendations("Organic", 0.91)

        self.assertTrue(res["is_fallback"])
        self.assertIn("Rule-Based Fallback", res["source"])
        self.assertIn("quota exceeded", res["error"])

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_empty_api_response_triggers_fallback(self, mock_call):
        """Test that empty or blank API responses trigger fallback."""
        mock_call.return_value = "   "

        res = self.assistant.generate_recommendations("Recyclable", 0.89)

        self.assertTrue(res["is_fallback"])
        self.assertIn("empty response", res["error"])

    def test_classifier_separation(self):
        """Test that the assistant never mutates or overrides the classifier's predicted category."""
        assistant = GenAIRecyclingAssistant(api_key="")

        # Test Organic stays Organic
        res_o = assistant.generate_recommendations("Organic", 0.99, "plastic bottle")
        self.assertEqual(res_o["category"], "Organic")

        # Test Recyclable stays Recyclable
        res_r = assistant.generate_recommendations("Recyclable", 0.99, "apple core")
        self.assertEqual(res_r["category"], "Recyclable")

    @patch.object(GenAIRecyclingAssistant, "_call_gemini_api")
    def test_convenience_function(self, mock_call):
        """Test get_recycling_advice top-level helper function."""
        mock_call.return_value = "Mocked advice text."
        res = get_recycling_advice("Organic", 0.90, api_key="mock_valid_api_key_12345")
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["category"], "Organic")

    def test_default_model_and_configurability(self):
        """Test default model is gemini-3.8-flash and can be customized."""
        from src.genai_assistant import DEFAULT_GEMINI_MODEL
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.8-flash")

        # Test initialization default
        default_inst = GenAIRecyclingAssistant(api_key="")
        self.assertEqual(default_inst.model_name, "gemini-3.8-flash")

        # Test custom model
        custom_inst = GenAIRecyclingAssistant(api_key="", model_name="gemini-2.5-flash")
        self.assertEqual(custom_inst.model_name, "gemini-2.5-flash")

    def test_backend_identification_in_fallback(self):
        """Test that fallback response explicitly identifies the rule-based backend."""
        assistant = GenAIRecyclingAssistant(api_key="")
        res = assistant.generate_recommendations("Organic", 0.95)
        self.assertEqual(res["backend"], "Rule-Based Fallback Engine")


if __name__ == "__main__":
    unittest.main()
