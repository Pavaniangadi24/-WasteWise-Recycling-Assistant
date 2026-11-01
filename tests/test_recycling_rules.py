"""
Unit tests for WasteWise Rule-Based Fallback Recycling Knowledge Module.

Tests verify:
1. Organic waste fallback response content, sections, and disclaimer.
2. Recyclable waste fallback response content, sections, and disclaimer.
3. Item description contextualization.
4. Confidence score formatting and display.
5. Category normalization.
"""

import unittest
from src.recycling_rules import get_fallback_recommendations, FALLBACK_DISCLAIMER


class TestRecyclingRules(unittest.TestCase):
    def test_organic_fallback_response(self):
        """Test Organic waste fallback output structure and required 6 sections."""
        res = get_fallback_recommendations("Organic", 0.95)

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["category"], "Organic")
        self.assertEqual(res["confidence"], 0.95)
        self.assertEqual(res["confidence_percent"], 95.0)
        self.assertIn("Rule-Based Fallback", res["source"])

        # Check required 6 sections exist
        sections = res["sections"]
        self.assertIn("explanation", sections)
        self.assertIn("disposal_instructions", sections)
        self.assertIn("recycling_recommendations", sections)
        self.assertIn("reuse_suggestions", sections)
        self.assertIn("environmental_benefits", sections)
        self.assertIn("safety_tip", sections)

        # Check content keywords
        self.assertIn("compost", sections["explanation"].lower())
        self.assertIn("green or brown", sections["disposal_instructions"].lower())
        self.assertIn("methane", sections["environmental_benefits"].lower())

        # Check disclaimer
        self.assertEqual(res["disclaimer"], FALLBACK_DISCLAIMER)
        self.assertIn(FALLBACK_DISCLAIMER, res["formatted_text"])

    def test_recyclable_fallback_response(self):
        """Test Recyclable waste fallback output structure and required 6 sections."""
        res = get_fallback_recommendations("Recyclable", 0.92)

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["category"], "Recyclable")
        self.assertEqual(res["confidence"], 0.92)
        self.assertEqual(res["confidence_percent"], 92.0)

        sections = res["sections"]
        self.assertIn("rigid plastics", sections["explanation"].lower())
        self.assertIn("rinse", sections["disposal_instructions"].lower())
        self.assertIn("aluminum", sections["recycling_recommendations"].lower())

    def test_item_description_contextualization(self):
        """Test that user-supplied item description is included in formatted text."""
        res = get_fallback_recommendations("Organic", 0.88, item_description="banana peel")
        self.assertEqual(res["item_description"], "banana peel")
        self.assertIn("banana peel", res["formatted_text"])
        self.assertIn("88.0%", res["formatted_text"])

    def test_category_normalization(self):
        """Test handling case-insensitive and variant category strings."""
        res_lower = get_fallback_recommendations("organic", 0.90)
        self.assertEqual(res_lower["category"], "Organic")

        res_upper = get_fallback_recommendations("RECYCLABLE", 0.90)
        self.assertEqual(res_upper["category"], "Recyclable")

        res_phrase = get_fallback_recommendations("organic waste", 0.90)
        self.assertEqual(res_phrase["category"], "Organic")


if __name__ == "__main__":
    unittest.main()
