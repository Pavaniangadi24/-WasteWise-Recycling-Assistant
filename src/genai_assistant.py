"""
Google Gemini AI Assistant and Fallback Orchestration Module for WasteWise.

Integrates the official Google Gemini API (preferring the unified `google-genai` SDK
with seamless fallback to direct Gemini REST endpoints) with automatic fallback to
the deterministic rule-based expert knowledge system (`src/recycling_rules.py`).

Data Transmission Notice:
Image classification is performed 100% locally by MobileNetV2 (no image bytes are uploaded).
When Gemini API is enabled with an active key, text metadata (predicted category, confidence
percentage, and optional user item description) is transmitted to Google's Gemini API to
generate contextual recycling advice. If the API key is absent, the system operates purely
offline using local expert rules.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.recycling_rules import get_fallback_recommendations

logger = logging.getLogger(__name__)

# Default Constants & Configuration
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS: int = 15
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "none", "null", "your_api_key"}


def load_env_variables(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file.
    Tries python-dotenv if installed, otherwise uses built-in line parser.
    """
    target = env_path if env_path is not None else (_project_root / ".env")
    if not target.is_file():
        return

    # Try python-dotenv first
    try:
        import dotenv
        dotenv.load_dotenv(dotenv_path=str(target), override=False)
        return
    except ImportError:
        pass

    # Built-in lightweight .env parser
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger.debug(f"Could not parse .env file: {e}")


# Initialize environment on module load
load_env_variables()


def mask_api_key(key: Optional[str]) -> str:
    """Mask an API key for safe logging, exposing only first 4 and last 2 characters."""
    if not key or len(key) < 8:
        return "[NOT_SET]"
    return f"{key[:4]}...{key[-2:]}"


def validate_confidence(confidence: float) -> float:
    """Validate and clamp confidence score to [0.0, 1.0]."""
    if not isinstance(confidence, (int, float)):
        raise TypeError(f"Confidence must be numeric, got {type(confidence).__name__}")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
    return float(confidence)


def build_gemini_prompt(category: str, confidence: float, item_description: Optional[str] = None) -> str:
    """
    Construct the expert prompt instructing Gemini to produce structured recycling advice.

    Args:
        category: Predicted category ('Organic' or 'Recyclable').
        confidence: Prediction confidence score in [0.0, 1.0].
        item_description: Optional user text description of the waste item.

    Returns:
        Formatted prompt string.
    """
    conf_pct = round(confidence * 100, 1)
    item_clause = (
        f"The user describes the item as: \"{item_description.strip()}\"."
        if item_description and item_description.strip()
        else "No specific item name was supplied by the user."
    )

    prompt = (
        "You are WasteWise AI, an expert environmental recycling and sustainability assistant.\n"
        "A computer vision image classifier has inspected an item of waste with the following result:\n"
        f"- Predicted Classification: {category}\n"
        f"- Classifier Confidence: {conf_pct}%\n"
        f"- Item Description: {item_clause}\n\n"
        "Please provide structured, beginner-friendly, and actionable waste-handling guidance.\n"
        "Your response MUST include the following 6 numbered sections:\n"
        f"1. Category Explanation: Explain simply why this waste is classified as {category}.\n"
        "2. Disposal Instructions: Step-by-step guidance on how and where to dispose of it (e.g. bin type, pre-cleaning, rinsing).\n"
        "3. Recycling / Composting Recommendations: Specific guidance on material recovery, home or municipal composting.\n"
        "4. Practical Reuse Ideas: Creative ways to upcycle or repurpose this type of item before discarding.\n"
        "5. Environmental Benefits: How proper disposal or diversion protects the ecosystem and reduces landfill emissions.\n"
        "6. Safety & Handling Tip: A practical safety precaution when handling this waste.\n\n"
        "CRITICAL GUIDELINES:\n"
        "- Do NOT claim that every single item in this category is accepted everywhere.\n"
        "- Explicitly mention that local municipal recycling facilities and sorting regulations vary by jurisdiction.\n"
        "- Keep the tone encouraging, concise, informative, and professional.\n"
        "- Format using clean markdown headers and bullet points."
    )
    return prompt


class GenAIRecyclingAssistant:
    """
    Orchestrates Generative AI requests to the Google Gemini API with seamless
    fallback to the local rule-based system if the API is offline or unconfigured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        # Refresh environment in case .env was recently created/updated
        load_env_variables()

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.last_backend_used: Optional[str] = None
        try:
            self.timeout = int(timeout or os.environ.get("GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        except (ValueError, TypeError):
            self.timeout = DEFAULT_TIMEOUT_SECONDS

    def is_api_key_configured(self) -> bool:
        """Check whether a non-placeholder API key is set."""
        if not self.api_key:
            return False
        clean_key = self.api_key.strip().lower()
        return clean_key not in PLACEHOLDER_KEYS and len(clean_key) > 10

    def generate_recommendations(
        self,
        category: str,
        confidence: float,
        item_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produce waste recommendations using Gemini AI, falling back to rule-based
        guidance if the API key is missing or any network/API error occurs.

        Args:
            category: 'Organic' or 'Recyclable'.
            confidence: Classifier confidence in [0.0, 1.0].
            item_description: Optional user-provided description.

        Returns:
            Dict[str, Any] containing recommendation text, metadata, and fallback flags.
        """
        confidence = validate_confidence(confidence)
        norm_cat = category.strip().capitalize()
        if norm_cat not in ["Organic", "Recyclable"]:
            norm_cat = "Organic" if "organ" in norm_cat.lower() else "Recyclable"

        # Check API key presence
        if not self.is_api_key_configured():
            logger.info("Gemini API key is not configured. Utilizing rule-based fallback.")
            fallback_res = get_fallback_recommendations(norm_cat, confidence, item_description)
            fallback_res["error"] = "Gemini API key is not configured or is a placeholder."
            fallback_res["backend"] = "Rule-Based Fallback Engine"
            return fallback_res

        prompt = build_gemini_prompt(norm_cat, confidence, item_description)

        # Attempt API invocation
        try:
            generated_text = self._call_gemini_api(prompt)
            if not generated_text or not generated_text.strip():
                raise ValueError("Received empty response from Gemini API.")

            backend = getattr(self, "last_backend_used", None) or "Google Gemini API"
            return {
                "status": "success",
                "source": f"Google Gemini API ({self.model_name})",
                "backend": backend,
                "is_fallback": False,
                "category": norm_cat,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 1),
                "item_description": item_description.strip() if item_description else None,
                "model_used": self.model_name,
                "formatted_text": generated_text.strip(),
            }

        except Exception as e:
            logger.warning(f"Gemini API call failed: {type(e).__name__}: {e}. Triggering rule-based fallback.")
            fallback_res = get_fallback_recommendations(norm_cat, confidence, item_description)
            fallback_res["error"] = f"Gemini API call failed ({type(e).__name__}): {str(e)}"
            fallback_res["backend"] = "Rule-Based Fallback Engine"
            return fallback_res

    def _call_gemini_api(self, prompt: str) -> str:
        """
        Dispatch request to Google Gemini API.
        Attempts official `google.genai` SDK first if available, then falls back to REST API via requests.
        """
        # Method A: Try official google-genai SDK if available
        try:
            from google import genai
            self.last_backend_used = "google-genai SDK"
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            if hasattr(response, "text") and response.text:
                return response.text
        except ImportError:
            pass  # SDK not installed; proceed to standard REST dispatch
        except Exception as sdk_err:
            logger.debug(f"google-genai SDK invocation failed: {sdk_err}. Falling back to REST API.")

        # Method B: Official Gemini REST API via `requests`
        self.last_backend_used = "Gemini REST API (requests)"
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000,
            },
        }

        resp = requests.post(
            url,
            params=params,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code == 429:
            raise RuntimeError("Gemini API quota exceeded (HTTP 429: Resource Exhausted).")
        if resp.status_code in [400, 401, 403]:
            err_msg = resp.text
            try:
                err_json = resp.json()
                err_msg = err_json.get("error", {}).get("message", err_msg)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API Authentication/Request error (HTTP {resp.status_code}): {err_msg}")
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API returned HTTP status {resp.status_code}: {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates returned by Gemini: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(f"No content parts returned in candidate: {candidates[0]}")

        text = "".join(p.get("text", "") for p in parts)
        return text


# Module-level convenience function
def get_recycling_advice(
    category: str,
    confidence: float,
    item_description: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience functional interface to generate recycling advice with fallback.
    """
    assistant = GenAIRecyclingAssistant(api_key=api_key, model_name=model_name)
    return assistant.generate_recommendations(category, confidence, item_description)


if __name__ == "__main__":
    print("Testing GenAIRecyclingAssistant (No key configured -> Expect Fallback):")
    assistant = GenAIRecyclingAssistant(api_key="")
    res = assistant.generate_recommendations("Organic", 0.95, "apple core")
    print(f"Source: {res['source']}")
    print(f"Is Fallback: {res['is_fallback']}")
    print(res["formatted_text"])
