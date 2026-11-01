"""
Manual Live Google Gemini API Verification Script for WasteWise.

This script is intentionally located outside the `tests/` directory and does NOT follow
the `test_*.py` naming convention so it is NEVER invoked during automated test discovery.

Usage:
    # 1. Safe offline verification / fallback demonstration:
    python scripts/verify_live_gemini.py

    # 2. Live API verification against Google Gemini API (requires active GEMINI_API_KEY):
    python scripts/verify_live_gemini.py --live

    # 3. Live API verification with custom model:
    python scripts/verify_live_gemini.py --live --model gemini-2.5-flash

Security Notice:
    API keys are strictly masked (e.g. 'AIza...78'). Raw keys are never logged or exposed.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.genai_assistant import (
    GenAIRecyclingAssistant,
    mask_api_key,
    load_env_variables,
    DEFAULT_GEMINI_MODEL,
)


def run_live_verification(live: bool, model: str, category: str, confidence: float, item: str) -> None:
    print("=" * 70)
    print(" WasteWise Recycling Assistant - Gemini API Integration Verifier")
    print("=" * 70)

    # 1. Load and check environment configuration
    load_env_variables()
    raw_key = os.environ.get("GEMINI_API_KEY", "")
    masked = mask_api_key(raw_key)

    print(f"[*] Environment Status:")
    print(f"    - Target Model   : {model}")
    print(f"    - API Key Status : {masked}")
    print(f"    - Live Flag      : {'ENABLED (--live passed)' if live else 'DISABLED (Offline Mode)'}")

    assistant = GenAIRecyclingAssistant(model_name=model)
    is_configured = assistant.is_api_key_configured()

    print(f"    - Key Configured : {is_configured}")
    print("-" * 70)

    if not live:
        print("[i] OFFLINE / FALLBACK DEMONSTRATION")
        print("    The --live flag was not specified. Performing local verification")
        print("    of the rule-based fallback system without contacting external APIs.\n")

        start = time.perf_counter()
        result = assistant.generate_recommendations(category, confidence, item)
        elapsed = time.perf_counter() - start

        print(f"    [+] Source       : {result['source']}")
        print(f"    [+] Is Fallback  : {result['is_fallback']}")
        print(f"    [+] Backend      : {result.get('backend', 'N/A')}")
        print(f"    [+] Latency      : {elapsed*1000:.1f} ms")
        print(f"    [+] Advice Snippet:\n")
        snippet = "\n".join("        " + line for line in result["formatted_text"].splitlines()[:10])
        print(snippet)
        print("        ...")
        print("\n" + "-" * 70)
        print("[*] To perform a live API verification against Google Gemini:")
        print("    1. Create or edit .env in the project root:")
        print("       GEMINI_API_KEY=AIzaSyYourActualKeyHere")
        print(f"       GEMINI_MODEL={model}")
        print("    2. Run with the --live flag:")
        print("       python scripts/verify_live_gemini.py --live\n")
        return

    # User explicitly requested live call
    if not is_configured:
        print("[!] ERROR: Live verification requested (--live), but no valid GEMINI_API_KEY")
        print("    was detected in the environment or .env file.")
        print(f"    Current API key evaluation: {masked}")
        print("    Please set a valid key in .env and retry.")
        return

    print(f"[*] Executing live API request for item: '{item}' ({category}, {confidence*100:.1f}%)...")
    print(f"    Sending metadata to model: {model}")

    start = time.perf_counter()
    result = assistant.generate_recommendations(category, confidence, item)
    elapsed = time.perf_counter() - start

    print("-" * 70)
    print(f"[*] Execution Summary:")
    print(f"    - Status      : {result.get('status', 'N/A')}")
    print(f"    - Source      : {result.get('source', 'N/A')}")
    print(f"    - Backend     : {result.get('backend', 'N/A')}")
    print(f"    - Is Fallback : {result.get('is_fallback', False)}")
    print(f"    - Latency     : {elapsed:.2f} s")

    if result.get("is_fallback"):
        print("\n[!] NOTICE: API call did not succeed; rule-based fallback engaged gracefully.")
        print(f"    Reported Diagnostic: {result.get('error')}")
        print("\n    Troubleshooting tips:")
        print(f"    - If model '{model}' is not found (404), your API key tier may require")
        print("      'gemini-2.5-flash', 'gemini-2.0-flash', or 'gemini-1.5-flash'.")
        print("      You can override the model with: --model gemini-2.5-flash")
        print("    - Verify your key permissions at https://aistudio.google.com/")
    else:
        print("\n[+] SUCCESS: Live Gemini response received!")
        print("\n[*] Response Preview:\n")
        preview = "\n".join("    " + line for line in result["formatted_text"].splitlines()[:16])
        print(preview)
        print("    ...")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Manual live verification utility for WasteWise Google Gemini integration."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute a live network request to the Gemini API using GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Target Gemini model (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Recyclable",
        help="Waste category to test (Organic or Recyclable).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.94,
        help="Classifier confidence score (0.0 to 1.0).",
    )
    parser.add_argument(
        "--item",
        type=str,
        default="aluminum soda can",
        help="Item description text.",
    )

    args = parser.parse_args()
    run_live_verification(
        live=args.live,
        model=args.model,
        category=args.category,
        confidence=args.confidence,
        item=args.item,
    )


if __name__ == "__main__":
    main()
