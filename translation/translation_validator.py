"""
translation_validator.py

Provides validation helpers, configuration constants, and offline evaluation
logic for the translator module.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Tuple, Any
import json
import statistics
import os
import time # For preflight delay
from pathlib import Path

# --- Configuration Constants ---

# Language maps
LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}
NAME_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_NAME.items()}

# Operational threshold for runtime validation (combined score)
# Combined = 0.6 * sequence_ratio + 0.4 * word_overlap
# This threshold determines if a translation is considered "good enough"
# based on round-trip consistency. If below this, TranslationQualityError is raised.
MIN_COMBINED_QUALITY_SCORE = 0.5

# Path for offline evaluation report (not used by core translator)
REPORT_PATH = Path("translation_validation_report.json")


# --- Core Validation & Metric Functions ---

def check_input(text: str | Any) -> None:
    """
    Basic validation for input text. Used by Translator class methods.
    Ensures input is a non-empty string within length limits.

    Raises:
        ValueError: On invalid inputs (not string, empty, too long).
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace")
    # Arbitrary limit to avoid extremely large inputs
    MAX_LENGTH = 50_000
    if len(text) > MAX_LENGTH:
        raise ValueError(f"Input text is too long (>{MAX_LENGTH} chars)")


def code_to_name(code: str) -> str | None:
    """Converts a language code (e.g., 'hi') to its name ('Hindi')."""
    return LANG_CODE_TO_NAME.get(code)


def name_to_code(name: str) -> str | None:
    """Converts a language name (e.g., 'Hindi') to its code ('hi')."""
    return NAME_TO_LANG_CODE.get(name)


def _sequence_ratio(a: str, b: str) -> float:
    """Calculates the difflib SequenceMatcher ratio (char-level similarity)."""
    if not isinstance(a, str) or not isinstance(b, str): return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_overlap(a: str, b: str) -> float:
    """Calculates the ratio of common words (set intersection) to words in 'a'."""
    if not isinstance(a, str) or not isinstance(b, str): return 0.0
    a_words = set(w for w in a.lower().split() if w) # Use lowercase set for robust comparison
    b_words = set(w for w in b.lower().split() if w)
    if not a_words:
        return 1.0 if not b_words else 0.0 # Handle empty strings
    common = a_words.intersection(b_words)
    return len(common) / len(a_words)


def validate_roundtrip(original_text: str, roundtrip_text: str) -> Dict[str, float]:
    """
    Computes quality metrics by comparing original text to its round-tripped version.
    Used for runtime validation within the Translator class.

    Returns:
        A dict with keys: 'sequence_ratio', 'word_overlap', 'combined'.
    """
    seq = _sequence_ratio(original_text, roundtrip_text)
    wov = _word_overlap(original_text, roundtrip_text)
    # Weighted combination: 60% sequence similarity, 40% word overlap
    combined = seq * 0.6 + wov * 0.4
    return {"sequence_ratio": seq, "word_overlap": wov, "combined": combined}


def is_acceptable(metrics: Dict[str, float], min_combined: float = MIN_COMBINED_QUALITY_SCORE) -> bool:
    """
    Checks if the computed validation metrics meet the minimum quality threshold.
    Used by Translator class methods.
    """
    # Ensure metrics dict is valid and contains the 'combined' score
    if not isinstance(metrics, dict): return False
    return metrics.get("combined", 0.0) >= min_combined


# --- Preflight Check (For setup verification) ---

def preflight_check() -> Dict:
    """
    Runs quick checks: Azure credentials, basic API connectivity, and language detection sanity.
    Instantiates the Translator class for testing.

    Returns:
        A dict with status ('credentials', 'basic_translate') and any error messages.
    """
    # Import locally AFTER defining helpers to avoid potential circularity issues
    # and to allow this file to be imported without immediate Azure dependency.
    from translator import Translator, TranslationQualityError

    status = {"credentials": False, "basic_translate": False, "errors": []}
    translator_instance = None

    # 1. Check Credentials (by attempting instantiation)
    try:
        translator_instance = Translator()
        status["credentials"] = True
        print("Preflight: Credentials loaded successfully.")
    except RuntimeError as e:
        status["errors"].append(f"Credential Check Failed: {e}")
    except Exception as e:
        status["errors"].append(f"Unexpected error during Translator instantiation: {e}")

    # 2. Basic Translate Test (if credentials loaded)
    if status["credentials"] and translator_instance:
        print("Preflight: Testing basic translation calls...")
        # Use slightly longer, less ambiguous phrases
        samples = [
            ("यह एक परीक्षण है।", "Hindi"),
            ("नमस्कार, कसे आहात?", "Marathi"), # "Hello, how are you?"
            ("This is a test.", "English")
        ]
        basic_translate_ok = True
        for i, (txt, expected_lang) in enumerate(samples):
            try:
                # Add a small delay between preflight checks
                if i > 0: time.sleep(0.5)

                # Test trans_for_rag
                translated_text, detected_lang = translator_instance.trans_for_rag(txt)

                if detected_lang != expected_lang:
                     status["errors"].append(f"Language detection mismatch for '{txt[:10]}...': Expected {expected_lang}, Got {detected_lang}")
                     basic_translate_ok = False
                     continue # Don't proceed if detection failed

                if expected_lang == "English":
                     if translated_text != txt:
                          status["errors"].append(f"English passthrough failed: Input '{txt}', Output '{translated_text}'")
                          basic_translate_ok = False
                elif not translated_text or translated_text == txt: # Should have translated
                     status["errors"].append(f"Translation failed or returned original text for {expected_lang}: '{txt}' -> '{translated_text}'")
                     basic_translate_ok = False
                else:
                    print(f"  - Preflight {expected_lang} -> English: OK ('{txt[:10]}...' -> '{translated_text[:10]}...')")
                    # Optionally test trans_for_output (adds more API calls)
                    # try:
                    #     time.sleep(0.5)
                    #     back_to_orig = translator_instance.trans_for_output(translated_text, expected_lang)
                    #     if not back_to_orig or back_to_orig == translated_text:
                    #          status["errors"].append(f"English -> {expected_lang} failed: '{translated_text[:10]}...' -> '{back_to_orig[:10]}...'")
                    #          basic_translate_ok = False
                    #     else:
                    #         print(f"  - Preflight English -> {expected_lang}: OK")
                    # except (RuntimeError, TranslationQualityError, ValueError) as e_out:
                    #      status["errors"].append(f"trans_for_output failed for {expected_lang}: {e_out}")
                    #      basic_translate_ok = False


            except (RuntimeError, TranslationQualityError, ValueError) as e_rag:
                status["errors"].append(f"trans_for_rag failed for {expected_lang} sample: {e_rag}")
                basic_translate_ok = False
            except Exception as e_unexp:
                 status["errors"].append(f"Unexpected error during preflight translation test ({expected_lang}): {e_unexp}")
                 basic_translate_ok = False

        if basic_translate_ok and not status["errors"]: # Check errors again in case detection failed
             status["basic_translate"] = True
             print("Preflight: Basic translation tests PASSED.")
        else:
             print("Preflight: Basic translation tests FAILED (see errors below).")


    # Print summary of errors if any
    if status["errors"]:
         print("\nPreflight Check Errors Encountered:")
         for err in status["errors"]:
              print(f"  - {err}")
    else:
         print("\nPreflight Check Completed Successfully.")

    return status

# --- Keep Offline Evaluation Functions (Optional) ---
# These are not directly used by the Translator class but might be useful
# for separate, more extensive testing scripts like test_translation.py

def round_trip_similarity(source_text: str) -> Dict[str, float | str]:
    """
    Offline evaluation: Performs a round-trip using a Translator instance.
    Imports Translator locally to avoid circular dependencies if this file
    is imported elsewhere without translator.py necessarily being ready.
    """
    from translator import Translator, TranslationQualityError # Local import

    translator_instance = Translator() # Assumes credentials are set
    check_input(source_text)
    metrics = {"sequence_ratio": 0.0, "word_overlap": 0.0, "combined": 0.0, "error": None}

    try:
        english_text, detected_lang = translator_instance.trans_for_rag(source_text)
        if detected_lang == "English": # No round trip possible if source is English
             metrics = {"sequence_ratio": 1.0, "word_overlap": 1.0, "combined": 1.0, "error": None}
             return metrics

        roundtrip_text = translator_instance.trans_for_output(english_text, detected_lang)
        # Calculate metrics using the core validation logic
        metrics = validate_roundtrip(source_text, roundtrip_text)
        metrics["error"] = None # Ensure error is None if successful

    except (ValueError, RuntimeError, TranslationQualityError) as e:
        metrics["error"] = str(e)
    except Exception as e:
         metrics["error"] = f"Unexpected error: {e}"

    return metrics


def evaluate_dataset(texts: List[str], quality_threshold: float = 0.86) -> Dict:
    """
    Offline evaluation: Evaluates a list of source texts using round_trip_similarity.
    Writes a report if average quality is below a threshold.
    """
    if not isinstance(texts, list) or not texts:
        raise ValueError("Provide a non-empty list of texts to evaluate")

    results: List[Dict] = []
    combined_scores: List[float] = []
    has_errors = False

    print(f"Evaluating dataset of {len(texts)} texts...")
    for i, t in enumerate(texts):
        # Add delay between eval calls
        if i > 0: time.sleep(0.5)
        print(f"  Processing item {i+1}/{len(texts)}...")
        metrics = round_trip_similarity(t)
        results.append({"index": i, "source": t, "metrics": metrics})
        combined_scores.append(metrics.get("combined", 0.0))
        if metrics.get("error"):
             has_errors = True
             print(f"    Error on item {i+1}: {metrics['error']}")


    avg_combined = statistics.mean(combined_scores) if combined_scores else 0.0
    median_combined = statistics.median(combined_scores) if combined_scores else 0.0

    report = {
        "num_samples": len(texts),
        "average_combined_score": avg_combined,
        "median_combined_score": median_combined,
        "quality_threshold": quality_threshold,
        "evaluation_details": results,
        "errors_occurred": has_errors,
        "recommended_action": None,
        "action_reason": None
    }

    # Decide on action based on average score
    if avg_combined < quality_threshold:
        report["recommended_action"] = "disable_runtime_validation_or_review"
        report["action_reason"] = (
            f"Average combined score ({avg_combined:.4f}) is below threshold ({quality_threshold}). "
            "Consider disabling runtime validation in the Translator class or reviewing the translation quality/provider."
        )
        # Persist report file
        try:
            print(f"Writing detailed evaluation report to: {REPORT_PATH}")
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write evaluation report: {e}")
    else:
        report["recommended_action"] = "keep_current_settings"
        report["action_reason"] = f"Average combined score ({avg_combined:.4f}) meets or exceeds the threshold ({quality_threshold})."

    print("Dataset evaluation complete.")
    print(f"  Average Combined Score: {avg_combined:.4f}")
    print(f"  Recommended Action: {report['recommended_action']}")

    return report
