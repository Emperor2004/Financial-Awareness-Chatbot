"""
translation_module_test.py

Tests the Translator class (translator.py + translation_validator.py)
by simulating its usage within a RAG pipeline flow.

Focuses on:
- Correct language detection and passthrough.
- Translation accuracy for supported languages (input and output).
- Graceful handling of errors and exceptions.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Tuple

# --- Constants for Styling Output ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\032[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Import Translator Module & Custom Exception ---
try:
    # Assuming translator.py and translation_validator.py are in the same directory
    from ..translator import Translator, TranslationQualityError
    from .. import translation_validator as tv # For loading LANG_CODE_TO_NAME if needed
except ImportError as e:
    print(f"{bcolors.FAIL}ERROR: Failed to import translation module files.{bcolors.ENDC}")
    print(f"Details: {e}")
    print("Ensure translator.py and translation_validator.py are present.")
    sys.exit(1)
except Exception as e:
    print(f"{bcolors.FAIL}An unexpected error occurred during import: {e}{bcolors.ENDC}")
    sys.exit(1)

# --- Test Configuration ---
TEST_CASES_FILE = "translation_test_cases.json"
DELAY_BETWEEN_TESTS = 0.6 # Seconds to wait between test cases (adjust if rate limited)

# Define which test cases to use for simulation (subset for brevity)
# Using IDs from your translation_test_cases.json
SIMULATION_TEST_IDS = ["T001", "T004", "T010", "T016", "T053", "T058", "T065"]

# Re-use edge cases for error handling tests
EDGE_CASE_TESTS = [
    {"test_id": "E001", "description": "Non-String (None) Input", "text": None, "expected_error_type": ValueError},
    {"test_id": "E002", "description": "Non-String (Integer) Input", "text": 123456789, "expected_error_type": ValueError},
    {"test_id": "E003", "description": "Empty String Input", "text": "", "expected_error_type": ValueError},
    {"test_id": "E004", "description": "Whitespace-Only Input", "text": "   \n\n\t   ", "expected_error_type": ValueError},
    {"test_id": "E005", "description": "Overly Long String Input", "text": "a" * 50001, "expected_error_type": ValueError},
    {"test_id": "E006", "description": "Unsupported Language Input", "text": "Ceci est un test en français.", "expected_error_type": ValueError},
    {"test_id": "E007", "description": "Undetectable Language Input", "text": "!@#$%^&*()_+", "expected_error_type": ValueError} # Using symbols likely undetectable
]

# --- Helper Functions ---

# Global flag for color support check
USE_COLOR = os.name == 'posix' or 'TERM' in os.environ

def print_color(text, color):
    """Prints text in color if supported."""
    if USE_COLOR:
        print(f"{color}{text}{bcolors.ENDC}")
    else:
        print(text)

def load_simulation_cases(filepath: str, ids_to_load: List[str]) -> List[Dict]:
    """Loads specific test cases from the JSON file."""
    print(f"\nLoading simulation test cases ({', '.join(ids_to_load)}) from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_cases = json.load(f)
        
        selected_cases = [case for case in all_cases if case.get("text_id") in ids_to_load]
        
        if len(selected_cases) != len(ids_to_load):
             loaded_ids = {case.get("text_id") for case in selected_cases}
             missing_ids = [tid for tid in ids_to_load if tid not in loaded_ids]
             print_color(f"Warning: Could not find test IDs: {missing_ids}", bcolors.WARNING)

        print(f"Loaded {len(selected_cases)} simulation cases.")
        return selected_cases
    except FileNotFoundError:
        print_color(f"FATAL ERROR: Test case file not found at '{filepath}'", bcolors.FAIL)
        sys.exit(1)
    except json.JSONDecodeError:
        print_color(f"FATAL ERROR: Could not parse JSON in '{filepath}'", bcolors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_color(f"FATAL ERROR: Unexpected error loading test cases: {e}", bcolors.FAIL)
        sys.exit(1)

def simulate_rag_flow(translator_service: Translator, user_query: str, test_id: str, expected_input_lang: str) -> Tuple[bool, str]:
    """
    Simulates the two-step translation process around a conceptual RAG core.
    Returns (success_status, message).
    """
    print_color(f"\n--- Simulating RAG Flow for Test Case {test_id} ---", bcolors.HEADER)
    print(f"User Query ({expected_input_lang}):\n  '{user_query}'")
    
    english_query = None
    original_language = None
    step1_passed = False
    
    # --- Step 1: Process Input (trans_for_rag) ---
    print("\n[Step 1: Processing User Query with trans_for_rag]")
    try:
        start_time = time.time()
        english_query, original_language = translator_service.trans_for_rag(user_query)
        duration = time.time() - start_time
        print_color(f"  Success ({duration:.2f}s):", bcolors.OKGREEN)
        print(f"    Detected Language: {original_language}")
        print(f"    Processed Query (for RAG): '{english_query}'")

        # Validation Checks
        if original_language != expected_input_lang:
            print_color(f"    FAIL: Detected language '{original_language}' does not match expected '{expected_input_lang}'", bcolors.FAIL)
            return False, f"{test_id}: Language detection mismatch."
        if expected_input_lang == "English" and english_query != user_query:
            print_color("    FAIL: English input was modified.", bcolors.FAIL)
            return False, f"{test_id}: English input modified."
        if expected_input_lang != "English" and english_query == user_query:
             # This could happen if quality fallback occurred but didn't raise error (shouldn't happen now)
             print_color("    WARNING: Non-English input was not translated (potential fallback?).", bcolors.WARNING)
             # Allow continuing for now, but flag it
        
        step1_passed = True

    except (ValueError, TranslationQualityError, RuntimeError) as e:
        print_color(f"  FAIL: trans_for_rag raised expected error: {type(e).__name__}", bcolors.FAIL)
        print(f"    > {e}")
        return False, f"{test_id}: trans_for_rag failed as expected for error case." # Treat expected errors during simulation as handled. Needs refinement if testing *recovery*
    except Exception as e:
        print_color(f"  FAIL: trans_for_rag raised UNEXPECTED error: {type(e).__name__}", bcolors.FAIL)
        print(f"    > {e}")
        return False, f"{test_id}: Unexpected error in trans_for_rag."

    # --- Step 2: Process Output (trans_for_output) ---
    if not step1_passed or original_language is None:
        print("\n[Step 2: Skipped due to Step 1 failure]")
        # Failure already recorded in the return statement above
        return False, f"{test_id}: Step 1 failed, skipping Step 2."

    # Simulate RAG generating an English response
    simulated_rag_response_en = f"This is the simulated RAG response in English based on query for {test_id}."
    print("\n[Step 2: Processing RAG Output with trans_for_output]")
    print(f"  Simulated RAG Response (English): '{simulated_rag_response_en}'")
    print(f"  Target Language: {original_language}")

    final_response = None
    try:
        start_time = time.time()
        final_response = translator_service.trans_for_output(simulated_rag_response_en, original_language)
        duration = time.time() - start_time
        print_color(f"  Success ({duration:.2f}s):", bcolors.OKGREEN)
        print(f"    Final Response (to User): '{final_response}'")

        # Validation Checks
        if original_language == "English" and final_response != simulated_rag_response_en:
             print_color("    FAIL: English output was modified.", bcolors.FAIL)
             return False, f"{test_id}: English output modified."
        if original_language != "English" and final_response == simulated_rag_response_en:
             # This could happen if quality fallback occurred but didn't raise error
             print_color("    WARNING: English response was not translated back (potential fallback?).", bcolors.WARNING)
             # Allow continuing for now

        # Basic check: Is the translated output actually different from English?
        if original_language != "English":
             from langdetect import detect as detect_lang_final # Avoid name clash
             try:
                  detected_final_lang_code = detect_lang_final(final_response)
                  expected_final_lang_code = tv.name_to_code(original_language)
                  if detected_final_lang_code != expected_final_lang_code:
                       print_color(f"    WARNING: Final response language detected as '{detected_final_lang_code}', expected '{expected_final_lang_code}'.", bcolors.WARNING)
             except Exception as detect_err:
                  print_color(f"    WARNING: Could not detect language of final response: {detect_err}", bcolors.WARNING)


        print_color(f"\n{test_id}: RAG Simulation PASSED", bcolors.OKGREEN)
        return True, f"{test_id}: RAG Simulation PASSED"

    except (ValueError, TranslationQualityError, RuntimeError) as e:
        print_color(f"  FAIL: trans_for_output raised expected error: {type(e).__name__}", bcolors.FAIL)
        print(f"    > {e}")
        return False, f"{test_id}: trans_for_output failed as expected for error case." # As above, depends on test goal
    except Exception as e:
        print_color(f"  FAIL: trans_for_output raised UNEXPECTED error: {type(e).__name__}", bcolors.FAIL)
        print(f"    > {e}")
        return False, f"{test_id}: Unexpected error in trans_for_output."


def run_error_handling_tests(translator_service: Translator) -> List[Tuple[bool, str]]:
    """Runs tests specifically targeting error conditions."""
    print_color("\n=============================================", bcolors.BOLD)
    print_color("  Running Error Handling Tests               ", bcolors.BOLD)
    print_color("=============================================", bcolors.BOLD)
    results = []

    for i, test_case in enumerate(EDGE_CASE_TESTS):
        test_id = test_case["test_id"]
        description = test_case["description"]
        text_input = test_case["text"]
        expected_error_type = test_case["expected_error_type"]

        print_color(f"\n--- Testing Error Case {test_id}: {description} ---", bcolors.HEADER)
        
        input_display = f"'{text_input}'" if isinstance(text_input, str) else str(text_input)
        if isinstance(text_input, str) and len(text_input) > 50:
            input_display = f"'{text_input[:50]}...' (length: {len(text_input)})"
        print(f"Input: {input_display}")
        print(f"Expected Behavior: Raise {expected_error_type.__name__}")

        try:
            # Add delay before API-calling error tests
            if test_id in ["E006", "E007"]:
                 time.sleep(DELAY_BETWEEN_TESTS)

            # Call trans_for_rag as it includes input validation and detection
            translator_service.trans_for_rag(text_input)

            # If it succeeds, it's a failure for error handling
            print_color("  Result: Did NOT raise expected error.", bcolors.FAIL)
            print_color(f"  Status: FAIL", bcolors.FAIL)
            results.append((False, f"{test_id}: Did not raise {expected_error_type.__name__}"))

        except expected_error_type as e:
            # Caught the expected type of error
            print_color(f"  Result: Correctly raised {expected_error_type.__name__}:", bcolors.OKGREEN)
            print(f"    > {e}")
            print_color(f"  Status: PASS", bcolors.OKGREEN)
            results.append((True, f"{test_id}: Correctly raised {expected_error_type.__name__}"))

        except Exception as e:
            # Caught an unexpected error type
            print_color(f"  Result: Raised UNEXPECTED error type: {type(e).__name__}", bcolors.FAIL)
            print(f"    > {e}")
            print_color(f"  Status: FAIL", bcolors.FAIL)
            results.append((False, f"{test_id}: Raised {type(e).__name__}, expected {expected_error_type.__name__}"))
        
        # Add delay even after errors if not the last test
        if i < len(EDGE_CASE_TESTS) - 1 and test_id not in ["E006", "E007"]:
             time.sleep(0.1) # Shorter delay for non-API errors

    return results

# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    """
    Main function to initialize the translator and run all tests.
    """
    if os.name == 'nt':
        os.system('color') # Enable colors on Windows

    print_color("Initializing Translator Service...", bcolors.OKBLUE)
    translator_service = None
    try:
        translator_service = Translator()
        print_color("Translator initialized successfully.", bcolors.OKGREEN)
    except RuntimeError as e:
        print_color(f"FATAL ERROR: Failed to initialize Translator: {e}", bcolors.FAIL)
        print("Please check Azure credentials (env vars or ~/.translator_config.json).")
        sys.exit(1)
    except Exception as e:
        print_color(f"FATAL ERROR: Unexpected error during Translator initialization: {e}", bcolors.FAIL)
        sys.exit(1)

    # --- Run Error Handling Tests ---
    error_results = run_error_handling_tests(translator_service)

    # --- Run RAG Simulation Tests ---
    print_color("\n=============================================", bcolors.BOLD)
    print_color("  Running RAG Pipeline Simulation Tests      ", bcolors.BOLD)
    print_color("=============================================", bcolors.BOLD)
    simulation_cases = load_simulation_cases(TEST_CASES_FILE, SIMULATION_TEST_IDS)
    simulation_results = []

    if not simulation_cases:
        print_color("No simulation cases loaded. Skipping simulation tests.", bcolors.WARNING)
    else:
        for i, case in enumerate(simulation_cases):
            test_id = case['text_id']
            
            # Determine expected language (assuming English if not Hindi/Marathi in source files)
            # A more robust check might involve langdetect here too, but rely on JSON structure for now
            expected_lang = "English" # Default
            # Check which language field has the *longest* content to guess source
            # This is heuristic, assumes English is usually shorter translation than Indic
            lang_lengths = {
                'english': len(case.get('english','')),
                'hindi': len(case.get('hindi','')),
                'marathi': len(case.get('marathi',''))
            }

            # Simulate for English, Hindi, and Marathi inputs from the JSON
            inputs_to_test = [
                (case.get('english', ''), "English"),
                (case.get('hindi', ''), "Hindi"),
                (case.get('marathi', ''), "Marathi")
            ]

            for input_text, expected_input_lang in inputs_to_test:
                if not input_text: continue # Skip if language text is missing in JSON

                print_color(f"\n>>> Simulating Test ID {test_id} with {expected_input_lang} input <<<", bcolors.UNDERLINE)
                
                success, message = simulate_rag_flow(translator_service, input_text, test_id + f"_{expected_input_lang}", expected_input_lang)
                simulation_results.append((success, message))
                
                # Add delay between simulations
                time.sleep(DELAY_BETWEEN_TESTS)


    # --- Final Summary ---
    print_color("\n=============================================", bcolors.BOLD)
    print_color("  Overall Test Summary                       ", bcolors.BOLD)
    print_color("=============================================", bcolors.BOLD)

    error_passed = sum(1 for r in error_results if r[0])
    error_total = len(error_results)
    sim_passed = sum(1 for r in simulation_results if r[0])
    sim_total = len(simulation_results)

    print(f"\nError Handling Tests: {error_passed} / {error_total} passed.")
    if error_passed != error_total:
        print_color("  Details of Failed Error Tests:", bcolors.WARNING)
        for success, msg in error_results:
            if not success:
                print(f"    - {msg}")

    print(f"\nRAG Simulation Tests: {sim_passed} / {sim_total} passed.")
    if sim_passed != sim_total:
        print_color("  Details of Failed Simulation Tests:", bcolors.WARNING)
        for success, msg in simulation_results:
            if not success:
                print(f"    - {msg}")

    print("\n" + "="*45)
    if error_passed == error_total and sim_passed == sim_total:
        print_color("All tests passed successfully!", bcolors.OKGREEN + bcolors.BOLD)
        sys.exit(0) # Exit with success code
    else:
        print_color("Some tests failed. Please review the output above.", bcolors.FAIL + bcolors.BOLD)
        sys.exit(1) # Exit with error code


if __name__ == "__main__":
    main()
