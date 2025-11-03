"""
test_translation.py

A new, organized test suite for the translation module.
This file will be expanded with more tests over time.

Part 1: Edge Case & Error Handling Tests
Part 2: Functional "Gold Standard" & NLP Evaluation

Logs all output to a timestamped .txt file.

Requires:
- pip install nltk rouge-score
"""

import sys
import os
import json
import statistics
import time
from datetime import datetime # <--- IMPORT ADDED
import io                    # <--- IMPORT ADDED
from typing import List, Dict, Any

# --- NLP Metrics Imports ---
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    from rouge import Rouge
except ImportError:
    print("ERROR: 'nltk' or 'rouge-score' not found.")
    print("Please run: pip install nltk rouge-score")
    sys.exit(1)

# --- Constants for Styling Output ---
class bcolors:
    """Class for terminal color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\032[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Import Translation Module ---
# We use a try/except block to provide a clean error message
# if the required files are missing or have credential errors.
try:
    from .. import translator
    from .. import translation_validator as tv
except ImportError:
    print(f"{bcolors.FAIL}ERROR: Failed to import 'translator.py' or 'translation_validator.py'.{bcolors.ENDC}")
    print("Please make sure both files are in the same directory as this script.")
    sys.exit(1)
except RuntimeError as e:
    print(f"{bcolors.FAIL}ERROR: Failed to initialize translation module.{bcolors.ENDC}")
    print(f"Details: {e}")
    print("Please ensure your Azure credentials (AZ_TRANSLATOR_KEY) are set correctly.")
    sys.exit(1)
except Exception as e:
    print(f"{bcolors.FAIL}An unexpected error occurred during import: {e}{bcolors.ENDC}")
    sys.exit(1)


# --- Helper to remove color codes for file logging ---
def strip_color_codes(text):
    """Removes ANSI escape codes from a string."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# ======================================================================
# PART 1: EDGE CASE & ERROR HANDLING
# ======================================================================

# --- Edge Case Test Data ---
EDGE_CASE_TESTS = [
    {
        "test_id": "E001",
        "description": "Non-String (None) Input",
        "text": None,
        "expected_error": "Input text must be a string"
    },
    {
        "test_id": "E002",
        "description": "Non-String (Integer) Input",
        "text": 123456789,
        "expected_error": "Input text must be a string"
    },
    {
        "test_id": "E003",
        "description": "Empty String Input",
        "text": "",
        "expected_error": "Input text is empty or whitespace"
    },
    {
        "test_id": "E004",
        "description": "Whitespace-Only Input",
        "text": "   \n\n\t   ",
        "expected_error": "Input text is empty or whitespace"
    },
    {
        "test_id": "E005",
        "description": "Overly Long String Input",
        "text": "a" * 50001,
        "expected_error": "Input text is too long"
    },
    {
        "test_id": "E006",
        "description": "Unsupported Language Input",
        "text": "Bonjour le monde. Ceci est un test.", # French
        "expected_error": "Language 'fr' not supported"
    },
    {
        "test_id": "E007",
        "description": "Undetectable Language Input",
        "text": "12345 67890 12345 67890 12345 67890",
        "expected_error": "Could not detect language"
    }
]


class EdgeCaseTester:
    """
    Runs a suite of tests designed to validate the error-handling
    capabilities of the translation module.
    """
    def __init__(self):
        self.passed_count = 0
        self.total_count = 0
        # Check if running in a terminal that supports color
        self.use_color = os.name == 'posix' or 'TERM' in os.environ

    def _print_color(self, text, color):
        """Helper to print in color only if supported."""
        if self.use_color:
            print(f"{color}{text}{bcolors.ENDC}")
        else:
            print(text)

    def _run_single_test(self, test_case):
        """
        Executes a single edge case test and prints the result.
        """
        self.total_count += 1
        test_id = test_case["test_id"]
        description = test_case["description"]
        text_input = test_case["text"]
        expected_error = test_case["expected_error"]

        header = f"--- Test {test_id}: {description} ---"
        self._print_color(f"\n{header}", bcolors.HEADER)

        # Format the input for clean printing
        input_display = f"'{text_input}'" if isinstance(text_input, str) else str(text_input)
        if isinstance(text_input, str) and len(text_input) > 50:
            input_display = f"'{text_input[:50]}...' (length: {len(text_input)})"

        print(f"Input: {input_display}")
        print(f"Expected Behavior: Raise ValueError with message '{expected_error}'")

        try:
            # --- Call the function under test ---
            # We add a small delay here too for tests E006 and E007
            if test_id in ["E006", "E007"]:
                time.sleep(0.5) # Add delay for API-calling error tests

            translator.trans_for_rag(text_input)

            # --- If it SUCCEEDS, it's a FAILURE ---
            self._print_color("Result: Module did NOT raise an exception.", bcolors.FAIL)
            self._print_color(f"Status: FAIL", bcolors.FAIL)

        except ValueError as e:
            # --- If it RAISES ValueError (as expected) ---
            actual_error = str(e)
            self._print_color(f"Result: Module raised ValueError:", bcolors.OKCYAN)
            print(f"  > '{actual_error}'")

            if actual_error == expected_error:
                self._print_color("Status: PASS", bcolors.OKGREEN)
                self.passed_count += 1
            else:
                self._print_color(f"Status: FAIL", bcolors.FAIL)
                print(f"  > Reason: Error message did not match expected message.")

        except Exception as e:
            # --- If it RAISES an UNEXPECTED error ---
            self._print_color(f"Result: Module raised an UNEXPECTED exception: {type(e).__name__}", bcolors.FAIL)
            print(f"  > {e}")
            self._print_color(f"Status: FAIL", bcolors.FAIL)

    def run_tests(self):
        """
        Runs all edge case tests and prints a final summary.
        """
        self._print_color("=============================================", bcolors.BOLD)
        self._print_color("  Running Part 1: Edge Case & Error Tests    ", bcolors.BOLD)
        self._print_color("=============================================", bcolors.BOLD)

        for test_case in EDGE_CASE_TESTS:
            self._run_single_test(test_case)

        # --- Print Final Summary ---
        print("\n" + "="*45)
        self._print_color("  Edge Case Test Summary", bcolors.BOLD)
        print("="*45)

        summary = f"Passed {self.passed_count} / {self.total_count} tests."

        if self.passed_count == self.total_count:
            self._print_color(summary, bcolors.OKGREEN)
        else:
            self._print_color(summary, bcolors.FAIL)
        print("="*45 + "\n")

        return self.passed_count == self.total_count


# ======================================================================
# PART 2: FUNCTIONAL & NLP EVALUATION
# ======================================================================

def setup_nltk():
    """Downloads required NLTK data."""
    print("Setting up NLTK (downloading 'punkt', 'wordnet', 'omw-1.4')...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    print("NLTK setup complete.")


def calculate_nlp_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculates BLEU, METEOR, and ROUGE-L scores.
    """
    metrics = {
        "BLEU": 0.0,
        "METEOR": 0.0,
        "ROUGE-L (f)": 0.0
    }

    try:
        # Pre-process: tokenize and lowercase
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        # BLEU
        chencherry = SmoothingFunction()
        metrics["BLEU"] = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)

        # METEOR (requires tokenized list)
        metrics["METEOR"] = meteor_score([ref_tokens], hyp_tokens)

        # ROUGE (requires non-tokenized strings)
        if hyp_tokens: # Avoid error on empty hypothesis
            rouge = Rouge()
            scores = rouge.get_scores(hypothesis, reference)
            metrics["ROUGE-L (f)"] = scores[0]['rouge-l']['f']

    except Exception as e:
        print(f"{bcolors.FAIL}  > Error calculating metrics: {e}{bcolors.ENDC}")

    return metrics


class FunctionalTester:
    """
    Runs the "gold standard" functional tests against the module
    and calculates NLP metrics.
    """
    def __init__(self, filepath="translation_test_cases.json", use_color=True):
        self.use_color = use_color
        self._print_color("=============================================", bcolors.BOLD)
        self._print_color("  Running Part 2: Functional & NLP Tests     ", bcolors.BOLD)
        self._print_color("=============================================", bcolors.BOLD)
        self.test_cases = self._load_test_cases(filepath)
        self.results = {
            "hi_en": [],
            "en_hi": [],
            "mr_en": [],
            "en_mr": []
        }
        # --- DELAY IN SECONDS ---
        # Set a delay to add between API calls to avoid rate limiting
        self.delay_between_calls = 0.5 # 0.5 seconds = 2 requests/sec

    def _print_color(self, text, color):
        """Helper to print in color only if supported."""
        if self.use_color:
            print(f"{color}{text}{bcolors.ENDC}")
        else:
            print(text)

    def _load_test_cases(self, filepath):
        """Loads the gold standard JSON file."""
        print(f"\nLoading test cases from '{filepath}'...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} test cases.")
            return data
        except FileNotFoundError:
            self._print_color(f"FATAL ERROR: Test case file not found at '{filepath}'", bcolors.FAIL)
            sys.exit(1)
        except json.JSONDecodeError:
            self._print_color(f"FATAL ERROR: Could not parse JSON in '{filepath}'", bcolors.FAIL)
            sys.exit(1)

    def _run_single_test_pair(self, test_case, pair_name):
        """
        Runs a single translation and returns the result.
        'pair_name' must be one of: 'hi_en', 'en_hi', 'mr_en', 'en_mr'
        """
        hypothesis = ""
        error = None

        # --- Set up source and reference ---
        if pair_name == 'hi_en':
            source_text = test_case['hindi']
            reference_text = test_case['english']
            lang_to_log = None
        elif pair_name == 'en_hi':
            source_text = test_case['english']
            reference_text = test_case['hindi']
            lang_to_log = "Hindi"
        elif pair_name == 'mr_en':
            source_text = test_case['marathi']
            reference_text = test_case['english']
            lang_to_log = None
        elif pair_name == 'en_mr':
            source_text = test_case['english']
            reference_text = test_case['marathi']
            lang_to_log = "Marathi"
        else:
            return None, None, None # Should not happen

        # --- Execute translation ---
        try:
            translator.translation_log = {} # Reset log
            if lang_to_log:
                # This is an English -> Other test
                # We must set the log manually as required by trans_for_output
                translator.translation_log["language"] = lang_to_log
                hypothesis = translator.trans_for_output(source_text)
            else:
                # This is a Hindi/Marathi -> English test
                hypothesis = translator.trans_for_rag(source_text)

        except Exception as e:
            error = str(e)
            hypothesis = f"ERROR: {e}"

        # --- Calculate metrics ---
        # Only calculate if there wasn't an error during translation
        metrics = {}
        if not error:
            metrics = calculate_nlp_metrics(reference_text, hypothesis)
        else:
             metrics = {"BLEU": 0.0, "METEOR": 0.0, "ROUGE-L (f)": 0.0}


        # --- Print individual result ---
        print(f"\n  {bcolors.BOLD}Source:{bcolors.ENDC}\n    {source_text}")
        print(f"\n  {bcolors.OKGREEN}Reference (Expected):{bcolors.ENDC}\n    {reference_text}")
        if error:
            self._print_color(f"\n  {bcolors.FAIL}Hypothesis (Actual):{bcolors.ENDC}\n    {hypothesis}", bcolors.FAIL)
        else:
            self._print_color(f"\n  {bcolors.OKCYAN}Hypothesis (Actual):{bcolors.ENDC}\n    {hypothesis}", bcolors.OKCYAN)

        return reference_text, hypothesis, metrics

    def run_all_tests(self):
        """
        Loops through all 65 test cases and runs all 4 translation pairs.
        """
        if not self.test_cases:
            self._print_color("No test cases loaded, skipping functional tests.", bcolors.WARNING)
            return

        total_calls = len(self.test_cases) * 4
        total_time_sec = total_calls * self.delay_between_calls
        print(f"\nRunning {len(self.test_cases)} test cases ({total_calls} total API calls)...")
        self._print_color(f"A delay of {self.delay_between_calls}s is added between calls.", bcolors.OKCYAN)
        print(f"Estimated test time: ~{total_time_sec / 60:.1f} minutes. This will take a moment.")

        for i, test_case in enumerate(self.test_cases):
            test_id = test_case['text_id']
            header = f"--- Testing Case {test_id} ({i+1}/{len(self.test_cases)}) ---"
            self._print_color(f"\n{header}", bcolors.HEADER)

            # --- 1. Hindi -> English ---
            self._print_color("\n[Pair 1/4: Hindi -> English]", bcolors.OKBLUE)
            ref, hyp, metrics = self._run_single_test_pair(test_case, 'hi_en')
            if ref: self.results['hi_en'].append(metrics)
            time.sleep(self.delay_between_calls) # <--- DELAY ADDED

            # --- 2. English -> Hindi ---
            self._print_color("\n[Pair 2/4: English -> Hindi]", bcolors.OKBLUE)
            ref, hyp, metrics = self._run_single_test_pair(test_case, 'en_hi')
            if ref: self.results['en_hi'].append(metrics)
            time.sleep(self.delay_between_calls) # <--- DELAY ADDED

            # --- 3. Marathi -> English ---
            self._print_color("\n[Pair 3/4: Marathi -> English]", bcolors.OKBLUE)
            ref, hyp, metrics = self._run_single_test_pair(test_case, 'mr_en')
            if ref: self.results['mr_en'].append(metrics)
            time.sleep(self.delay_between_calls) # <--- DELAY ADDED

            # --- 4. English -> Marathi ---
            self._print_color("\n[Pair 4/4: English -> Marathi]", bcolors.OKBLUE)
            ref, hyp, metrics = self._run_single_test_pair(test_case, 'en_mr')
            if ref: self.results['en_mr'].append(metrics)
            time.sleep(self.delay_between_calls) # <--- DELAY ADDED

        print("\n...Functional tests complete.")

    def _print_summary(self, pair_name: str, results_list: List[Dict[str, float]]):
        """Helper to calculate and print average metrics."""
        if not results_list:
            self._print_color(f"No results for {pair_name}", bcolors.WARNING)
            return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE-L (f)": 0.0} # Return zero dict

        # Filter out potential None or empty dicts if metric calculation failed
        valid_results = [r for r in results_list if r and isinstance(r, dict)]
        if not valid_results:
             self._print_color(f"Metric calculation failed for all samples in {pair_name}", bcolors.WARNING)
             return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE-L (f)": 0.0}

        avg_bleu = statistics.mean([r.get('BLEU', 0.0) for r in valid_results])
        avg_meteor = statistics.mean([r.get('METEOR', 0.0) for r in valid_results])
        avg_rouge = statistics.mean([r.get('ROUGE-L (f)', 0.0) for r in valid_results])

        self._print_color(f"\n  {pair_name} (Total Samples: {len(results_list)})", bcolors.HEADER)
        print("  " + "-"*30)
        self._print_color(f"    Avg BLEU:      {avg_bleu:.4f}", bcolors.OKGREEN)
        self._print_color(f"    Avg METEOR:    {avg_meteor:.4f}", bcolors.OKGREEN)
        self._print_color(f"    Avg ROUGE-L:   {avg_rouge:.4f}", bcolors.OKGREEN)

        return {"BLEU": avg_bleu, "METEOR": avg_meteor, "ROUGE-L (f)": avg_rouge} # Return calculated averages


    def print_final_summary(self):
        """Prints the final summary box for all NLP metrics."""
        print("\n" + "="*45)
        self._print_color("  NLP Evaluation Summary", bcolors.BOLD)
        print("="*45)

        summary_data = {}
        summary_data['hi_en'] = self._print_summary("Hindi -> English", self.results['hi_en'])
        summary_data['en_hi'] = self._print_summary("English -> Hindi", self.results['en_hi'])
        summary_data['mr_en'] = self._print_summary("Marathi -> English", self.results['mr_en'])
        summary_data['en_mr'] = self._print_summary("English -> Marathi", self.results['en_mr'])

        print("="*45 + "\n")
        return summary_data # Return the summary numbers


# ======================================================================
# MAIN EXECUTION
# ======================================================================

# --- Tee class to duplicate stdout ---
class Tee(object):
    """Duplicates print output to stdout and a StringIO buffer."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    """
    Main function to run the entire test suite and log output.
    """
    # --- Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"translation_test_log_{timestamp}.txt"
    original_stdout = sys.stdout
    log_buffer = io.StringIO()
    sys.stdout = Tee(original_stdout, log_buffer)

    print(f"Starting test suite run at: {timestamp}")
    print(f"Logging output to: {log_filename}\n")

    try:
        # Ensure terminal supports colors (for Windows)
        if os.name == 'nt':
            os.system('color')

        # --- Part 1: Run Edge Case Tests ---
        edge_tester = EdgeCaseTester()
        edge_cases_passed = edge_tester.run_tests()

        if not edge_cases_passed:
            print(f"{bcolors.WARNING}WARNING: Not all edge case tests passed.{bcolors.ENDC}")
            print("Continuing to functional tests, but results may be unreliable.\n")

        # --- Part 2: Run Functional & NLP Tests ---

        # 1. Setup NLTK
        setup_nltk()

        # 2. Instantiate and run tester
        func_tester = FunctionalTester(use_color=edge_tester.use_color)
        func_tester.run_all_tests()

        # 3. Print the final summary
        summary_metrics = func_tester.print_final_summary() # Store summary

        print("Test suite finished.")

    finally:
        # --- Restore stdout and Write Log File ---
        sys.stdout = original_stdout # Restore stdout
        print(f"\nWriting full test log to '{log_filename}'...")
        log_content = log_buffer.getvalue()
        log_buffer.close()

        try:
            # Strip color codes before writing to file
            plain_log_content = strip_color_codes(log_content)
            with open(log_filename, 'w', encoding='utf-8') as log_file:
                log_file.write(plain_log_content)
            print(f"Successfully wrote log file: {log_filename}")
        except Exception as e:
            print(f"{bcolors.FAIL}Error writing log file: {e}{bcolors.ENDC}")


if __name__ == "__main__":
    main()

