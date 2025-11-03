# Translation Module for RAG Pipeline

## Overview

This Python module provides language detection and translation capabilities specifically designed for integration into a Retrieval-Augmented Generation (RAG) pipeline. It supports English, Hindi, and Marathi.

The module performs two primary functions:

- **Input Processing (`trans_for_rag`)**: Detects the language of an input text. If it's Hindi or Marathi, it translates the text to English. English text is passed through unchanged. It includes a basic round-trip validation to check translation consistency.
- **Output Processing (`trans_for_output`)**: Translates an English text (e.g., a RAG response) back into a specified target language (Hindi or Marathi). It includes a validation step by translating the result back to English and comparing.

Unsupported languages or inputs failing validation checks will raise exceptions.

## File Structure

```
translation/
│
├── __init__.py           # Makes 'translation' a Python package
├── translator.py         # Core Translator class and Azure API logic
├── translation_validator.py # Validation helpers, metric calculations, config
├── TranslationREADME.md  # This file
│
└── tests/                # Testing scripts, data, and logs
    ├── __init__.py       # Makes 'tests' a sub-package
    ├── translation_module_test.py # RAG simulation & error handling tests
    ├── test_translation.py         # Full NLP evaluation script
    │
    ├── data/             # Test data files
    │   ├── edge_case_test_cases.json
    │   └── translation_test_cases.json
    │
    └── logs/             # Test output logs (timestamped)
        └── ...
```

## Dependencies

### Core Module Requirements

The core module requires the following Python packages:

- **requests**: For making API calls to Azure Translator.
- **langdetect**: For initial language detection.

Install them using pip:

```bash
pip install requests langdetect
```

### Testing Requirements

For running the test scripts (`test_translation.py`, `translation_module_test.py`), you also need:

- **nltk**: For NLP evaluation metrics (BLEU, METEOR).
- **rouge-score**: For the ROUGE evaluation metric.

Install them using pip:

```bash
pip install nltk rouge-score
```

**Note**: The first time you run the test scripts involving NLP metrics, nltk may attempt to download required data packages (punkt, wordnet, omw-1.4). Ensure you have an internet connection.

## Setup: Azure Translator Credentials

This module uses the Azure Cognitive Services Translator API. You need a valid **Subscription Key** and the corresponding **Region** for your Azure resource.

The module loads these credentials using one of the following methods (in order of priority):

### 1. Environment Variables (Recommended)

Set the following environment variables in your system or terminal session:

- `AZ_TRANSLATOR_KEY`: Your Azure Translator subscription key.
- `AZ_TRANSLATOR_REGION`: Your Azure resource region (e.g., `centralindia`, `eastus`). If omitted, it defaults to `centralindia`.

#### Example (Linux/macOS)

```bash
export AZ_TRANSLATOR_KEY="YOUR_ACTUAL_KEY_HERE"
export AZ_TRANSLATOR_REGION="centralindia"
python your_main_script.py
```

#### Example (Windows CMD)

```cmd
set AZ_TRANSLATOR_KEY=YOUR_ACTUAL_KEY_HERE
set AZ_TRANSLATOR_REGION=centralindia
python your_main_script.py
```

#### Example (Windows PowerShell)

```powershell
$env:AZ_TRANSLATOR_KEY = "YOUR_ACTUAL_KEY_HERE"
$env:AZ_TRANSLATOR_REGION = "centralindia"
python your_main_script.py
```

### 2. Local Configuration File (Optional Fallback)

Create a JSON file named `.translator_config.json` in your user home directory.

- **Windows**: `C:\Users\YourUsername\.translator_config.json`
- **macOS/Linux**: `/Users/YourUsername/.translator_config.json` (or `~/.translator_config.json`)

The file should contain your key and region:

```json
{
  "subscription_key": "YOUR_ACTUAL_KEY_HERE",
  "region": "YOUR_REGION_HERE"
}
```

**Note**: The `region` key is optional and defaults to `centralindia` if omitted.

If credentials are not found using either method, the `Translator` class will raise a `RuntimeError` during initialization.

## Usage Example (within a RAG Pipeline)

```python
# Import the Translator class and custom exception
try:
    # Adjust import path based on your project structure
    from translation.translator import Translator, TranslationQualityError
    import translation.translation_validator as tv # Optional, for constants
except ImportError:
    print("Error: Could not import the translation module.")
    # Handle module not found
    sys.exit(1)

# --- 1. Initialize Translator (once at startup) ---
translator_service = None
try:
    translator_service = Translator()
    print("Translator service initialized.")
except RuntimeError as e:
    print(f"FATAL: Failed to initialize translator: {e}")
    # Handle failure (e.g., disable translation features)

# --- Function to process a user query ---
def process_query_for_rag(user_query: str):
    if not translator_service:
        # Basic check if translator failed init but query might be English
        try:
            from langdetect import detect
            if tv.code_to_name(detect(user_query)) == 'English':
                 print("Translator unavailable, proceeding with query as English.")
                 return user_query, 'English'
            else:
                 return None, "Sorry, the translation service is currently unavailable."
        except Exception:
             return None, "Sorry, the translation service is currently unavailable."


    try:
        # --- 2. Translate Input for RAG Core ---
        english_query, original_language = translator_service.trans_for_rag(user_query)
        print(f"Original language: {original_language}, Query for RAG: {english_query}")
        return english_query, original_language

    except (ValueError, TranslationQualityError, RuntimeError) as e:
        print(f"Error processing user query: {e}")
        # Return error message or handle appropriately
        return None, "Sorry, I couldn't process your query due to a translation issue."
    except Exception as e:
         print(f"Unexpected error during query processing: {e}")
         return None, "An unexpected error occurred."

# --- Function to translate RAG response back ---
def translate_rag_response(rag_response_en: str, original_language: str):
    if not translator_service or original_language == 'English':
        return rag_response_en # Return English if no service or already English

    try:
        # --- 3. Translate Output for User ---
        final_response = translator_service.trans_for_output(rag_response_en, original_language)
        return final_response

    except (ValueError, TranslationQualityError, RuntimeError) as e:
        print(f"Error translating RAG response back to {original_language}: {e}")
        # Return English response with a disclaimer
        return f"(English) {rag_response_en}\n\n[Disclaimer: Could not reliably translate this response to {original_language}.]"
    except Exception as e:
        print(f"Unexpected error during response translation: {e}")
        return f"(English) {rag_response_en}\n\n[Disclaimer: An unexpected error occurred during translation.]"

# --- Example Pipeline Flow ---
if __name__ == "__main__":
    test_query_hindi = "मनी लॉन्ड्रिंग क्या है?"
    test_query_english = "What is KYC?"

    # --- Process Hindi Query ---
    query_for_rag_1, lang_1 = process_query_for_rag(test_query_hindi)
    if query_for_rag_1:
        # Simulate RAG response
        simulated_response_en_1 = f"RAG response about Money Laundering based on '{query_for_rag_1}'"
        final_answer_1 = translate_rag_response(simulated_response_en_1, lang_1)
        print(f"\nFinal Answer (Hindi): {final_answer_1}")
    else:
        print(f"\nError processing Hindi query: {lang_1}") # lang_1 contains error message here

    # --- Process English Query ---
    query_for_rag_2, lang_2 = process_query_for_rag(test_query_english)
    if query_for_rag_2:
        # Simulate RAG response
        simulated_response_en_2 = f"RAG response about KYC based on '{query_for_rag_2}'"
        final_answer_2 = translate_rag_response(simulated_response_en_2, lang_2)
        print(f"\nFinal Answer (English): {final_answer_2}")
    else:
        print(f"\nError processing English query: {lang_2}") # lang_2 contains error message here
```

## Error Handling

The module uses standard Python exceptions (`ValueError`, `RuntimeError`) for issues like invalid input or API failures.

Additionally, it defines a custom exception:

- **`TranslationQualityError`**: Raised by `trans_for_rag` or `trans_for_output` if the internal round-trip validation score falls below the threshold defined by `MIN_COMBINED_QUALITY_SCORE` in `translation_validator.py`. Your calling code (e.g., the RAG pipeline) should catch this exception to handle potentially low-quality translations gracefully (e.g., by informing the user or falling back to English).

## Testing

The `tests/` subfolder contains scripts for evaluating the module:

- **`translation_module_test.py`**: Simulates the RAG pipeline flow using various test cases (including edge cases) and reports success/failure for each step. Useful for verifying integration logic.
- **`test_translation.py`**: Performs a comprehensive evaluation using a larger dataset (`translation_test_cases.json`) and calculates standard NLP metrics (BLEU, METEOR, ROUGE-L) for different language pairs. Useful for assessing overall translation quality.

Both scripts log their detailed output to timestamped `.txt` files in the `tests/logs/` directory.

### Running Tests

To run the tests, navigate to the directory containing the `translation` folder (i.e., the project root) in your terminal and execute the scripts:

```bash
# Example: Running the RAG simulation test
python -m translation.tests.translation_module_test

# Example: Running the full NLP evaluation test
python -m translation.tests.test_translation
```
