# !pip install -q langdetect requests

import json
import os
import requests
from pathlib import Path
from langdetect import detect
# Import validation helpers
import translation_validator as tv

# --- Custom Exception for Quality Issues ---
class TranslationQualityError(Exception):
    """Custom exception raised when translation quality validation fails."""
    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics # Optionally store metrics for logging

# --- Main Translator Class ---
class Translator:
    """
    Handles language detection and translation using Azure Cognitive Services,
    designed for integration into pipelines like RAG.

    Includes validation checks and raises TranslationQualityError on failures.
    Manages Azure credentials internally.
    """
    SUPPORTED_LANGUAGES = {"English", "Hindi", "Marathi"}

    def __init__(self):
        """
        Initializes the Translator by loading Azure credentials.
        Raises RuntimeError if credentials are not found.
        """
        self.subscription_key, self.region = self._load_azure_credentials()

    def _load_azure_credentials(self) -> tuple[str, str]:
        """
        Loads Azure credentials securely from environment variables or a local
        config file (~/.translator_config.json).
        Prioritizes environment variables.
        """
        key = os.environ.get("AZ_TRANSLATOR_KEY")
        region = os.environ.get("AZ_TRANSLATOR_REGION")

        if key:
            # Use default region if not specified in env var
            return key, region or "centralindia"

        # Fallback: Check for user-level config file in home directory
        cfg_path_str = os.environ.get("AZ_TRANSLATOR_CONFIG") or str(Path.home() / ".translator_config.json")
        cfg_file = Path(cfg_path_str)
        if cfg_file.exists():
            try:
                raw = json.loads(cfg_file.read_text(encoding="utf-8"))
                key = raw.get("subscription_key")
                # Use default region if not specified in config file
                region = raw.get("region", "centralindia")
                if key:
                    return key, region
            except Exception as e:
                # Log the error but proceed to raise the main RuntimeError
                print(f"Warning: Could not load config file '{cfg_path_str}': {e}")
                pass # Fall through to raise RuntimeError below

        # If neither env vars nor config file worked, raise error
        raise RuntimeError(
            "Azure Translator credentials not found. Set AZ_TRANSLATOR_KEY and optionally "
            "AZ_TRANSLATOR_REGION as environment variables, or create a config file at "
            "~/.translator_config.json with keys {'subscription_key': '...', 'region': '...'}. "
            "Ensure the file is valid JSON."
        )

    def _azure_translate(self, text: str, to_lang_code: str, from_lang_code: str | None = None) -> str:
        """
        Internal method to call the Microsoft Translator REST API.

        Args:
            text: The text to translate.
            to_lang_code: The target language code (e.g., 'en', 'hi').
            from_lang_code: The source language code (e.g., 'hi') or None for auto-detect (not recommended).

        Returns:
            The translated text.

        Raises:
            RuntimeError: If the API request fails or the response format is unexpected.
        """
        endpoint = f"https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to={to_lang_code}"
        if from_lang_code:
            endpoint += f"&from={from_lang_code}"

        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-Type": "application/json; charset=UTF-8", # Specify UTF-8
        }

        # Request body must be a list containing a dict with 'Text' key
        body = [{"Text": text}]

        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=20) # Increased timeout
            resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            # Catch network errors, timeout errors, etc.
            status = getattr(resp, 'status_code', 'N/A')
            body_text = getattr(resp, 'text', 'N/A')
            raise RuntimeError(f"Azure translation request failed: {e}; status={status}; body={body_text}") from e

        # Process the successful response
        try:
            data = resp.json()
            # Expected structure: [{ "translations": [ { "text": "...", "to": "en" } ] }]
            translated_text = data[0]["translations"][0]["text"]
            return translated_text
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unexpected Azure response format: {e}; status={resp.status_code}; raw_body={resp.text}") from e

    def detect_language(self, text: str) -> str | None:
        """
        Detects the language of the input text using langdetect.
        Includes a fallback for short strings.

        Args:
            text: The input text.

        Returns:
            The detected language name ('English', 'Hindi', 'Marathi') or None if unsupported/undetectable.

        Raises:
            ValueError: If langdetect fails on longer text.
        """
        try:
            lang_code = detect(text)
        except Exception as e:
            raise ValueError("Could not detect language using langdetect.")

        language_name = tv.code_to_name(lang_code)

        # Check if the detected language is one we support
        if language_name not in self.SUPPORTED_LANGUAGES:
             # Raise error for unsupported languages explicitly detected
             if tv.code_to_name(lang_code) is not None:
                  raise ValueError(f"Language '{lang_code}' ({language_name or 'Unknown'}) detected but is not supported.")
             else:
                  # This case handles codes not in our map at all
                  raise ValueError(f"Detected language code '{lang_code}' is not recognized or supported.")

        return language_name


    def trans_for_rag(self, user_query: str) -> tuple[str, str]:
        """
        Processes a user query for the RAG pipeline.
        1. Validates input.
        2. Detects language (Hindi, Marathi, English supported).
        3. Translates Hindi/Marathi to English.
        4. Performs round-trip quality validation.

        Args:
            user_query: The user's input query string.

        Returns:
            A tuple: (processed_text: str, detected_language_name: str)
            - processed_text: The query in English (or original if already English).
            - detected_language_name: The name of the detected language ('English', 'Hindi', 'Marathi').

        Raises:
            ValueError: For invalid input, undetectable, or unsupported languages.
            TranslationQualityError: If the translation's round-trip validation fails.
            RuntimeError: If the Azure API call fails.
        """
        # 1. Validate input using the helper function
        tv.check_input(user_query)

        # 2. Detect language
        detected_language_name = self.detect_language(user_query)
        # detect_language raises ValueError if not supported/detectable

        # 3. Handle English passthrough
        if detected_language_name == "English":
            return user_query, detected_language_name

        # 4. Translate Hindi/Marathi to English
        source_lang_code = tv.name_to_code(detected_language_name)
        try:
            english_translation = self._azure_translate(user_query, to_lang_code="en", from_lang_code=source_lang_code)
        except RuntimeError as e:
            # Add context to API errors
            raise RuntimeError(f"Failed to translate query from {detected_language_name} to English: {e}") from e


        # 5. Perform Round-Trip Quality Validation
        try:
            # Translate back to the original language
            roundtrip_back = self._azure_translate(english_translation, to_lang_code=source_lang_code, from_lang_code="en")

            # Calculate similarity metrics
            metrics = tv.validate_roundtrip(user_query, roundtrip_back)

            # Check if metrics meet the minimum acceptable threshold
            if not tv.is_acceptable(metrics):
                # --- QUALITY FAILURE ---
                # Raise exception instead of returning original text
                raise TranslationQualityError(
                    f"Translation quality from {detected_language_name} to English failed validation.",
                    metrics=metrics
                )
            # else: Validation passed

        except RuntimeError as e:
            # Handle API errors during the validation step
            print(f"Warning: Could not complete round-trip validation for RAG input due to API error: {e}")
            # Decide whether to proceed with potentially unvalidated translation or raise
            # For now, we proceed but log a warning (could be changed to raise TranslationQualityError)
            pass
        except TranslationQualityError:
             # Re-raise the quality error if it happened within validation steps
             raise
        except Exception as e:
            # Catch unexpected errors during validation
            print(f"Warning: Unexpected error during round-trip validation for RAG input: {e}")
            pass # Proceed with the translation

        # 6. Return the English translation and detected language
        return english_translation, detected_language_name

    def trans_for_output(self, english_rag_response: str, target_language_name: str) -> str:
        """
        Translates the English RAG response back to the user's original language.
        1. Validates input.
        2. Translates English to the target language (Hindi or Marathi).
        3. Performs quality validation (translating back to English).

        Args:
            english_rag_response: The RAG system's response in English.
            target_language_name: The desired output language ('Hindi' or 'Marathi').
                                 (Should be the language detected by trans_for_rag).

        Returns:
            The RAG response translated into the target language.

        Raises:
            ValueError: If input is invalid or target language is not supported/English.
            TranslationQualityError: If the translation's validation fails.
            RuntimeError: If the Azure API call fails.
        """
        # 1. Validate input
        tv.check_input(english_rag_response)
        if not isinstance(target_language_name, str) or target_language_name not in self.SUPPORTED_LANGUAGES:
             raise ValueError(f"Invalid target_language_name '{target_language_name}'. Must be one of {self.SUPPORTED_LANGUAGES}.")

        # Handle English output (no translation needed)
        if target_language_name == "English":
            return english_rag_response

        # 2. Translate English -> Target Language (Hindi/Marathi)
        target_lang_code = tv.name_to_code(target_language_name)
        try:
            final_translation = self._azure_translate(english_rag_response, to_lang_code=target_lang_code, from_lang_code="en")
        except RuntimeError as e:
            # Add context to API errors
            raise RuntimeError(f"Failed to translate RAG output from English to {target_language_name}: {e}") from e

        # 3. Perform Quality Validation (Translate back to English)
        try:
            # Translate the result back to English for comparison
            # Using None for from_lang_code relies on Azure's auto-detect, can be less reliable
            # but avoids issues if the 'final_translation' contains mixed scripts accidentally.
            # Alternatively, specify from_lang_code=target_lang_code
            back_to_english = self._azure_translate(final_translation, to_lang_code="en", from_lang_code=target_lang_code)

            # Calculate similarity metrics against the original English response
            metrics = tv.validate_roundtrip(english_rag_response, back_to_english)

            # Check if metrics meet the minimum acceptable threshold
            if not tv.is_acceptable(metrics):
                # --- QUALITY FAILURE ---
                # Raise exception instead of returning original user query
                raise TranslationQualityError(
                    f"Translation quality from English to {target_language_name} failed validation.",
                    metrics=metrics
                )
            # else: Validation passed

        except RuntimeError as e:
            # Handle API errors during the validation step
            print(f"Warning: Could not complete validation for RAG output due to API error: {e}")
            # Proceed with potentially unvalidated translation
            pass
        except TranslationQualityError:
             # Re-raise the quality error if it happened within validation steps
             raise
        except Exception as e:
            # Catch unexpected errors during validation
            print(f"Warning: Unexpected error during validation for RAG output: {e}")
            pass # Proceed with the translation

        # 4. Return the final translated response
        return final_translation
