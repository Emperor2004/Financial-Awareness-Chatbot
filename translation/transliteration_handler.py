"""
transliteration_handler.py

Lightweight transliteration handler for Roman-script Hindi and Marathi input.
Integrates seamlessly with the Translator module.
"""

from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI
import re

# Common Roman-script Hindi/Marathi clue words
COMMON_HINDI_PATTERNS = [
    r"\b(kya|kaise|hai|nahi|mein|tum|mera|tera|kyon|sab|acha|bura|paisa|ghar|ladka|ladki|samay|kam)\b",
]
COMMON_MARATHI_PATTERNS = [
    r"\b(mhanje|kay|tumhi|ahe|nahi|karaycha|pahije|kasa|kon|mala|tula|acha|thoda)\b",
]

def looks_like_roman_hindi(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in COMMON_HINDI_PATTERNS)

def looks_like_roman_marathi(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in COMMON_MARATHI_PATTERNS)

def maybe_transliterate_to_devanagari(text: str) -> tuple[str, str | None]:
    """
    Detects if Roman-script text is Hindi or Marathi and transliterates accordingly.

    Returns:
        (processed_text, detected_language_hint)
    """
    if looks_like_roman_hindi(text):
        try:
            devanagari = transliterate(text, ITRANS, DEVANAGARI)
            return devanagari, "Hindi"
        except Exception:
            return text, None

    if looks_like_roman_marathi(text):
        try:
            devanagari = transliterate(text, ITRANS, DEVANAGARI)
            return devanagari, "Marathi"
        except Exception:
            return text, None

    # If no Roman Hindi/Marathi patterns found, return unchanged
    return text, None
