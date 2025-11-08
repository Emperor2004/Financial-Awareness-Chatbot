"""
Transliteration utilities for Indic languages
Handles romanization and script conversion
"""

import re
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Devanagari to Latin transliteration map (simplified)
DEVANAGARI_TO_LATIN = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'ळ': 'la', 'क्ष': 'ksha',
    'ज्ञ': 'gya',
    
    # Vowel signs
    'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'm', 'ः': 'h', 'ँ': 'n'
}

# Tamil to Latin transliteration map (simplified)
TAMIL_TO_LATIN = {
    # Vowels
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii', 'உ': 'u', 'ஊ': 'uu',
    'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
    
    # Consonants
    'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஞ': 'nya', 'ட': 'ta',
    'ண': 'na', 'த': 'tha', 'ந': 'na', 'ப': 'pa', 'ம': 'ma',
    'ய': 'ya', 'ர': 'ra', 'ல': 'la', 'வ': 'va', 'ழ': 'zha',
    'ள': 'la', 'ற': 'ra', 'ன': 'na',
    
    # Vowel signs
    'ா': 'aa', 'ி': 'i', 'ீ': 'ii', 'ு': 'u', 'ூ': 'uu',
    'ெ': 'e', 'ே': 'ee', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oo', 'ௌ': 'au',
    '்': ''
}

def transliterate_devanagari_to_latin(text: str) -> str:
    """
    Transliterate Devanagari script to Latin alphabet
    
    Args:
        text: Text in Devanagari script
        
    Returns:
        Transliterated text in Latin script
    """
    result = []
    for char in text:
        if char in DEVANAGARI_TO_LATIN:
            result.append(DEVANAGARI_TO_LATIN[char])
        else:
            result.append(char)
    
    return ''.join(result)

def transliterate_tamil_to_latin(text: str) -> str:
    """
    Transliterate Tamil script to Latin alphabet
    
    Args:
        text: Text in Tamil script
        
    Returns:
        Transliterated text in Latin script
    """
    result = []
    for char in text:
        if char in TAMIL_TO_LATIN:
            result.append(TAMIL_TO_LATIN[char])
        else:
            result.append(char)
    
    return ''.join(result)

def detect_script(text: str) -> Optional[str]:
    """
    Detect the script of the input text
    
    Args:
        text: Input text
        
    Returns:
        Script name ('devanagari', 'tamil', etc.) or None
    """
    # Devanagari (Hindi, Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        return 'devanagari'
    
    # Tamil
    if re.search(r'[\u0B80-\u0BFF]', text):
        return 'tamil'
    
    # Telugu
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'telugu'
    
    # Bengali
    if re.search(r'[\u0980-\u09FF]', text):
        return 'bengali'
    
    # Gujarati
    if re.search(r'[\u0A80-\u0AFF]', text):
        return 'gujarati'
    
    # Kannada
    if re.search(r'[\u0C80-\u0CFF]', text):
        return 'kannada'
    
    # Malayalam
    if re.search(r'[\u0D00-\u0D7F]', text):
        return 'malayalam'
    
    # Gurmukhi (Punjabi)
    if re.search(r'[\u0A00-\u0A7F]', text):
        return 'gurmukhi'
    
    return None

def transliterate_text(text: str, source_script: Optional[str] = None) -> str:
    """
    Transliterate text from Indic script to Latin alphabet
    
    Args:
        text: Text to transliterate
        source_script: Source script name (auto-detected if None)
        
    Returns:
        Transliterated text
    """
    if source_script is None:
        source_script = detect_script(text)
    
    if source_script is None:
        return text  # No transliteration needed
    
    try:
        if source_script == 'devanagari':
            return transliterate_devanagari_to_latin(text)
        elif source_script == 'tamil':
            return transliterate_tamil_to_latin(text)
        else:
            logger.warning(f"Transliteration not supported for script: {source_script}")
            return text
    except Exception as e:
        logger.error(f"Transliteration error: {str(e)}")
        return text

def romanize_for_search(text: str) -> str:
    """
    Romanize text for search indexing purposes
    
    Args:
        text: Text to romanize
        
    Returns:
        Romanized text
    """
    script = detect_script(text)
    if script:
        return transliterate_text(text, script)
    return text

def normalize_indic_text(text: str) -> str:
    """
    Normalize Indic text by removing diacritics and extra marks
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Remove zero-width characters
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text