"""
Translation Service for FIU-Sahayak Chatbot
Main translation wrapper with language detection and caching
"""

import re
import logging
from typing import Optional, Dict, Tuple
from functools import lru_cache
import torch
from .model_loader import (
    get_translation_model, 
    is_language_supported,
    SUPPORTED_LANGUAGES,
    get_device
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translation service wrapper for multi-language support
    Handles language detection, translation, and caching
    """
    
    def __init__(self, model_name: str = "sarvamai/sarvam-translate"):
        """
        Initialize translation service
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = get_device()
        self._load_models()
        
        # Cache for translations
        self._translation_cache = {}
        
    def _load_models(self):
        """Load translation models"""
        try:
            self.model, self.tokenizer = get_translation_model(self.model_name)
            logger.info("Translation service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text to detect language
            
        Returns:
            Language code (en, hi, mr, etc.) or 'en' as default
        """
        # Remove extra whitespace
        text = text.strip()
        
        if not text:
            return 'en'
        
        # Simple heuristic-based language detection
        # Check for Devanagari script (Hindi, Marathi)
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        if devanagari_pattern.search(text):
            # Rough differentiation between Hindi and Marathi
            # Marathi has more usage of specific characters
            marathi_chars = re.compile(r'[ळ]')
            if marathi_chars.search(text):
                return 'mr'
            return 'hi'
        
        # Check for Tamil script
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        if tamil_pattern.search(text):
            return 'ta'
        
        # Check for Telugu script
        telugu_pattern = re.compile(r'[\u0C00-\u0C7F]')
        if telugu_pattern.search(text):
            return 'te'
        
        # Check for Bengali script
        bengali_pattern = re.compile(r'[\u0980-\u09FF]')
        if bengali_pattern.search(text):
            return 'bn'
        
        # Check for Gujarati script
        gujarati_pattern = re.compile(r'[\u0A80-\u0AFF]')
        if gujarati_pattern.search(text):
            return 'gu'
        
        # Check for Kannada script
        kannada_pattern = re.compile(r'[\u0C80-\u0CFF]')
        if kannada_pattern.search(text):
            return 'kn'
        
        # Check for Malayalam script
        malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')
        if malayalam_pattern.search(text):
            return 'ml'
        
        # Check for Odia script
        odia_pattern = re.compile(r'[\u0B00-\u0B7F]')
        if odia_pattern.search(text):
            return 'or'
        
        # Check for Gurmukhi script (Punjabi)
        gurmukhi_pattern = re.compile(r'[\u0A00-\u0A7F]')
        if gurmukhi_pattern.search(text):
            return 'pa'
        
        # Default to English
        return 'en'
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512
    ) -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum length of generated translation
            
        Returns:
            Translated text
        """
        # Validate languages
        if not is_language_supported(source_lang):
            logger.warning(f"Unsupported source language: {source_lang}")
            return text
        
        if not is_language_supported(target_lang):
            logger.warning(f"Unsupported target language: {target_lang}")
            return text
        
        # No translation needed if same language
        if source_lang == target_lang:
            return text
        
        # Check cache
        cache_key = f"{source_lang}_{target_lang}_{text[:100]}"
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        
        try:
            # Prepare input with language tags
            # Format: "Translate from {source} to {target}: {text}"
            input_text = f"Translate from {source_lang} to {target_lang}: {text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )
            
            # Decode output
            translated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Cache the translation
            self._translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text on error
    
    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """
        Translate text to English (for system processing)
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detected if None)
            
        Returns:
            English translation
        """
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        if source_lang == 'en':
            return text
        
        return self.translate(text, source_lang, 'en')
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language (for user response)
        
        Args:
            text: English text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if target_lang == 'en':
            return text
        
        return self.translate(text, 'en', target_lang)
    
    def process_conversation(
        self,
        user_input: str,
        system_response: str,
        user_lang: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Process a conversation round with translation
        
        Args:
            user_input: User's input text
            system_response: System's response in English
            user_lang: User's language code (auto-detected if None)
            
        Returns:
            Tuple of (detected_lang, english_input, translated_response)
        """
        # Detect user language
        if user_lang is None:
            user_lang = self.detect_language(user_input)
        
        logger.info(f"Detected language: {user_lang} ({SUPPORTED_LANGUAGES.get(user_lang, 'Unknown')})")
        
        # Translate user input to English for system processing
        english_input = self.translate_to_english(user_input, user_lang)
        
        # Translate system response to user's language
        translated_response = self.translate_from_english(system_response, user_lang)
        
        return user_lang, english_input, translated_response
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return SUPPORTED_LANGUAGES.copy()
    
    def clear_cache(self):
        """Clear translation cache"""
        self._translation_cache.clear()
        logger.info("Translation cache cleared")


# Singleton instance
_translation_service_instance = None

def get_translation_service() -> TranslationService:
    """Get or create singleton translation service instance"""
    global _translation_service_instance
    
    if _translation_service_instance is None:
        _translation_service_instance = TranslationService()
    
    return _translation_service_instance