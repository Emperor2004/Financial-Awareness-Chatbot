"""
Model Loader for Sarvam AI Translation
Handles loading and caching of translation models
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PreTrainedModel
from typing import Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for models
_model_cache = {}
_tokenizer_cache = {}

# Model configuration
MODEL_NAME = "ai4bharat/indictrans-v2-all-gpu"  # Alternate model that supports Indian languages

# Supported language codes
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Odia',
    'pa': 'Punjabi'
}

MODEL_NAME = "sarvamai/sarvam-translate"

def get_device() -> str:
    """Determine the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_translation_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None
) -> Tuple[Union[AutoModelForSeq2SeqLM, AutoModelForCausalLM], AutoTokenizer]:
    """
    Load or retrieve cached translation model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on (cuda/mps/cpu)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _model_cache, _tokenizer_cache
    
    # Return cached model if available
    if model_name in _model_cache:
        logger.info(f"Using cached model: {model_name}")
        return _model_cache[model_name], _tokenizer_cache[model_name]
    
    try:
        logger.info(f"Loading translation model: {model_name}")
        
        # Determine device
        if device is None:
            device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Try different model architectures
        model = None
        for model_class in [AutoModelForSeq2SeqLM, AutoModelForCausalLM]:
            try:
                model = model_class.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                break
            except Exception as e:
                logger.debug(f"Failed to load with {model_class.__name__}: {str(e)}")
                continue
                
        if model is None:
            raise ValueError(f"Could not load model {model_name} with any available model class")
        
        # Move to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Cache the model and tokenizer
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading translation model: {str(e)}")
        raise

def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache, _tokenizer_cache
    _model_cache.clear()
    _tokenizer_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")

def is_language_supported(lang_code: str) -> bool:
    """Check if a language code is supported"""
    return lang_code.lower() in SUPPORTED_LANGUAGES

def get_supported_languages() -> dict:
    """Get dictionary of supported language codes and names"""
    return SUPPORTED_LANGUAGES.copy()