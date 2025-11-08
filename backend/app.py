"""
Updated Flask Backend with Translation Support
Add these imports and routes to your existing backend/app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os

# Add translation module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from translation.translation import TranslationService, get_translation_service
from translation.model_loader import get_supported_languages

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize translation service (singleton)
translation_service = None

def init_translation_service():
    """Initialize translation service on startup"""
    global translation_service
    try:
        translation_service = get_translation_service()
        logging.info("Translation service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize translation service: {str(e)}")
        translation_service = None

# Call during app startup
@app.before_first_request
def startup():
    """Initialize services before first request"""
    init_translation_service()

# ============================================
# Translation Endpoints
# ============================================

@app.route('/api/translate/detect', methods=['POST'])
def detect_language():
    """
    Detect language of input text
    
    Request body:
    {
        "text": "मैं FIU-IND के बारे में जानना चाहता हूं"
    }
    
    Response:
    {
        "language": "hi",
        "language_name": "Hindi",
        "confidence": "high"
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if translation_service is None:
            return jsonify({'error': 'Translation service not available'}), 503
        
        lang_code = translation_service.detect_language(text)
        lang_name = get_supported_languages().get(lang_code, 'Unknown')
        
        return jsonify({
            'language': lang_code,
            'language_name': lang_name,
            'confidence': 'high' if lang_code != 'en' else 'medium'
        })
        
    except Exception as e:
        logging.error(f"Language detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate/supported', methods=['GET'])
def get_supported_langs():
    """
    Get list of supported languages
    
    Response:
    {
        "languages": {
            "en": "English",
            "hi": "Hindi",
            ...
        }
    }
    """
    try:
        return jsonify({
            'languages': get_supported_languages()
        })
    except Exception as e:
        logging.error(f"Error fetching supported languages: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint with automatic translation
    
    Request body:
    {
        "message": "मुझे PMLA के बारे में बताएं",
        "language": "hi"  // Optional, auto-detected if not provided
    }
    
    Response:
    {
        "response": "PMLA के बारे में...",  // Translated to user's language
        "detected_language": "hi",
        "english_query": "Tell me about PMLA",  // For debugging
        "sources": [...],
        "model": "gemma2:9b"
    }
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        user_language = data.get('language', None)
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check if translation service is available
        if translation_service is None:
            logging.warning("Translation service not available, processing in English only")
            detected_lang = 'en'
            english_query = user_message
        else:
            # Process with translation
            detected_lang, english_query, _ = translation_service.process_conversation(
                user_input=user_message,
                system_response="",  # Will be filled after RAG
                user_lang=user_language
            )
        
        # Get RAG response (assuming you have a get_rag_response function)
        # from rag_pipeline import get_rag_response
        # rag_result = get_rag_response(english_query)
        
        # For demonstration, using a placeholder
        english_response = f"This is a response about {english_query}..."
        sources = []
        model_used = "gemma2:9b"
        
        # Translate response back to user's language
        if translation_service and detected_lang != 'en':
            translated_response = translation_service.translate_from_english(
                english_response,
                detected_lang
            )
        else:
            translated_response = english_response
        
        return jsonify({
            'response': translated_response,
            'detected_language': detected_lang,
            'language_name': get_supported_languages().get(detected_lang, 'Unknown'),
            'english_query': english_query,
            'english_response': english_response,
            'sources': sources,
            'model': model_used
        })
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate/clear-cache', methods=['POST'])
def clear_translation_cache():
    """Clear translation cache"""
    try:
        if translation_service:
            translation_service.clear_cache()
            return jsonify({'message': 'Translation cache cleared'})
        return jsonify({'error': 'Translation service not available'}), 503
    except Exception as e:
        logging.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================
# Health Check with Translation Status
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check with translation service status"""
    translation_status = 'healthy' if translation_service else 'unavailable'
    
    return jsonify({
        'status': 'healthy',
        'translation_service': translation_status,
        'supported_languages': len(get_supported_languages())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)