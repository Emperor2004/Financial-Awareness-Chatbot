# Translation Module Documentation

## Overview

The Translation Module provides comprehensive multi-language support for the FIU-Sahayak chatbot using Sarvam AI's translation model. It acts as an intelligent wrapper that:

1. **Detects** the user's input language automatically
2. **Translates** user queries from regional languages to English for RAG processing
3. **Translates** system responses back to the user's preferred language
4. **Caches** translations for improved performance

## Architecture

```
User Input (Hindi/Tamil/etc.)
        â†"
Language Detection
        â†"
Translation to English
        â†"
RAG Pipeline Processing
        â†"
English Response
        â†"
Translation to User Language
        â†"
User Response (Hindi/Tamil/etc.)
```

## Supported Languages

| Code | Language | Script |
|------|----------|--------|
| `en` | English | Latin |
| `hi` | Hindi | Devanagari |
| `mr` | Marathi | Devanagari |
| `ta` | Tamil | Tamil |
| `te` | Telugu | Telugu |
| `bn` | Bengali | Bengali |
| `gu` | Gujarati | Gujarati |
| `kn` | Kannada | Kannada |
| `ml` | Malayalam | Malayalam |
| `or` | Odia | Odia |
| `pa` | Punjabi | Gurmukhi |

## Installation

### 1. Install Dependencies

```bash
cd Financial-Awareness-Chatbot
pip install transformers torch sentencepiece accelerate
```

### 2. Download Translation Model

The Sarvam AI model will be automatically downloaded on first use. Ensure you have:
- **Disk Space**: ~2-3 GB for model files
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended (CUDA-enabled)

### 3. Configure Environment

Create or update `backend/.env`:

```env
TRANSLATION_MODEL=sarvamai/sarvam-translate
TRANSLATION_DEVICE=cuda  # or cpu/mps
ENABLE_TRANSLATION_CACHE=True
```

## Usage

### Basic Usage

```python
from translation import TranslationService

# Initialize service
translator = TranslationService()

# Detect language
lang = translator.detect_language("मुझे FIU-IND के बारे में बताएं")
print(f"Detected: {lang}")  # Output: hi

# Translate to English
english = translator.translate_to_english("मुझे FIU-IND के बारे में बताएं")
print(english)  # Output: Tell me about FIU-IND

# Translate back to Hindi
hindi = translator.translate_from_english("FIU-IND is...", "hi")
print(hindi)  # Output: FIU-IND...
```

### Conversation Processing

```python
# Process full conversation round
user_input = "FIU-IND क्या है?"
system_response_en = "FIU-IND is the Financial Intelligence Unit..."

lang, english_input, translated_response = translator.process_conversation(
    user_input=user_input,
    system_response=system_response_en
)

print(f"Language: {lang}")
print(f"English: {english_input}")
print(f"Response: {translated_response}")
```

### Singleton Pattern

```python
from translation import get_translation_service

# Get or create singleton instance
translator = get_translation_service()
```

## API Endpoints

### 1. Detect Language

**Endpoint**: `POST /api/translate/detect`

**Request**:
```json
{
  "text": "मुझे PMLA के बारे में बताएं"
}
```

**Response**:
```json
{
  "language": "hi",
  "language_name": "Hindi",
  "confidence": "high"
}
```

### 2. Get Supported Languages

**Endpoint**: `GET /api/translate/supported`

**Response**:
```json
{
  "languages": {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    ...
  }
}
```

### 3. Chat with Translation

**Endpoint**: `POST /api/chat`

**Request**:
```json
{
  "message": "FIU-IND क्या है?",
  "language": "hi"  // Optional
}
```

**Response**:
```json
{
  "response": "FIU-IND भारतीय वित्तीय खुफिया इकाई है...",
  "detected_language": "hi",
  "language_name": "Hindi",
  "english_query": "What is FIU-IND?",
  "english_response": "FIU-IND is...",
  "sources": [...],
  "model": "gemma2:9b"
}
```

### 4. Clear Translation Cache

**Endpoint**: `POST /api/translate/clear-cache`

**Response**:
```json
{
  "message": "Translation cache cleared"
}
```

## Language Detection

The module uses **Unicode-based script detection** for automatic language identification:

### Detection Logic

1. **Script Analysis**: Identifies Unicode ranges for each script
2. **Character Patterns**: Looks for script-specific characters
3. **Fallback**: Defaults to English if no script detected

### Example Detection

```python
from translation import TranslationService

translator = TranslationService()

# Hindi (Devanagari)
translator.detect_language("नमस्ते")  # Returns: 'hi'

# Tamil
translator.detect_language("வணக்கம்")  # Returns: 'ta'

# English
translator.detect_language("Hello")  # Returns: 'en'
```

## Transliteration

The module includes basic transliteration support for romanization:

```python
from translation import transliterate_text

# Devanagari to Latin
latin = transliterate_text("नमस्ते")
print(latin)  # Output: namaste

# Tamil to Latin
latin = transliterate_text("வணக்கம்")
print(latin)  # Output: vanakkam
```

## Performance Optimization

### 1. Translation Caching

The service automatically caches translations to reduce latency:

```python
# First call: ~2-3 seconds (model inference)
result1 = translator.translate_to_english("नमस्ते")

# Second call: ~0.01 seconds (cache hit)
result2 = translator.translate_to_english("नमस्ते")
```

### 2. Model Caching

Models are loaded once and cached globally:

```python
from translation.model_loader import clear_model_cache

# Clear to free memory
clear_model_cache()
```

### 3. GPU Acceleration

Automatically uses CUDA if available:

```python
# Check device
from translation.model_loader import get_device

device = get_device()
print(device)  # cuda, mps, or cpu
```

## Integration with RAG Pipeline

### Updated Chat Flow

```python
from translation import get_translation_service
from rag_pipeline import get_rag_response

translator = get_translation_service()

def handle_chat(user_message, user_lang=None):
    # Detect and translate to English
    lang, english_query, _ = translator.process_conversation(
        user_input=user_message,
        system_response="",
        user_lang=user_lang
    )
    
    # Process with RAG
    rag_result = get_rag_response(english_query)
    english_response = rag_result['response']
    
    # Translate back to user language
    translated_response = translator.translate_from_english(
        english_response,
        lang
    )
    
    return {
        'response': translated_response,
        'detected_language': lang,
        'sources': rag_result['sources']
    }
```

## Frontend Integration

### Language Selector Component

```typescript
// components/LanguageSelector.tsx
import { useState } from 'react';

export function LanguageSelector() {
  const [language, setLanguage] = useState('en');
  
  const languages = {
    en: 'English',
    hi: 'हिंदी',
    mr: 'मराठी',
    ta: 'தமிழ்',
    te: 'తెలుగు'
  };
  
  return (
    <select 
      value={language} 
      onChange={(e) => setLanguage(e.target.value)}
    >
      {Object.entries(languages).map(([code, name]) => (
        <option key={code} value={code}>{name}</option>
      ))}
    </select>
  );
}
```

### Auto-Detection in Chat

```typescript
// lib/chat.ts
export async function sendMessage(message: string, language?: string) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      message, 
      language  // Optional: auto-detected if not provided
    })
  });
  
  return response.json();
}
```

## Error Handling

### Graceful Degradation

```python
try:
    translated = translator.translate_to_english(text)
except Exception as e:
    logger.error(f"Translation failed: {e}")
    translated = text  # Fallback to original
```

### Service Unavailability

```python
if translation_service is None:
    # Process in English only
    return process_english_query(user_message)
```

## Testing

### Unit Tests

```python
# tests/test_translation.py
import pytest
from translation import TranslationService

def test_language_detection():
    translator = TranslationService()
    
    assert translator.detect_language("Hello") == "en"
    assert translator.detect_language("नमस्ते") == "hi"
    assert translator.detect_language("வணக்கம்") == "ta"

def test_translation():
    translator = TranslationService()
    
    english = translator.translate_to_english("नमस्ते", "hi")
    assert len(english) > 0
    assert english.lower() != "नमस्ते"
```

### Integration Tests

```bash
# Test language detection endpoint
curl -X POST http://localhost:5000/api/translate/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "मुझे FIU-IND के बारे में बताएं"}'

# Test chat with translation
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "FIU-IND क्या है?", "language": "hi"}'
```

## Troubleshooting

### Model Download Issues

```bash
# Manually download model
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('sarvamai/sarvam-translate'); AutoModelForSeq2SeqLM.from_pretrained('sarvamai/sarvam-translate')"
```

### Memory Issues

```python
# Use CPU if GPU memory insufficient
os.environ['TRANSLATION_DEVICE'] = 'cpu'
```

### Slow Translation

```bash
# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Best Practices

1. **Cache Management**: Clear cache periodically to prevent memory bloat
2. **Language Detection**: Provide explicit language codes for better accuracy
3. **Error Handling**: Always have fallback to English
4. **Batch Processing**: For multiple translations, use batch mode
5. **Monitoring**: Log language detection and translation times

## Limitations

- **Translation Quality**: Dependent on Sarvam AI model capabilities
- **Context Loss**: Some domain-specific terms may not translate accurately
- **Latency**: First translation takes 2-3 seconds (model loading)
- **Memory**: Requires ~4GB RAM minimum for model

## Future Enhancements

- [ ] Custom fine-tuning for financial domain
- [ ] Batch translation support
- [ ] Language-specific prompts for RAG
- [ ] Translation quality metrics
- [ ] User feedback on translations
- [ ] Offline mode with smaller models
- [ ] Real-time streaming translation

## Support

For issues or questions:
- Check logs: `translation_service.log`
- Review model status: `GET /health`
- Clear cache: `POST /api/translate/clear-cache`

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Maintainer**: FIU-Sahayak Team