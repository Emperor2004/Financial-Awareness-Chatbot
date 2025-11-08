# Translation Module Integration Guide

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd Financial-Awareness-Chatbot
pip install -r translation/requirements_translation.txt
```

### Step 2: Test Installation

```bash
cd translation
python test_translation.py
```

### Step 3: Update Backend

Add these lines to `backend/app.py`:

```python
# At the top with other imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from translation import get_translation_service

# Initialize translation service
translation_service = None

@app.before_first_request
def startup():
    global translation_service
    translation_service = get_translation_service()
```

### Step 4: Update Chat Endpoint

Replace your existing chat endpoint with:

```python
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_language = data.get('language', None)
    
    # Process with translation
    if translation_service:
        lang, english_query, _ = translation_service.process_conversation(
            user_input=user_message,
            system_response="",
            user_lang=user_language
        )
    else:
        lang = 'en'
        english_query = user_message
    
    # Get RAG response
    rag_result = get_rag_response(english_query)
    
    # Translate response back
    if translation_service and lang != 'en':
        translated_response = translation_service.translate_from_english(
            rag_result['response'],
            lang
        )
    else:
        translated_response = rag_result['response']
    
    return jsonify({
        'response': translated_response,
        'detected_language': lang,
        'sources': rag_result['sources']
    })
```

### Step 5: Run and Test

```bash
# Terminal 1: Start backend
cd backend
python app.py

# Terminal 2: Test translation
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "FIU-IND क्या है?"}'
```

## Complete Integration Steps

### 1. Backend Integration

#### A. Update `backend/app.py`

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add translation module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from translation import get_translation_service
from translation.model_loader import get_supported_languages

app = Flask(__name__)
CORS(app)

# Global translation service
translation_service = None

@app.before_first_request
def initialize_services():
    """Initialize translation service"""
    global translation_service
    try:
        translation_service = get_translation_service()
        print("✓ Translation service initialized")
    except Exception as e:
        print(f"✗ Translation service failed: {e}")
        translation_service = None

# Add new endpoints
@app.route('/api/translate/detect', methods=['POST'])
def detect_language():
    """Detect language of input text"""
    data = request.json
    text = data.get('text', '')
    
    if translation_service:
        lang = translation_service.detect_language(text)
        lang_name = get_supported_languages().get(lang, 'Unknown')
        return jsonify({
            'language': lang,
            'language_name': lang_name
        })
    return jsonify({'error': 'Service unavailable'}), 503

@app.route('/api/translate/supported', methods=['GET'])
def supported_languages():
    """Get supported languages"""
    return jsonify({
        'languages': get_supported_languages()
    })

# Update existing chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with automatic translation"""
    data = request.json
    user_message = data.get('message', '')
    user_language = data.get('language', None)
    
    # Translate to English
    if translation_service:
        lang, english_query, _ = translation_service.process_conversation(
            user_input=user_message,
            system_response="",
            user_lang=user_language
        )
    else:
        lang = 'en'
        english_query = user_message
    
    # Get RAG response (your existing logic)
    from rag_pipeline import get_rag_response
    rag_result = get_rag_response(english_query)
    english_response = rag_result['response']
    
    # Translate back to user language
    if translation_service and lang != 'en':
        translated_response = translation_service.translate_from_english(
            english_response,
            lang
        )
    else:
        translated_response = english_response
    
    return jsonify({
        'response': translated_response,
        'detected_language': lang,
        'language_name': get_supported_languages().get(lang, 'Unknown'),
        'english_query': english_query,
        'sources': rag_result.get('sources', []),
        'model': rag_result.get('model', 'gemma2:9b')
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check with translation status"""
    return jsonify({
        'status': 'healthy',
        'translation_service': 'available' if translation_service else 'unavailable',
        'supported_languages': len(get_supported_languages())
    })
```

#### B. Update `backend/rag_pipeline.py`

No changes needed! Translation is handled at the API layer.

### 2. Frontend Integration

#### A. Add Language Selector Component

Create `frontend/components/LanguageSelector.tsx`:

```typescript
'use client';

import { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface Language {
  code: string;
  name: string;
  nativeName: string;
}

export function LanguageSelector({ 
  onLanguageChange 
}: { 
  onLanguageChange: (lang: string) => void 
}) {
  const [languages, setLanguages] = useState<Language[]>([]);
  const [selectedLang, setSelectedLang] = useState('en');

  useEffect(() => {
    // Fetch supported languages
    fetch('http://localhost:5000/api/translate/supported')
      .then(res => res.json())
      .then(data => {
        const langs = Object.entries(data.languages).map(([code, name]) => ({
          code,
          name: name as string,
          nativeName: getNativeName(code)
        }));
        setLanguages(langs);
      });
  }, []);

  const getNativeName = (code: string): string => {
    const nativeNames: Record<string, string> = {
      en: 'English',
      hi: 'हिंदी',
      mr: 'मराठी',
      ta: 'தமிழ்',
      te: 'తెలుగు',
      bn: 'বাংলা',
      gu: 'ગુજરાતી',
      kn: 'ಕನ್ನಡ',
      ml: 'മലയാളം',
      or: 'ଓଡ଼ିଆ',
      pa: 'ਪੰਜਾਬੀ'
    };
    return nativeNames[code] || code;
  };

  const handleChange = (value: string) => {
    setSelectedLang(value);
    onLanguageChange(value);
  };

  return (
    <Select value={selectedLang} onValueChange={handleChange}>
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="Select Language" />
      </SelectTrigger>
      <SelectContent>
        {languages.map(lang => (
          <SelectItem key={lang.code} value={lang.code}>
            {lang.nativeName} ({lang.name})
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
```

#### B. Update Chat Page

Update `frontend/app/chat/page.tsx`:

```typescript
'use client';

import { useState } from 'react';
import { LanguageSelector } from '@/components/LanguageSelector';

export default function ChatPage() {
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [input, setInput] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: input,
          language: selectedLanguage === 'auto' ? null : selectedLanguage
        })
      });

      const data = await response.json();

      // Update detected language
      if (data.detected_language) {
        setDetectedLanguage(data.language_name);
      }

      // Add assistant response
      const assistantMessage = { role: 'assistant', content: data.response };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Chat error:', error);
    }

    setInput('');
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header with language selector */}
      <div className="p-4 border-b flex justify-between items-center">
        <h1 className="text-2xl font-bold">FIU-Sahayak</h1>
        <div className="flex gap-4 items-center">
          {detectedLanguage && (
            <span className="text-sm text-gray-600">
              Detected: {detectedLanguage}
            </span>
          )}
          <LanguageSelector onLanguageChange={setSelectedLanguage} />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg, i) => (
          <div key={i} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block p-3 rounded-lg ${
              msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type your message in any language..."
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={sendMessage}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 3. Testing the Integration

#### A. Test Backend

```bash
# Start backend
cd backend
python app.py

# In another terminal, test endpoints

# 1. Check health
curl http://localhost:5000/health

# 2. Test language detection
curl -X POST http://localhost:5000/api/translate/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "मुझे FIU-IND के बारे में बताएं"}'

# 3. Test chat with Hindi
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "FIU-IND क्या है?"}'

# 4. Test chat with Tamil
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "FIU-IND என்றால் என்ன?"}'
```

#### B. Test Frontend

```bash
# Start frontend
cd frontend
npm run dev

# Visit http://localhost:3000/chat
# Try sending messages in different languages
```

### 4. Environment Configuration

Create `backend/.env`:

```env
# Translation Settings
TRANSLATION_MODEL=sarvamai/sarvam-translate
TRANSLATION_DEVICE=cuda  # or cpu
ENABLE_TRANSLATION_CACHE=True
MAX_TRANSLATION_LENGTH=512

# Existing settings
DB_PATH=db_e5_section_aware
FLASK_ENV=development
```

### 5. Update README.md

Add to your main README:

```markdown
## 🌍 Multi-Language Support

FIU-Sahayak now supports **11 Indian languages**:
- English, Hindi, Marathi, Tamil, Telugu
- Bengali, Gujarati, Kannada, Malayalam
- Odia, Punjabi

### Usage
1. **Automatic Detection**: Simply type in your preferred language
2. **Manual Selection**: Use the language selector in the UI
3. **Seamless Translation**: All responses automatically translated

### Features
- Auto language detection
- Real-time translation
- Translation caching for performance
- 11 official Indian languages supported
```

### 6. Deployment Considerations

#### A. Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM with GPU
- **GPU**: CUDA-enabled GPU (optional but recommended)

#### B. Docker Support

Add to your `Dockerfile`:

```dockerfile
# Install translation dependencies
RUN pip install transformers torch sentencepiece accelerate

# Download model (optional - will auto-download on first use)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('sarvamai/sarvam-translate'); \
    AutoModelForSeq2SeqLM.from_pretrained('sarvamai/sarvam-translate')"
```

#### C. Performance Optimization

```python
# In backend/app.py startup
import torch

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable automatic mixed precision
torch.set_float32_matmul_precision('medium')
```

## Troubleshooting

### Issue: Model download fails

**Solution**:
```bash
# Manually download
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('sarvamai/sarvam-translate'); \
    AutoModelForSeq2SeqLM.from_pretrained('sarvamai/sarvam-translate')"
```

### Issue: Out of memory

**Solution**:
```python
# Use CPU instead
os.environ['TRANSLATION_DEVICE'] = 'cpu'

# Or reduce batch size
translator = TranslationService()
translator.model.config.max_length = 256  # Reduce from 512
```

### Issue: Slow translation

**Solution**:
```bash
# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Wrong language detection

**Solution**:
```javascript
// Explicitly specify language in frontend
fetch('/api/chat', {
  body: JSON.stringify({
    message: userInput,
    language: 'hi'  // Force Hindi
  })
})
```

## Next Steps

1. ✅ Test translation with sample queries
2. ✅ Update frontend with language selector
3. ✅ Test performance with caching
4. ⏳ Fine-tune on financial domain (optional)
5. ⏳ Add user feedback mechanism
6. ⏳ Monitor translation quality

## Support

For issues:
- Check logs: `translation_service.log`
- Run tests: `python translation/test_translation.py`
- Check health: `curl http://localhost:5000/health`

---

**Estimated Integration Time**: 30-60 minutes  
**Difficulty**: Moderate  
**Support**: GitHub Issues