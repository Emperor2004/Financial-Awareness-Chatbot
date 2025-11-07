# Financial Awareness Chatbot - FIU-IND NLP Project

## 🎯 Project Overview

**FIU-Sahayak** is an intelligent chatbot designed to provide accurate information about Indian financial regulations, PMLA compliance, money laundering prevention, and FIU-IND procedures. Built using Retrieval-Augmented Generation (RAG) architecture, it combines advanced language models with a comprehensive knowledge base of official financial documents.

## 🏆 Key Achievements

- ✅ **Functional Full-Stack Application**: Backend API + Frontend UI + Authentication
- ✅ **Advanced RAG Pipeline**: E5-large-v2 embeddings + ChromaDB vector store
- ✅ **Multi-Model Evaluation**: Comprehensive comparison of 3 LLMs (Llama 3.2, Mistral 7B, Gemma 2)
- ✅ **Production-Ready Architecture**: Scalable Flask backend + Next.js frontend
- ✅ **Comprehensive Test Dataset**: 50-question evaluation framework
- ✅ **Login System**: Secure authentication with protected routes

## 🚀 Live Application Features

### Backend (Flask API)
- **RAG Pipeline**: Advanced document retrieval and response generation
- **Model Management**: Dynamic switching between Ollama models
- **Health Monitoring**: System status and performance metrics
- **CORS Support**: Cross-origin requests for frontend integration

### Frontend (Next.js)
- **Modern UI**: Clean, responsive chat interface
- **Authentication**: Login/signup pages with protected routes
- **Real-time Chat**: Interactive conversation with the AI assistant
- **Source Citations**: Transparent document references
- **Theme Support**: Dark/light mode toggle

### Knowledge Base
- **E5-large-v2 Embeddings**: State-of-the-art semantic search
- **ChromaDB Vector Store**: Efficient document retrieval
- **Official Documents**: FIU-IND and Income Tax Department data
- **Smart Chunking**: Optimized document segmentation

## 🌐 Multilingual Support (Newly Added)

The FIU-Sahayak Chatbot now supports multilingual input and output across English, Hindi, and Marathi, including queries written in Roman script (e.g., "money laundering kya hai?" or "kyc mhanje kay?").

### 🔧 Implementation Overview

* Integrated Translation Module powered by Azure Cognitive Translator for bi-directional translation.
* Added automatic language detection using `langdetect` for Hindi, Marathi, and English.
* Introduced a Transliteration Handler that detects Roman-script Hindi/Marathi and converts it into Devanagari script using the `indic-transliteration` library.
* Ensured that RAG operates entirely in English, while users can interact in their preferred language seamlessly.

### 🧠 End-to-End Workflow

1. **User Input:**
   * Detects the language and script.
   * If Hindi/Marathi in Roman script → Transliterates to Devanagari.
   * Non-English queries are translated to English before RAG processing.

2. **RAG Processing:**
   * Retrieval and synthesis occur using English embeddings and documents.

3. **Output Translation:**
   * The English RAG response is translated back to the detected original language.
   * Output appears in Devanagari for Hindi/Marathi or in Roman script for English.

### 💬 Example Queries

| User Query | Auto-detected Language | Internal Processing | Final Output |
|------------|------------------------|---------------------|--------------|
| `What is PMLA?` | English | English → RAG → English | English |
| `मनी लॉन्ड्रिंग क्या है?` | Hindi | Hindi → English → RAG → Hindi | Hindi (Devanagari) |
| `money laundering kya hai?` | Roman Hindi | Roman → Devanagari → English → RAG → Hindi | Hindi (Devanagari) |
| `kyc mhanje kay?` | Roman Marathi | Roman → Devanagari → English → RAG → Marathi | Marathi (Devanagari) |

### ⚙️ Key Libraries

* `langdetect` → Language detection
* `requests` → Azure API communication
* `indic-transliteration` → Roman → Devanagari conversion
* `nltk`, `rouge-score` → Translation quality metrics (for testing)

### 🧪 Quality Handling

* Round-trip translation validation ensures reliability.
* Automatic fallback: short or mixed-language inputs skip strict validation to prevent false failures.
* Full error handling integrated with Flask API to prevent crashes (`500` errors).

### 💡 Benefits

* Seamless multilingual access for a diverse user base.
* Accurate responses from English-only financial knowledge base.
* Robust performance even for mixed-script or partial queries.
* Completely modular — can be reused for other RAG-based multilingual projects.

### 🧠 Optional Addition to "Technology Stack" Section

You can add one more row like this:

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language & Translation | Azure Cognitive Translator, `langdetect`, `indic-transliteration` | Multilingual support for English, Hindi, and Marathi (with Roman-script detection) |

### 📘 For `translation/TranslationREADME.md` (module-specific)

#### 🪄 New Feature: Roman-Script Handling

This module now supports automatic transliteration of Roman-script Hindi and Marathi into Devanagari script before translation.

##### How it Works

1. Detects if input text (like `"money laundering kya hai?"`) matches common Roman Hindi or Marathi patterns.
2. Uses `indic-transliteration` to convert it into Devanagari (`"मनी लॉन्ड्रिंग क्या है?"`).
3. The standard translation workflow (to English and back) then proceeds unchanged.

##### Benefits

* Allows users to type naturally without switching keyboard scripts.
* Improves detection accuracy and translation reliability.
* Requires no configuration changes — works automatically as part of `trans_for_rag()`.

## 📊 Model Performance Results

### Three-Model Comparison (50 Questions)

| Model | Overall Score | Response Time | Best Metric |
|-------|-------------|---------------|-------------|
| **Mistral 7B Instruct** | **0.272** | 19.11s | Balanced Performance |
| Gemma 2 9B | 0.271 | 37.77s | Semantic Similarity (0.59) |
| Llama 3.2 3B | 0.255 | 8.21s | Speed Champion |

### Key Insights
- **Winner**: Mistral 7B Instruct (best overall balance)
- **Retrieval Quality**: Consistent F1 scores (~0.43-0.44) across all models
- **E5 Embeddings**: Effective semantic search with 0.68-0.69 recall
- **Speed vs Quality**: Clear trade-offs identified

## 🛠 Technology Stack

### Backend
- **Python 3.13**: Core application language
- **Flask**: Web framework and API server
- **LangChain**: RAG pipeline orchestration
- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: E5-large-v2 embedding model

### Frontend
- **Next.js 14**: React framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Component library
- **React Hook Form**: Form management

### AI/ML
- **Embeddings**: intfloat/e5-large-v2
- **LLMs**: Llama 3.2 3B, Mistral 7B Instruct, Gemma 2 9B
- **Evaluation**: BLEU, ROUGE, Semantic Similarity, F1 scores
- **Vector Search**: ChromaDB with similarity search

## 📁 Project Structure

```
Financial-Awareness-Chatbot/
├── ai_core/
│   └── ingest_e5.py              # E5 embeddings data ingestion
├── backend/
│   ├── app.py                    # Flask API server
│   ├── rag_pipeline.py          # Core RAG implementation
│   └── db_e5/                   # E5 embeddings database
├── frontend/
│   ├── app/                     # Next.js pages
│   │   ├── chat/               # Chat interface
│   │   ├── login/              # Authentication
│   │   └── signup/             # User registration
│   ├── components/              # React components
│   └── lib/                    # Utilities
├── translation/                      # Multilingual translation & validation module
│   ├── __init__.py                   # Package initializer
│   ├── translator.py                 # Core translation logic (Azure API + validation)
│   ├── translation_validator.py      # Quality check & similarity metrics
│   ├── transliteration_handler.py    # Roman-script detection & conversion (new)
│   ├── TranslationREADME.md          # Detailed documentation for this module
│   │
│   └── tests/                        # Unit & integration tests for translation module
│       ├── __init__.py
│       ├── translation_module_test.py # Simulated RAG integration tests
│       ├── test_translation.py        # Full NLP evaluation (BLEU, ROUGE, etc.)
│       │
│       ├── data/                     # Test data for translation validation
│       │   ├── edge_case_test_cases.json
│       │   └── translation_test_cases.json
│       │
│       └── logs/                     # Test logs (timestamped)
│           └── ...
├── evaluation/
│   ├── compare_models.py        # Model comparison script
│   ├── metrics.py              # Evaluation metrics
│   ├── test_dataset_template.json # 50-question test dataset
│   └── results_e5_three_models/ # Latest evaluation results
├── data/
│   ├── fiu/                    # FIU-IND documents
│   └── incometax/              # Income Tax documents
├── scripts/                    # Data scraping scripts
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+
- Ollama installed locally
- Git


### 1. Clone and Setup
```bash
git clone <repository-url>
cd Financial-Awareness-Chatbot
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv fin_venv
fin_venv\Scripts\activate  # Windows
# source fin_venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download Ollama models
ollama pull llama3.2:3b
ollama pull mistral:7b-instruct
ollama pull gemma2:9b

# Run backend
cd backend
python app.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Chat Interface**: http://localhost:3000/chat

## 🔧 Configuration

### Environment Variables
Create `.env` file in backend directory:
```env
DB_PATH=db_e5
FLASK_ENV=development
FLASK_DEBUG=True
```

### Model Configuration
Models are configured in `backend/rag_pipeline.py`:
- **Default Model**: llama3.2:3b
- **Embeddings**: intfloat/e5-large-v2
- **Retrieval Count**: 5 documents
- **Temperature**: 0.1 (factual responses)

## 📈 Evaluation Framework

### Test Dataset
- **50 Questions**: Comprehensive coverage of financial domains
- **Categories**: Factual Recall, Comparative Analysis, Procedural, Scenario-Based, Adversarial
- **Domains**: FIU-IND & PMLA, Income Tax, Out-of-Scope

### Metrics Used
- **BLEU Score**: Exact word overlap
- **ROUGE Scores**: N-gram overlap (1, 2, L)
- **Semantic Similarity**: Meaning-based comparison
- **Retrieval Metrics**: Precision, Recall, F1
- **Performance**: Response time, token count

### Running Evaluation
```bash
cd evaluation
python compare_models.py --models llama3.2:3b mistral:7b-instruct gemma2:9b --test-dataset test_dataset_template.json
```

## 🎯 Key Features Implemented

### 1. Advanced RAG Pipeline
- **E5-large-v2 Embeddings**: Superior semantic understanding
- **ChromaDB Integration**: Efficient vector search
- **Context Synthesis**: Coherent narrative generation
- **Source Attribution**: Transparent document references

### 2. Multi-Model Support
- **Dynamic Switching**: Runtime model selection
- **Performance Monitoring**: Response time tracking
- **Error Handling**: Graceful failure management
- **Model Metadata**: Detailed model information

### 3. Production-Grade Security
- **PII Protection**: No personal data collection
- **Input Validation**: Query sanitization
- **Error Boundaries**: Safe error handling
- **CORS Configuration**: Secure cross-origin requests

### 4. User Experience
- **Responsive Design**: Mobile-friendly interface
- **Real-time Chat**: Instant responses
- **Loading States**: User feedback
- **Theme Support**: Dark/light modes

## 📊 Performance Benchmarks

### Retrieval Performance
- **F1 Score**: 0.43-0.44 (Good)
- **Precision**: 0.35-0.36 (Moderate)
- **Recall**: 0.68-0.69 (Excellent)
- **Response Time**: 8-38 seconds (Model dependent)

### Generation Quality
- **Semantic Similarity**: 0.55-0.59 (Good)
- **ROUGE-L**: 0.16-0.17 (Moderate)
- **BLEU**: 0.03-0.04 (Low - common for generative models)

## 🔮 Next Steps & Future Enhancements

### Immediate Priorities
1. **Fine-tuning**: Optimize Mistral 7B on financial domain
2. **Retrieval Optimization**: Improve precision scores
3. **Performance Tuning**: Reduce response times
4. **Mobile App**: React Native implementation

### Long-term Goals
1. **Multi-language Support**: Hindi, Marathi, regional languages
2. **Advanced Analytics**: User interaction insights
3. **Integration**: FIU-IND official systems
4. **Scalability**: Cloud deployment and load balancing

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Run tests and evaluation
5. Submit pull request

### Code Standards
- **Python**: PEP 8 compliance
- **TypeScript**: Strict type checking
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests

## 📄 License

This project is developed for academic and research purposes. Please ensure compliance with FIU-IND guidelines and Indian financial regulations.

## 📞 Support

For technical support or questions:
- **Issues**: GitHub Issues
- **Documentation**: Project Wiki
- **Contact**: Project maintainers

---

## 🏆 Project Status: **PRODUCTION READY**

✅ **Core Features**: Complete  
✅ **Evaluation Framework**: Implemented  
✅ **Multi-Model Support**: Functional  
✅ **Frontend Integration**: Working  
✅ **Authentication**: Secure  
✅ **Documentation**: Comprehensive  

**Ready for deployment and further development!** 🚀