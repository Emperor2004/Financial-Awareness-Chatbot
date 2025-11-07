"""
Flask Backend API for Financial Awareness Chatbot
Provides REST API endpoints for the frontend
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os, sys
import logging
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline, SUPPORTED_MODELS
import json
import time

# translation related imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from translation.translator import Translator, TranslationQualityError
import translation.translation_validator as tv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize RAG pipeline
# Default model - can be changed via API
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gemma2:9b')
rag_pipeline = None

def get_rag_pipeline():
    """Lazy initialization of RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline with E5 embeddings...")
        rag_pipeline = RAGPipeline(
            db_path=os.getenv('DB_PATH', None),  # None = auto-detect db_e5_section_aware
            model_name=DEFAULT_MODEL,
            k=int(os.getenv('RETRIEVAL_K', '7')),  # Default: 7 (adaptive 3-10 based on query)
            use_reranker=os.getenv('USE_RERANKER', 'true').lower() == 'true',  # Default: True
            reranker_model=os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base'),
            rerank_k=int(os.getenv('RERANK_K', '50')) if os.getenv('RERANK_K') else None,  # Default: 50
            reranker_batch_size=int(os.getenv('RERANKER_BATCH_SIZE', '8'))  # Default: 8 (optimized for 6GB VRAM)
        )
    return rag_pipeline

translator_service = None

def get_translator():
    """Lazy initialize translator service"""
    global translator_service
    if translator_service is None:
        try:
            translator_service = Translator()
            logger.info("Translator service initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Translator: {e}")
            translator_service = None
    return translator_service


# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_rag_pipeline()
        
        # Check GPU and reranker status
        gpu_status = {}
        try:
            import torch
            gpu_status = {
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except ImportError:
            gpu_status = {'cuda_available': False, 'error': 'PyTorch not available'}
        
        return jsonify({
            'status': 'healthy',
            'model': pipeline.model_name,
            'reranker_enabled': pipeline.use_reranker,
            'reranker_loaded': pipeline.reranker is not None,
            'gpu': gpu_status,
            'timestamp': time.time()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify({
        'current_model': get_rag_pipeline().model_name,
        'available_models': SUPPORTED_MODELS
    }), 200


@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'error': 'model_name is required'}), 400
    
    if model_name not in SUPPORTED_MODELS:
        return jsonify({
            'error': f'Model {model_name} not supported',
            'available_models': list(SUPPORTED_MODELS.keys())
        }), 400
    
    try:
        pipeline = get_rag_pipeline()
        pipeline.switch_model(model_name)
        
        return jsonify({
            'success': True,
            'current_model': model_name,
            'message': f'Switched to {SUPPORTED_MODELS[model_name]["name"]}'
        }), 200
    
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    
    Request body:
    {
        "message": "What is PMLA?",
        "session_id": "session_123",  # optional
        "model": "mistral:7b-instruct",  # optional
        "k": 5,  # optional, number of documents to retrieve
        "conversation_history": [  # optional, for conversation memory
            {"role": "user", "content": "What is PMLA?"},
            {"role": "assistant", "content": "PMLA is..."}
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'message' not in data:
            return jsonify({'error': 'message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'message cannot be empty'}), 400
        
        # Optional parameters
        session_id = data.get('session_id', 'default')
        k = data.get('k', None)
        requested_model = data.get('model')
        conversation_history = data.get('conversation_history', [])
        
        # Get pipeline
        pipeline = get_rag_pipeline()
        
        # Temporarily switch model if requested
        original_model = pipeline.model_name
        if requested_model and requested_model in SUPPORTED_MODELS:
            pipeline.switch_model(requested_model)
        
        # Process query with conversation history
        """logger.info(f"Processing query from session {session_id}: {user_message[:50]}...")
        result = pipeline.query(user_message, k=k)"""

        logger.info(f"Processing query from session {session_id}: {user_message[:50]}...")
        logger.info(f"Conversation history length: {len(conversation_history)}")
        result = pipeline.query(user_message, k=k, conversation_history=conversation_history)
        
        # Restore original model if changed
        if requested_model and requested_model != original_model:
            pipeline.switch_model(original_model)
        
        # Return response
        response = {
            'success': True,
            'response': result['answer'],
            'sources': result['sources'],
            'metadata': result['metadata'],
            'session_id': session_id
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I apologize, but I encountered an error processing your request. Please try again."
        }), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint (Server-Sent Events)
    
    Request body:
    {
        "message": "What is PMLA?",
        "session_id": "session_123",  # optional
        "model": "mistral:7b-instruct",  # optional
        "k": 5,  # optional, number of documents to retrieve
        "conversation_history": [  # optional, for conversation memory
            {"role": "user", "content": "What is PMLA?"},
            {"role": "assistant", "content": "PMLA is..."}
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'message' not in data:
            return jsonify({'error': 'message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'message cannot be empty'}), 400
        
        # Optional parameters
        session_id = data.get('session_id', 'default')
        k = data.get('k', None)
        requested_model = data.get('model')
        conversation_history = data.get('conversation_history', [])
        
        # Get pipeline
        pipeline = get_rag_pipeline()
        
        # Temporarily switch model if requested
        original_model = pipeline.model_name
        if requested_model and requested_model in SUPPORTED_MODELS:
            pipeline.switch_model(requested_model)
        
        # Process query with streaming
        logger.info(f"Processing streaming query from session {session_id}: {user_message[:50]}...")
        logger.info(f"Conversation history length: {len(conversation_history)}")
        
        def generate():
            try:
                # Stream from pipeline
                for item in pipeline.query_stream(user_message, k=k, conversation_history=conversation_history):
                    if item['type'] == 'sources':
                        yield f"data: {json.dumps({'type': 'sources', 'data': item['data']})}\n\n"
                    elif item['type'] == 'chunk':
                        yield f"data: {json.dumps({'type': 'chunk', 'data': item['data']})}\n\n"
                    elif item['type'] == 'done':
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            finally:
                # Restore original model if changed
                if requested_model and requested_model != original_model:
                    pipeline.switch_model(original_model)
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
    
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/retrieve', methods=['POST'])
def retrieve_documents():
    """
    Retrieve documents without generating response
    Useful for debugging and testing retrieval quality
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'query is required'}), 400
        
        query = data['query']
        k = data.get('k', 5)
        
        pipeline = get_rag_pipeline()
        docs = pipeline.retrieve_documents(query, k=k)
        
        return jsonify({
            'success': True,
            'query': query,
            'documents': docs,
            'count': len(docs)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in retrieve endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        pipeline = get_rag_pipeline()
        
        # Get collection info from ChromaDB
        collection = pipeline.vectordb._collection
        count = collection.count()
        
        return jsonify({
            'success': True,
            'total_documents': count,
            'collection_name': 'financial_regulations',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'current_llm': pipeline.model_name
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    # Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'  # Changed default to False
    
    logger.info("="*60)
    logger.info("Financial Awareness Chatbot - Backend API")
    logger.info("="*60)
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Debug: {DEBUG}")
    logger.info(f"Default Model: {DEFAULT_MODEL}")
    logger.info("="*60)
    
    # Run Flask app
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        use_reloader=False  # Disable auto-reloader to prevent restart loops
    )
