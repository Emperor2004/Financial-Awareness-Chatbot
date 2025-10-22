"""
Flask Backend API for Financial Awareness Chatbot
Provides REST API endpoints for the frontend
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline, SUPPORTED_MODELS
import json
import time

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
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'llama3.2:3b')
rag_pipeline = None

def get_rag_pipeline():
    """Lazy initialization of RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline with E5 embeddings...")
        rag_pipeline = RAGPipeline(
            db_path=os.getenv('DB_PATH', 'db_e5'),
            model_name=DEFAULT_MODEL,
            k=int(os.getenv('RETRIEVAL_K', '5'))
        )
    return rag_pipeline


# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_rag_pipeline()
        return jsonify({
            'status': 'healthy',
            'model': pipeline.model_name,
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
        "k": 5  # optional, number of documents to retrieve
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
        
        # Get pipeline
        pipeline = get_rag_pipeline()
        
        # Temporarily switch model if requested
        original_model = pipeline.model_name
        if requested_model and requested_model in SUPPORTED_MODELS:
            pipeline.switch_model(requested_model)
        
        # Process query
        logger.info(f"Processing query from session {session_id}: {user_message[:50]}...")
        result = pipeline.query(user_message, k=k)
        
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
    For future implementation of streaming responses
    """
    # TODO: Implement streaming with ollama.generate(stream=True)
    return jsonify({'error': 'Streaming not yet implemented'}), 501


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
