<<<<<<< HEAD
"""
Flask Backend API for Financial Awareness Chatbot
Provides REST API endpoints for the frontend
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import logging
=======
# app.py

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Set

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
>>>>>>> main
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline, SUPPORTED_MODELS
import json
import time

<<<<<<< HEAD
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
=======
# --- 1. Load Environment Variables and API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- 2. Initialize FastAPI Application ---
app = FastAPI(
    title="Financial Information Retrieval Assistant API",
    description="An API for querying a financial knowledge base.",
    version="1.0.0",
>>>>>>> main
)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
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
=======
# --- 3. CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows your frontend (running on a different port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 4. Load AI and Data Components ---
try:
    print("Loading AI and data components...")
    
    # Initialize the LLM (Large Language Model)
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Load the persistent vector database
    persist_directory = 'db'
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector database not found. Please run ingest.py first.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Create the core RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 results
        return_source_documents=True,
        verbose=False, # Set to True for debugging
    )
    print("Components loaded successfully.")

except Exception as e:
    print(f"Error loading components: {e}")
    qa_chain = None # Set to None if loading fails

# --- 5. Define API Request and Response Models ---
class ChatRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str

class ChatResponse(BaseModel):
    answer: str
    sources: Set[str] # Use a set to automatically handle unique sources

# --- 6. Define the API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Receives a question, processes it through the RAG chain, and returns the answer with sources.
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG chain is not available. Check server logs.")

    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        print(f"Received query: {request.question}")
        llm_response = qa_chain.invoke(request.question)
        
        # Process the response to extract the answer and unique source file paths
        answer = llm_response.get('result', 'Sorry, I could not find an answer.')
        
        # Extract the filename from the full path of the source documents
        source_filenames = {
            os.path.basename(doc.metadata.get('source', 'N/A'))
            for doc in llm_response.get("source_documents", [])
        }
        
        return ChatResponse(answer=answer, sources=source_filenames)

    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# --- 7. Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

# To run this app, use the command: uvicorn app:app --reload
>>>>>>> main
