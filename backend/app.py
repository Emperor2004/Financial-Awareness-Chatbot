# app.py

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Set

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

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
)

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