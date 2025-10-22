"""
Document Ingestion Script for Financial Awareness Chatbot
Loads documents, splits them, creates embeddings, and persists them into ChromaDB
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def ingest_documents(data_path: str = "./data", persist_directory: str = "db"):
    """
    Loads documents, splits them, creates embeddings, and persists them into a ChromaDB vector store.
    
    Args:
        data_path: Path to directory containing text files
        persist_directory: Directory to persist the vector database
    """
    logger.info("============================================================")
    logger.info("Financial Knowledge Base - HuggingFace Embeddings")
    logger.info("============================================================")
    
    # 1. Load documents
    logger.info(f"Loading documents from: {data_path}")
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Add metadata to documents (e.g., source file name)
    for doc in documents:
        file_path = doc.metadata.get('source')
        if file_path:
            doc.metadata['document'] = os.path.basename(file_path)
    logger.info("Document metadata added")
    
    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # 3. Create embeddings
    logger.info("Initializing HuggingFace embeddings...")
    logger.info("Downloading all-MiniLM-L6-v2 model (first time only, ~90MB)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU (CUDA not available)
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Embedding model loaded")
    
    # 4. Create vector database
    logger.info(f"Creating vector database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="financial_regulations"
    )
    vectordb.persist()
    logger.info(f"Vector database saved to '{persist_directory}'")
    logger.info(f"Total chunks: {len(chunks)}")

def search_knowledge(query: str, db_path: str = "db", k: int = 5) -> list[str]:
    """
    Search function using HuggingFace embeddings
    
    Args:
        query: Search query
        db_path: Path to the vector database
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved document contents
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="financial_regulations"
        )
        docs = vectordb.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return []

if __name__ == "__main__":
    ingest_documents()
