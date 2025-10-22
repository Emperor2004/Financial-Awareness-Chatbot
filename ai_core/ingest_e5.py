# ai_core/ingest_e5.py
import os
import logging
from typing import List

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_knowledge_base_e5():
    """
    Build financial knowledge base using E5-large-v2 embeddings
    """
    
    # 1. Load documents
    data_directory = './data'
    
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory '{data_directory}' not found.")
    
    logger.info(f"Loading documents from: {data_directory}")
    
    loader = DirectoryLoader(
        data_directory, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
    )
    
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    if not documents:
        raise ValueError("No documents found in data directory")
    
    # Add metadata
    for doc in documents:
        filename = os.path.basename(doc.metadata['source'])
        doc.metadata['filename'] = filename
        doc.metadata['word_count'] = len(doc.page_content.split())
        
        # Categorize document type based on filename
        filename_lower = filename.lower()
        if 'rbi' in filename_lower:
            doc.metadata['doc_type'] = 'RBI'
        elif 'sebi' in filename_lower:
            doc.metadata['doc_type'] = 'SEBI' 
        elif 'fiu' in filename_lower:
            doc.metadata['doc_type'] = 'FIU'
        elif 'faq' in filename_lower:
            doc.metadata['doc_type'] = 'FAQ'
        elif 'incometax' in filename_lower or 'tax' in filename_lower:
            doc.metadata['doc_type'] = 'Income Tax'
        else:
            doc.metadata['doc_type'] = 'General'
    
    logger.info("Document metadata added")
    
    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(texts)} chunks")
    
    # Add chunk metadata
    for i, chunk in enumerate(texts):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_length'] = len(chunk.page_content)
    
    # 3. Create E5-large-v2 embeddings
    logger.info("Initializing E5-large-v2 embeddings...")
    logger.info("Downloading E5 model (first time only, ~1.3GB)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",  # E5-large-v2 model
        model_kwargs={'device': 'cpu'},  # Use CPU (CUDA not available)
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 8  # Smaller batch for CPU
        }
    )
    
    logger.info("E5 embedding model loaded")
    
    # 4. Create vector database
    persist_directory = 'db_e5'  # Different directory for E5
    
    logger.info("Creating vector database with E5 embeddings...")
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="financial_regulations_e5"
    )
    
    vectordb.persist()
    logger.info(f"Vector database saved to '{persist_directory}'")
    logger.info(f"Total chunks: {len(texts)}")
    
    return vectordb

def search_knowledge_e5(query: str, db_path: str = "db_e5", k: int = 5) -> List[str]:
    """
    Search function using E5 embeddings
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8
            }
        )
        
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="financial_regulations_e5"
        )
        
        docs = vectordb.similarity_search(query, k=k)
        
        relevant_chunks = []
        for doc in docs:
            relevant_chunks.append(doc.page_content.strip())
        
        logger.info(f"Found {len(relevant_chunks)} chunks for: '{query}'")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def test_search_e5():
    """Test search functionality with E5"""
    print("\n" + "="*60)
    print("TESTING E5 SEARCH")
    print("="*60)
    
    test_queries = [
        "What is PMLA?",
        "RBI penalties",
        "SEBI compliance",
        "Income tax benefits"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        results = search_knowledge_e5(query, k=2)
        if results:
            for i, chunk in enumerate(results, 1):
                preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                print(f"\nResult {i}: {preview}")
        else:
            print("No results")
    
    print(f"\n{'='*60}")

def main():
    """Main function"""
    try:
        print("="*60)
        print("Financial Knowledge Base - E5-large-v2 Embeddings")
        print("="*60)
        
        vectordb = build_knowledge_base_e5()
        
        print("\nE5 Knowledge Base Built Successfully!")
        print(f"Database: ./db_e5/")
        print("Using E5-large-v2 embeddings")
        
        test_search_e5()
        
        print(f"\nFor RAG Pipeline:")
        print("Update rag_pipeline.py to use db_path='db_e5'")
        
    except Exception as e:
        logger.error(f"Build failed: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
