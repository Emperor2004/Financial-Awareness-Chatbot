# scripts/knowledge_base.py
import os
import logging
from typing import List
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Environment management
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinancialKnowledgeBase:
    """
    Financial Knowledge Base using LangChain + Google AI Embeddings + Chroma
    Optimized for FIU-IND regulatory documents
    """
    
    def __init__(self, data_dir: str = "./data/fiu", db_dir: str = "./db"):
        """
        Initialize the knowledge base
        
        Args:
            data_dir: Directory containing scraped text files
            db_dir: Directory to store the vector database
        """
        self.data_dir = data_dir
        self.db_dir = db_dir
        
        # Ensure Google API key is set
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        os.environ["GOOGLE_API_KEY"] = self.google_api_key
        
        # Initialize components
        self.embeddings = None
        self.vectordb = None
        
        logger.info(f"Initialized FinancialKnowledgeBase")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Database directory: {self.db_dir}")
    
    def load_documents(self) -> List[Document]:
        """
        Load all text documents from the data directory
        """
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
            # Load documents using DirectoryLoader
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",  # Load all .txt files recursively
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
            )
            
            documents = loader.load()
            
            if not documents:
                raise ValueError("No documents found in the data directory")
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Add metadata to documents for better tracking
            for doc in documents:
                # Extract filename from source path
                filename = os.path.basename(doc.metadata.get('source', 'unknown'))
                doc.metadata.update({
                    'filename': filename,
                    'document_type': self._classify_document_type(filename),
                    'word_count': len(doc.page_content.split())
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def _classify_document_type(self, filename: str) -> str:
        """
        Classify document type based on filename for better metadata
        """
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['rbi', 'reserve_bank']):
            return 'RBI_Regulation'
        elif any(term in filename_lower for term in ['sebi', 'securities']):
            return 'SEBI_Regulation'
        elif any(term in filename_lower for term in ['fiu', 'financial_intelligence']):
            return 'FIU_Document'
        elif 'faq' in filename_lower:
            return 'FAQ'
        elif any(term in filename_lower for term in ['advisory', 'warning']):
            return 'Advisory'
        elif any(term in filename_lower for term in ['order', 'penalty']):
            return 'Enforcement_Action'
        else:
            return 'General_Document'
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks optimized for financial content
        """
        try:
            # Configure text splitter for financial documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,        # Reasonable size for regulatory content
                chunk_overlap=200,      # Good overlap to maintain context
                length_function=len,
                separators=[
                    "\n\n\n",          # Multiple line breaks
                    "\n\n",            # Paragraph breaks
                    "\n",              # Single line breaks
                    ". ",              # Sentence endings
                    ", ",              # Clause breaks
                    " ",               # Word breaks
                    ""                 # Character level (last resort)
                ]
            )
            
            # Split documents
            texts = text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(texts):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content)
                })
            
            logger.info(f"Split {len(documents)} documents into {len(texts)} chunks")
            return texts
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def create_embeddings(self):
        """
        Initialize Google Generative AI embeddings
        """
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            logger.info("Created Google Generative AI embeddings model")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def build_vector_database(self, texts: List[Document]):
        """
        Build and persist the Chroma vector database
        """
        try:
            # Create the vector database
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.db_dir,
                collection_name="financial_regulations"
            )
            
            # Persist the database
            self.vectordb.persist()
            
            logger.info(f"Vector database created and saved to {self.db_dir}")
            logger.info(f"Total chunks in database: {len(texts)}")
            
        except Exception as e:
            logger.error(f"Error building vector database: {str(e)}")
            raise
    
    def load_existing_database(self):
        """
        Load an existing vector database
        """
        try:
            if not os.path.exists(self.db_dir):
                raise FileNotFoundError(f"Database directory not found: {self.db_dir}")
            
            # Initialize embeddings if not already done
            if not self.embeddings:
                self.create_embeddings()
            
            # Load existing database
            self.vectordb = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
                collection_name="financial_regulations"
            )
            
            logger.info(f"Loaded existing vector database from {self.db_dir}")
            
        except Exception as e:
            logger.error(f"Error loading existing database: {str(e)}")
            raise
    
    def build_knowledge_base(self) -> bool:
        """
        Complete pipeline to build the knowledge base
        """
        try:
            logger.info("=== Starting Knowledge Base Construction ===")
            
            # 1. Load documents
            documents = self.load_documents()
            
            # 2. Split into chunks
            texts = self.split_documents(documents)
            
            # 3. Create embeddings
            self.create_embeddings()
            
            # 4. Build vector database
            self.build_vector_database(texts)
            
            logger.info("=== Knowledge Base Built Successfully ===")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {str(e)}")
            return False

def search_knowledge(query: str, db_dir: str = "./db", k: int = 5) -> List[str]:
    """
    DELIVERABLE FUNCTION: Search the knowledge base for relevant chunks
    
    Args:
        query: Search query from user
        db_dir: Path to vector database directory  
        k: Number of top results to return
        
    Returns:
        List of relevant text chunks
    """
    try:
        # Load environment variables
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found")
            return []
        
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Load vector database
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name="financial_regulations"
        )
        
        # Perform similarity search
        docs = vectordb.similarity_search(query, k=k)
        
        # Extract text content
        relevant_chunks = [doc.page_content for doc in docs]
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in search_knowledge: {str(e)}")
        return []

def search_knowledge_with_metadata(query: str, db_dir: str = "./db", k: int = 5) -> List[dict]:
    """
    Enhanced search function that returns chunks with metadata
    
    Args:
        query: Search query from user
        db_dir: Path to vector database directory
        k: Number of top results to return
        
    Returns:
        List of dictionaries with text content and metadata
    """
    try:
        # Load environment variables
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found")
            return []
        
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Load vector database
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name="financial_regulations"
        )
        
        # Perform similarity search with scores
        docs_with_scores = vectordb.similarity_search_with_score(query, k=k)
        
        # Format results with metadata
        results = []
        for doc, score in docs_with_scores:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(1 - score),  # Convert distance to similarity
                'source_file': doc.metadata.get('filename', 'unknown'),
                'document_type': doc.metadata.get('document_type', 'unknown')
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} relevant chunks with metadata")
        return results
        
    except Exception as e:
        logger.error(f"Error in search_knowledge_with_metadata: {str(e)}")
        return []

def main():
    """
    Main function to build and test the knowledge base
    """
    # Configuration
    DATA_DIR = "./data/fiu"  # Adjust path to your scraped data
    DB_DIR = "./db"          # Vector database storage
    
    # Initialize knowledge base
    kb = FinancialKnowledgeBase(data_dir=DATA_DIR, db_dir=DB_DIR)
    
    # Build knowledge base
    success = kb.build_knowledge_base()
    
    if success:
        # Test search functionality
        test_queries = [
            "What is money laundering?",
            "RBI penalty for banks",
            "FIU-IND reporting requirements",
            "SEBI regulations compliance",
            "KYC documentation requirements"
        ]
        
        print("\n=== Testing Search Functionality ===")
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Test basic search
            results = search_knowledge(query, DB_DIR, k=2)
            
            if results:
                for i, chunk in enumerate(results, 1):
                    preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                    print(f"\nResult {i}: {preview}")
            else:
                print("No results found")
            
            print("-" * 60)
        
        print(f"\n✅ Knowledge base is ready at: {DB_DIR}")
        print("📋 Use search_knowledge(query) for basic search")
        print("📋 Use search_knowledge_with_metadata(query) for detailed results")
        
    else:
        print("❌ Failed to build knowledge base")

if __name__ == "__main__":
    main()