# scripts/ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# Ensure your GOOGLE_API_KEY is set in your .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 1. Load your scraped text files from the main 'data' directory
print("Loading documents from all data sources...")
loader = DirectoryLoader(
    './data/', 
    glob="**/*.txt", 
    loader_cls=TextLoader,
    # --- THIS IS THE CRUCIAL FIX ---
    # Specify UTF-8 encoding to handle special characters correctly
    loader_kwargs={'encoding': 'utf-8'},
    # --- END OF FIX ---
    show_progress=True,
    use_multithreading=True
)

try:
    documents = loader.load()
except Exception as e:
    print(f"An error occurred during document loading: {e}")
    print("Please check for corrupted files in your 'data' directory.")
    documents = [] # Ensure documents is a list to avoid further errors

if not documents:
    print("No documents were loaded. Exiting.")
    exit()

print(f"Successfully loaded {len(documents)} documents.")

# 2. Split the documents into smaller chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# 3. Create the embedding model
print("Initializing embedding model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. Set up the Chroma vector database
print("Creating and persisting the vector database...")
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()
print("Vector database has been created and saved successfully.")