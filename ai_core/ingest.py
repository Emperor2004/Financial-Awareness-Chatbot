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

# 1. Load your scraped text files
# Assume you have scraped data and saved it in a 'data/' directory
loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. Create the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. Set up the Chroma vector database
# This will create and save the DB in the 'db' directory
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()
print("Vector database has been created and saved.")