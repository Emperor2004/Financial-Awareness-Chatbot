"""
Section-Aware Document Ingestion Script for Financial Awareness Chatbot

This script implements a Hybrid Chunking Strategy:

1. Primary Strategy: Major Section Splitting
   - Splits documents ONLY by top-level section boundaries (e.g., "Section 4", "Section 13")
   - Each chunk contains the COMPLETE section content (including all subsections like (a), (i))
   - This prevents mixing different sections (e.g., Section 4 and Section 13) in the same chunk
   - Subsection markers like (a), (i), (1) are NOT used as split points

2. Fallback Strategy: Paragraph Splitting
   - For documents without major section headers (penalty orders, IT files, etc.)
   - Falls back to paragraph-based splitting (\n\n)

Strategy Rationale:
- Legal documents (like PMLA Act) are chunked by complete sections
- This ensures Section 4 and Section 13 remain in separate chunks
- The reranker then chooses between "Complete Section 4" vs "Complete Section 13"
- This is much easier and more accurate than choosing between mixed chunks

Designed for high-performance RAG pipeline with k=50 retrieval + reranker.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION HEADER REGEX PATTERNS
# ============================================================================

def get_major_section_pattern() -> re.Pattern:
    """
    Creates a regex pattern to identify ONLY major, top-level section boundaries.
    
    This pattern matches ONLY parent-level headers, NOT subsections:
    - "Section 4"
    - "SECTION 13"
    - "PART I"
    - "CHAPTER 2"
    - "Article 14"
    
    Does NOT match:
    - "(a)", "(i)", "(1)" - these are subsections (children)
    - "Section 4(2)(a)" - only matches "Section 4" part
    
    Strategy: Split by major boundaries only to create complete section chunks.
    Each chunk will contain the ENTIRE section content (including all subsections).
    
    Returns:
        Compiled regex pattern for major section headers only
    """
    # Pattern matches:
    # - Start of line (after optional whitespace)
    # - Major section markers: Section, PART, CHAPTER, Article
    # - Followed by number or roman numeral
    # - Word boundary to ensure we don't match mid-word
    
    # Pattern matches major section headers at the start of a line
    # Uses (?m) for multiline mode so ^ matches start of each line
    # Uses (?i) for case-insensitive matching
    pattern = r"(?mi)^\s*(?:Section\s+\d+|PART\s+[IVXLC]+|CHAPTER\s+\d+|Article\s+\d+)\b"
    
    return re.compile(pattern)


# ============================================================================
# CHUNKING FUNCTIONS
# ============================================================================

def split_oversized_chunk(
    chunk_text: str, 
    section_header: Optional[str],
    max_chunk_size: int = 2000
) -> List[Tuple[str, Optional[str]]]:
    """
    Splits a chunk that exceeds max_chunk_size into smaller chunks.
    
    Attempts to split at natural boundaries (sentences, lines) to preserve context.
    Preserves section header for all sub-chunks if provided.
    
    Args:
        chunk_text: Text chunk that needs to be split
        section_header: Original section header (if any)
        max_chunk_size: Maximum size for resulting chunks
        
    Returns:
        List of tuples: (chunk_text, section_header)
    """
    if len(chunk_text) <= max_chunk_size:
        return [(chunk_text, section_header)]
    
    chunks = []
    
    # Try splitting by double newlines first (paragraphs within section)
    paragraphs = re.split(r'\n\n+', chunk_text)
    
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If single paragraph exceeds max_chunk_size, split it further
        if para_size > max_chunk_size:
            # Save current chunk if exists
            if current_chunk:
                chunk_text_combined = '\n\n'.join(current_chunk)
                if chunk_text_combined.strip():
                    chunks.append((chunk_text_combined.strip(), section_header))
                current_chunk = []
                current_size = 0
            
            # Split oversized paragraph by single newlines
            lines = para.split('\n')
            para_chunk = []
            para_chunk_size = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_size = len(line)
                
                # If adding line would exceed max, save current para_chunk
                if para_chunk_size + line_size + 1 > max_chunk_size and para_chunk:
                    chunk_text_combined = '\n'.join(para_chunk)
                    if chunk_text_combined.strip():
                        chunks.append((chunk_text_combined.strip(), section_header))
                    para_chunk = [line]
                    para_chunk_size = line_size
                else:
                    para_chunk.append(line)
                    para_chunk_size += line_size + 1
            
            # Add remaining para_chunk
            if para_chunk:
                chunk_text_combined = '\n'.join(para_chunk)
                if chunk_text_combined.strip():
                    # If still too large, split by sentences
                    if len(chunk_text_combined) > max_chunk_size:
                        sentences = re.split(r'(?<=[.!?])\s+', chunk_text_combined)
                        sent_chunk = []
                        sent_chunk_size = 0
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            sent_size = len(sentence)
                            
                            if sent_chunk_size + sent_size + 1 > max_chunk_size and sent_chunk:
                                chunk_text_final = ' '.join(sent_chunk)
                                if chunk_text_final.strip():
                                    chunks.append((chunk_text_final.strip(), section_header))
                                sent_chunk = [sentence]
                                sent_chunk_size = sent_size
                            else:
                                sent_chunk.append(sentence)
                                sent_chunk_size += sent_size + 1
                        
                        if sent_chunk:
                            chunk_text_final = ' '.join(sent_chunk)
                            if chunk_text_final.strip():
                                chunks.append((chunk_text_final.strip(), section_header))
                    else:
                        chunks.append((chunk_text_combined.strip(), section_header))
        
        else:
            # Paragraph fits - check if we can add it to current chunk
            if current_size + para_size + 2 <= max_chunk_size:  # +2 for \n\n
                current_chunk.append(para)
                current_size += para_size + 2
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text_combined = '\n\n'.join(current_chunk)
                    if chunk_text_combined.strip():
                        chunks.append((chunk_text_combined.strip(), section_header))
                current_chunk = [para]
                current_size = para_size
    
    # Add remaining chunk
    if current_chunk:
        chunk_text_combined = '\n\n'.join(current_chunk)
        if chunk_text_combined.strip():
            chunks.append((chunk_text_combined.strip(), section_header))
    
    # If no paragraphs were found (document has no paragraph breaks), split directly
    if not chunks and len(chunk_text) > max_chunk_size:
        # Direct character-based split as last resort
        for i in range(0, len(chunk_text), max_chunk_size):
            sub_chunk = chunk_text[i:i + max_chunk_size].strip()
            if sub_chunk:
                chunks.append((sub_chunk, section_header))
    
    return chunks if chunks else [(chunk_text, section_header)]


def merge_small_chunks(
    chunks: List[Tuple[str, Optional[str]]],
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000
) -> List[Tuple[str, Optional[str]]]:
    """
    Merges adjacent small chunks together until they reach min_chunk_size.
    
    Preserves section boundaries - only merges chunks with the same section header.
    
    Args:
        chunks: List of (chunk_text, section_header) tuples
        min_chunk_size: Minimum desired chunk size
        max_chunk_size: Maximum chunk size (won't merge beyond this)
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged_chunks = []
    current_chunk = []
    current_section = None
    current_size = 0
    
    for chunk_text, section_header in chunks:
        chunk_size = len(chunk_text)
        
        # If chunk is already large enough, add as-is
        if chunk_size >= min_chunk_size:
            # Save any accumulated small chunks first
            if current_chunk:
                merged_text = '\n\n'.join(current_chunk)
                if merged_text.strip():
                    merged_chunks.append((merged_text.strip(), current_section))
                current_chunk = []
                current_size = 0
                current_section = None
            merged_chunks.append((chunk_text, section_header))
        
        # If chunk is too small, try to merge with adjacent chunks
        else:
            # Can only merge if same section (or both None)
            can_merge = (
                section_header == current_section and
                current_size + chunk_size + 2 <= max_chunk_size  # +2 for separator
            )
            
            if can_merge and current_chunk:
                current_chunk.append(chunk_text)
                current_size += chunk_size + 2
            else:
                # Save previous accumulated chunks if any
                if current_chunk:
                    merged_text = '\n\n'.join(current_chunk)
                    if merged_text.strip():
                        merged_chunks.append((merged_text.strip(), current_section))
                
                # Start new accumulation
                if chunk_size < min_chunk_size:
                    current_chunk = [chunk_text]
                    current_size = chunk_size
                    current_section = section_header
                else:
                    merged_chunks.append((chunk_text, section_header))
                    current_chunk = []
                    current_size = 0
                    current_section = None
    
    # Add any remaining accumulated chunks
    if current_chunk:
        merged_text = '\n\n'.join(current_chunk)
        if merged_text.strip():
            merged_chunks.append((merged_text.strip(), current_section))
    
    return merged_chunks


def split_by_major_sections(
    text: str, 
    major_pattern: re.Pattern,
    max_chunk_size: int = 5000
) -> List[Tuple[str, Optional[str]]]:
    """
    Splits text by MAJOR section headers only (e.g., "Section 4", "Section 13").
    
    This creates one chunk per major section, preserving the complete context
    of each section (including all its subsections like (a), (i), etc.).
    
    Large sections that exceed max_chunk_size will be split further while
    preserving the section context.
    
    Args:
        text: Input text to split
        major_pattern: Compiled regex pattern for major section headers only
        max_chunk_size: Maximum size for a single chunk (sections larger than this will be split)
        
    Returns:
        List of tuples: (chunk_text, section_header)
        section_header is None if no section found for that chunk
    """
    chunks = []
    matches = list(major_pattern.finditer(text))
    
    if not matches:
        # No major section headers found - return entire text as single chunk
        # But split if it's too large
        initial_chunk = (text.strip(), None)
        if len(text.strip()) > max_chunk_size:
            return split_oversized_chunk(text.strip(), None, max_chunk_size)
        return [initial_chunk]
    
    # Process each major section
    for i, match in enumerate(matches):
        section_start = match.start()
        section_text = match.group(0).strip()
        
        # Determine chunk boundaries
        if i == 0:
            # First chunk: everything before first major section (preamble, etc.)
            pre_section = text[:section_start].strip()
            if pre_section:
                if len(pre_section) > max_chunk_size:
                    chunks.extend(split_oversized_chunk(pre_section, None, max_chunk_size))
                else:
                    chunks.append((pre_section, None))
            
            # Get content from this section until next major section (or end)
            if len(matches) > 1:
                chunk_end = matches[i + 1].start()
                chunk_content = text[section_start:chunk_end]
            else:
                chunk_content = text[section_start:]
            
            # Extract and clean section header
            section_header = extract_section_header(section_text)
            chunk_content = chunk_content.strip()
            
            # Split if section is too large
            if len(chunk_content) > max_chunk_size:
                chunks.extend(split_oversized_chunk(chunk_content, section_header, max_chunk_size))
            else:
                chunks.append((chunk_content, section_header))
        
        else:
            # Subsequent chunks: from this major section to next (or end)
            if i < len(matches) - 1:
                chunk_end = matches[i + 1].start()
                chunk_content = text[section_start:chunk_end]
            else:
                chunk_content = text[section_start:]
            
            section_header = extract_section_header(section_text)
            chunk_content = chunk_content.strip()
            
            # Split if section is too large
            if len(chunk_content) > max_chunk_size:
                chunks.extend(split_oversized_chunk(chunk_content, section_header, max_chunk_size))
            else:
                chunks.append((chunk_content, section_header))
    
    # Filter out empty chunks
    chunks = [(chunk, section) for chunk, section in chunks if chunk]
    
    return chunks


def extract_section_header(section_match_text: str) -> str:
    """
    Extracts a clean major section header string from regex match text.
    
    Args:
        section_match_text: Raw text matched by major section pattern
        
    Returns:
        Cleaned section header (e.g., "Section 13" or "PART I")
    """
    # Clean up the matched text
    header = section_match_text.strip()
    
    # Remove trailing colons/periods if they're the last character
    header = re.sub(r'[\.:]$', '', header)
    
    # Normalize whitespace (multiple spaces to single space)
    header = re.sub(r'\s+', ' ', header)
    
    return header.strip()


def split_by_paragraphs(
    text: str, 
    max_chunk_size: int = 2000
) -> List[Tuple[str, Optional[str]]]:
    """
    Fallback strategy: Split text by paragraphs (double newlines).
    
    If paragraphs exceed max_chunk_size or if document has no paragraph breaks,
    splits further using split_oversized_chunk.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum size for a single chunk
        
    Returns:
        List of tuples: (chunk_text, None) - section is always None for paragraphs
    """
    # Split by double newlines
    paragraphs = re.split(r'\n\n+', text)
    
    # Filter out empty paragraphs and strip whitespace
    initial_chunks = [(para.strip(), None) for para in paragraphs if para.strip()]
    
    # If no paragraphs found (document has no paragraph breaks), treat entire text as one chunk
    if not initial_chunks:
        initial_chunks = [(text.strip(), None)]
    
    # Split oversized chunks and merge small ones
    processed_chunks = []
    for chunk_text, section_header in initial_chunks:
        if len(chunk_text) > max_chunk_size:
            # Split oversized chunks
            processed_chunks.extend(split_oversized_chunk(chunk_text, section_header, max_chunk_size))
        else:
            processed_chunks.append((chunk_text, section_header))
    
    return processed_chunks


def apply_hybrid_chunking(
    text: str, 
    major_pattern: re.Pattern,
    min_sections_threshold: int = 2,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    section_max_chunk_size: int = 5000
) -> List[Tuple[str, Optional[str]]]:
    """
    Hybrid chunking strategy: Try major section splitting first, fallback to paragraphs.
    
    Strategy:
    1. Try splitting by major section boundaries (Section 4, Section 13, etc.)
    2. If few or no major sections found, fall back to paragraph splitting
    3. Enforce max chunk size limits (stricter for paragraphs, more lenient for sections)
    4. Merge small chunks together to reach minimum size
    
    This ensures legal documents (like PMLA) are chunked by complete sections,
    while non-legal documents (penalty orders, IT files) use paragraph splitting.
    
    Args:
        text: Input text to chunk
        major_pattern: Compiled regex pattern for major section headers only
        min_sections_threshold: Minimum number of major sections required to use section splitting
        min_chunk_size: Minimum desired chunk size (chunks smaller than this will be merged)
        max_chunk_size: Maximum chunk size for paragraph-based chunks
        section_max_chunk_size: Maximum chunk size for section-based chunks (more lenient)
        
    Returns:
        List of tuples: (chunk_text, section_header) with normalized chunk sizes
    """
    # Try major section-aware splitting first
    section_chunks = split_by_major_sections(text, major_pattern, max_chunk_size=section_max_chunk_size)
    
    # Check if we found meaningful section structure
    sections_found = sum(1 for _, section in section_chunks if section is not None)
    
    if sections_found >= min_sections_threshold:
        logger.info(f"Using major section splitting: found {sections_found} major sections")
        chunks = section_chunks
        # For section chunks, use a more lenient max size but still enforce limits
        # Re-split any chunks that exceed the paragraph max_chunk_size
        normalized_chunks = []
        for chunk_text, section_header in chunks:
            if len(chunk_text) > max_chunk_size:
                normalized_chunks.extend(split_oversized_chunk(chunk_text, section_header, max_chunk_size))
            else:
                normalized_chunks.append((chunk_text, section_header))
        chunks = normalized_chunks
    else:
        # Fallback to paragraph splitting for documents without major section structure
        logger.info(f"Falling back to paragraph splitting: only {sections_found} major sections found")
        chunks = split_by_paragraphs(text, max_chunk_size=max_chunk_size)
    
    # Merge small chunks to reach minimum size
    chunks = merge_small_chunks(chunks, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
    
    # Final pass: split any chunks that still exceed max_chunk_size (safety check)
    final_chunks = []
    for chunk_text, section_header in chunks:
        if len(chunk_text) > max_chunk_size:
            final_chunks.extend(split_oversized_chunk(chunk_text, section_header, max_chunk_size))
        else:
            final_chunks.append((chunk_text, section_header))
    
    return final_chunks


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def load_documents_from_paths(file_paths: List[str]) -> List[Document]:
    """
    Loads documents from a list of file paths.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            loader = TextLoader(
                file_path, 
                encoding='utf-8',
                autodetect_encoding=True
            )
            doc = loader.load()[0]  # TextLoader returns list with one doc
            
            # Add filename to metadata
            doc.metadata['source_file'] = os.path.basename(file_path)
            doc.metadata['source_path'] = file_path
            
            documents.append(doc)
            logger.info(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return documents


def process_documents(
    file_paths: Optional[List[str]] = None,
    data_directory: str = "./data"
) -> List[Document]:
    """
    Main function to process documents with hybrid chunking strategy.
    
    If file_paths is None, automatically discovers all .txt files in data_directory.
    
    Args:
        file_paths: Optional list of specific file paths to process
        data_directory: Directory to search for documents if file_paths is None
        
    Returns:
        List of chunked Document objects with metadata
    """
    logger.info("=" * 60)
    logger.info("Section-Aware Document Ingestion - Hybrid Chunking")
    logger.info("=" * 60)
    
    # Get file paths
    if file_paths is None:
        logger.info(f"Discovering documents in: {data_directory}")
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory '{data_directory}' not found.")
        
        # Find all .txt files recursively
        file_paths = []
        for root, dirs, files in os.walk(data_directory):
            for file in files:
                if file.lower().endswith('.txt'):
                    file_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(file_paths)} .txt files")
    else:
        logger.info(f"Processing {len(file_paths)} specified files")
    
    if not file_paths:
        raise ValueError("No documents found to process")
    
    # Load documents
    documents = load_documents_from_paths(file_paths)
    
    if not documents:
        raise ValueError("No documents could be loaded")
    
    # Initialize major section pattern (only matches top-level boundaries)
    major_pattern = get_major_section_pattern()
    
    # Process each document with hybrid chunking
    all_chunks = []
    
    for doc in documents:
        text = doc.page_content
        source_file = doc.metadata['source_file']
        
        logger.info(f"Processing document: {source_file}")
        logger.info(f"  Original length: {len(text)} characters")
        
        # Apply hybrid chunking (major sections first, fallback to paragraphs)
        chunks_data = apply_hybrid_chunking(text, major_pattern)
        
        logger.info(f"  Created {len(chunks_data)} chunks")
        
        # Create Document objects for each chunk
        for i, (chunk_text, section_header) in enumerate(chunks_data):
            chunk_metadata = {
                'source_file': source_file,
                'source_path': doc.metadata.get('source_path', ''),
                'chunk_id': i,
                'chunk_length': len(chunk_text),
            }
            
            # Add section information if available
            if section_header:
                chunk_metadata['section'] = section_header
            
            # Categorize document type based on filename
            filename_lower = source_file.lower()
            if 'rbi' in filename_lower:
                chunk_metadata['doc_type'] = 'RBI'
            elif 'sebi' in filename_lower:
                chunk_metadata['doc_type'] = 'SEBI'
            elif 'fiu' in filename_lower:
                chunk_metadata['doc_type'] = 'FIU'
            elif 'faq' in filename_lower:
                chunk_metadata['doc_type'] = 'FAQ'
            elif 'incometax' in filename_lower or 'tax' in filename_lower:
                chunk_metadata['doc_type'] = 'Income Tax'
            else:
                chunk_metadata['doc_type'] = 'General'
            
            chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            all_chunks.append(chunk_doc)
    
    logger.info(f"=" * 60)
    logger.info(f"Total chunks created: {len(all_chunks)}")
    logger.info(f"=" * 60)
    
    return all_chunks


# ============================================================================
# PLACEHOLDER FUNCTIONS FOR EMBEDDING AND VECTOR DB
# ============================================================================

def embed_chunks(chunks: List[Document], model_name: str = "intfloat/e5-large-v2") -> HuggingFaceEmbeddings:
    """
    Initialize the embedding model for chunk embedding.
    
    Note: This function initializes the model. The actual embedding is done by ChromaDB
    when adding documents to the vector database.
    
    Args:
        chunks: List of Document objects to embed (used for logging)
        model_name: Name of the embedding model to use
        
    Returns:
        Initialized HuggingFaceEmbeddings model
    """
    # Check if CUDA is available for GPU acceleration
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 32 if device == 'cuda' else 8
        
        if device == 'cuda':
            logger.info(f"Initializing {model_name} embeddings with GPU acceleration (CUDA)...")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            logger.info(f"Batch size: {batch_size} (GPU optimized)")
        else:
            logger.info(f"Initializing {model_name} embeddings on CPU...")
            logger.info(f"Batch size: {batch_size}")
    except ImportError:
        logger.warning("PyTorch not found, defaulting to CPU")
        device = 'cpu'
        batch_size = 8
    
    logger.info(f"Downloading model (first time only, ~1.3GB)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': batch_size
        }
    )
    
    logger.info(f"Embedding model '{model_name}' loaded successfully")
    logger.info(f"Ready to embed {len(chunks)} chunks")
    
    return embeddings


def add_to_vector_db(
    chunks: List[Document], 
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = "db_e5_section_aware",
    collection_name: str = "financial_regulations_section_aware"
) -> Chroma:
    """
    Create vector database and add chunks with embeddings.
    
    This function:
    1. Creates a ChromaDB vector database
    2. Adds chunks with their embeddings and metadata (embeddings generated automatically)
    3. Persists the database to disk
    
    Args:
        chunks: List of Document objects to add
        embeddings: HuggingFaceEmbeddings model for generating embeddings
        persist_directory: Directory to persist the vector database
        collection_name: Name of the collection/table in the vector DB
        
    Returns:
        Chroma vector database instance
    """
    logger.info(f"Creating vector database with E5 embeddings...")
    logger.info(f"  Persist directory: {persist_directory}")
    logger.info(f"  Collection name: {collection_name}")
    logger.info(f"  Adding {len(chunks)} chunks...")
    
    # Create vector database (Chroma handles embedding generation and persistence automatically)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Note: Chroma 0.4.x+ automatically persists, no need to call persist()
    
    logger.info(f"✓ Vector database saved to '{persist_directory}'")
    logger.info(f"✓ Total chunks indexed: {len(chunks)}")
    
    return vectordb


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    try:
        # Process documents (auto-discovers .txt files in ./data)
        chunks = process_documents()
        
        # Initialize embedding model
        embeddings = embed_chunks(chunks)
        
        # Create vector database and add chunks (embeddings generated automatically)
        vectordb = add_to_vector_db(chunks, embeddings)
        
        logger.info("\n" + "=" * 60)
        logger.info("Ingestion completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Vector database: db_e5_section_aware/")
        logger.info(f"Collection: financial_regulations_section_aware")
        logger.info("\nThe knowledge base is ready for use in your RAG pipeline.")
        
        return vectordb
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
