"""
RAG Pipeline for Financial Awareness Chatbot
Handles document retrieval and LLM response generation
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import reranker (optional dependency)
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding FlagReranker not available. Reranking will be disabled. Install with: pip install FlagEmbedding")

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for financial queries
    """
    
    def __init__(self, 
                 db_path: str = None,
                 model_name: str = "llama3.2:3b",
                 k: int = 7,
                 use_reranker: bool = True,
                 reranker_model: str = "BAAI/bge-reranker-base",
                 rerank_k: int = None,
                 reranker_batch_size: int = 8,
                 enable_query_transformation: bool = True):
        """
        Initialize RAG pipeline
        
        Args:
            db_path: Path to ChromaDB database (default: auto-detect)
            model_name: Ollama model name to use
            k: Number of documents to return after reranking (default: 7, adaptive 3-10 based on query)
            use_reranker: Whether to use reranking (default: True)
            reranker_model: Reranker model name (default: BAAI/bge-reranker-base)
            rerank_k: Number of documents to retrieve before reranking. 
                     If None and reranking is enabled, defaults to 50 for better accuracy.
            reranker_batch_size: Batch size for reranking (default: 8, optimized for 6GB VRAM)
            enable_query_transformation: Whether to transform follow-up queries into standalone queries (default: True)
        """
        # Auto-detect database path (works from both root and backend directory)
        if db_path is None:
            # Get script directory to build absolute paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            # Try multiple possible paths, verify it's a valid ChromaDB directory
            possible_paths = [
                os.path.join(project_root, "db_e5_section_aware"),  # Absolute path from project root
                "db_e5_section_aware",  # If running from root
                "../db_e5_section_aware",  # If running from backend
            ]
            
            db_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                # Check if path exists and contains chroma.sqlite3 (ChromaDB signature file)
                if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, "chroma.sqlite3")):
                    db_path = abs_path
                    logger.info(f"Auto-detected database path: {db_path}")
                    break
            
            if db_path is None:
                # Fallback to project root
                db_path = os.path.join(project_root, "db_e5_section_aware")
                logger.warning(f"Database path not auto-detected, using: {db_path}")
        
        self.db_path = db_path
        self.model_name = model_name
        self.k = k
        self.use_reranker = use_reranker and RERANKER_AVAILABLE
        self.reranker_model_name = reranker_model
        self.reranker_batch_size = reranker_batch_size
        self.enable_query_transformation = enable_query_transformation
        
        # Initialize embeddings with GPU detection
        logger.info("Loading E5-large-v2 embedding model...")
        # Check if CUDA is available for GPU acceleration
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = 32 if device == 'cuda' else 8  # Larger batch for GPU
            
            if device == 'cuda':
                logger.info(f"Initializing embeddings with GPU acceleration (CUDA)...")
                logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
                logger.info(f"Batch size: {batch_size} (GPU optimized)")
            else:
                logger.info(f"Initializing embeddings on CPU...")
                logger.info(f"Batch size: {batch_size}")
        except ImportError:
            logger.warning("PyTorch not found, defaulting to CPU")
            device = 'cpu'
            batch_size = 8
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': batch_size
            }
        )
        
        # Load vector database
        logger.info(f"Loading vector database from {db_path}...")
        self.vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name="financial_regulations_section_aware"
        )
        
        # Initialize reranker (lazy loading - only when needed)
        self.reranker = None
        if self.use_reranker:
            logger.info(f"Reranker enabled (will load {reranker_model} on first use)")
            # Default to k=50 for better accuracy when reranking is enabled
            self.rerank_k = rerank_k if rerank_k is not None else 50
            logger.info(f"Will retrieve {self.rerank_k} documents, then rerank to top {k}")
            logger.info(f"Reranker batch size: {reranker_batch_size} (optimized for 6GB VRAM)")
        else:
            self.rerank_k = None
            if use_reranker and not RERANKER_AVAILABLE:
                logger.warning("Reranking requested but not available. Install FlagEmbedding to enable: pip install FlagEmbedding")
        
        logger.info("RAG Pipeline initialized successfully")
    
    
    def _load_reranker(self):
        """Lazy load reranker model with fp16 for efficient VRAM usage"""
        if self.reranker is None and self.use_reranker:
            logger.info(f"Loading reranker model: {self.reranker_model_name}...")
            
            # Check GPU availability for reranker
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"Reranker will use GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("Reranker will use CPU (GPU not available)")
            except ImportError:
                logger.info("PyTorch not available, reranker will use default device")
            
            logger.info("Using fp16 precision to optimize for 6GB VRAM...")
            logger.info("Downloading model (first time only, ~420MB)...")
            try:
                # Use fp16=True to cut VRAM usage in half with minimal accuracy loss
                # FlagReranker automatically uses GPU if available
                self.reranker = FlagReranker(self.reranker_model_name, use_fp16=True)
                logger.info("Reranker loaded successfully with fp16 precision")
                
                # Verify GPU usage
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"Reranker device: GPU ({torch.cuda.get_device_name(0)})")
                except:
                    pass
            except Exception as e:
                logger.error(f"Failed to load reranker: {str(e)}")
                logger.error("Make sure FlagEmbedding is installed: pip install FlagEmbedding")
                self.use_reranker = False
                self.reranker = None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using FlagReranker with fp16 precision.
        
        Uses batch processing to efficiently handle large document sets within VRAM limits.
        
        Args:
            query: User query
            documents: List of retrieved documents with 'content', 'metadata', and 'score'
            top_k: Number of top documents to return after reranking
            
        Returns:
            List of reranked documents (top_k), sorted by relevance score (highest first)
        """
        if not self.use_reranker or not documents:
            return documents[:top_k]
        
        # Load reranker if not already loaded
        self._load_reranker()
        
        if not self.reranker:
            logger.warning("Reranker not available, returning original documents")
            return documents[:top_k]
        
        try:
            # Prepare query-document pairs for reranking
            # FlagReranker expects pairs in the format [query, document_content]
            pairs = [[query, doc['content']] for doc in documents]
            
            # Get reranking scores with batching
            # compute_score handles batching automatically based on the batch_size parameter
            logger.info(f"Reranking {len(documents)} documents in batches of {self.reranker_batch_size}...")
            rerank_scores = self.reranker.compute_score(pairs, batch_size=self.reranker_batch_size)
            
            # Handle both single score and list of scores
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
            
            # Combine scores with documents
            scored_docs = [
                {
                    **doc,
                    'rerank_score': float(score),
                    'original_score': doc.get('score', 0.0)
                }
                for doc, score in zip(documents, rerank_scores)
            ]
            
            # Sort by rerank score (descending) and return top_k
            scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            reranked = scored_docs[:top_k]
            
            logger.info(f"Reranking complete. Top {len(reranked)} documents selected.")
            for i, doc in enumerate(reranked[:3], 1):
                logger.info(f"  Rank {i}: rerank_score={doc['rerank_score']:.4f}, "
                          f"original_score={doc['original_score']:.4f}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.warning("Falling back to original retrieval results")
            return documents[:top_k]
    
    def _determine_adaptive_k(self, query: str, base_k: int = None) -> int:
        """
        Determine optimal k value based on query complexity and characteristics.
        
        Strategy:
        - Simple factual queries (what, who, when): 3-5 docs
        - Complex/comparison queries (compare, difference, explain): 7-10 docs
        - Procedural queries (how, steps): 5-7 docs
        - Multi-part questions: 7-10 docs
        
        Args:
            query: User query
            base_k: Base k value (default: self.k)
            
        Returns:
            Adaptive k value
        """
        if base_k is None:
            base_k = self.k
        
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Multi-part indicators (higher k needed)
        multi_part_indicators = ['compare', 'difference', 'versus', 'vs', 'both', 'and', 'also', 'additionally']
        complex_indicators = ['explain', 'describe', 'elaborate', 'details', 'comprehensive']
        procedural_indicators = ['how', 'steps', 'process', 'procedure', 'method']
        
        # Simple factual indicators (lower k sufficient)
        simple_indicators = ['what is', 'who is', 'when', 'where', 'which']
        
        # Count indicators
        multi_part_count = sum(1 for indicator in multi_part_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        procedural_count = sum(1 for indicator in procedural_indicators if indicator in query_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        
        # Determine adaptive k
        if multi_part_count >= 1 or complex_count >= 2:
            # Complex/comparison queries: 7-10 docs
            adaptive_k = max(7, min(10, base_k + 3))
            logger.info(f"Query complexity: Complex/Multi-part → k={adaptive_k}")
        elif procedural_count >= 1:
            # Procedural queries: 5-7 docs
            adaptive_k = max(5, min(7, base_k + 1))
            logger.info(f"Query complexity: Procedural → k={adaptive_k}")
        elif word_count > 15:
            # Long queries (likely complex): 6-8 docs
            adaptive_k = max(6, min(8, base_k + 2))
            logger.info(f"Query complexity: Long query ({word_count} words) → k={adaptive_k}")
        elif simple_count >= 1 and word_count < 10:
            # Simple factual queries: 3-5 docs (slight reduction from base)
            adaptive_k = max(3, min(5, base_k))
            logger.info(f"Query complexity: Simple factual → k={adaptive_k}")
        else:
            # Default: use base_k (typically 5)
            adaptive_k = base_k
            logger.info(f"Query complexity: Default → k={adaptive_k}")
        
        return adaptive_k
    
    def retrieve_documents(
        self, 
        query: str, 
        k: int = None, 
        verbose_logging: bool = True,
        use_reranking: bool = None,
        adaptive_k: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query, optionally with reranking.
        
        Args:
            query: User query
            k: Number of documents to retrieve (default: self.k, or adaptive if enabled)
            verbose_logging: If True, log detailed chunk information for debugging
            use_reranking: Override default reranking behavior (default: self.use_reranker)
            adaptive_k: If True, adjust k based on query complexity (default: True)
            
        Returns:
            List of retrieved documents with metadata (and rerank scores if reranking used)
        """
        if k is None:
            if adaptive_k:
                # Use adaptive k based on query complexity
                k = self._determine_adaptive_k(query, base_k=self.k)
            else:
                k = self.k
        elif adaptive_k:
            # User provided k, but we can still log complexity analysis
            logger.info(f"Using user-specified k={k} (adaptive analysis disabled)")
        
        if use_reranking is None:
            use_reranking = self.use_reranker
        
        # Determine how many documents to retrieve initially
        # If reranking, retrieve more documents initially, then rerank to top k
        if use_reranking and self.use_reranker and self.rerank_k:
            initial_k = self.rerank_k
        else:
            initial_k = k
        
        logger.info(f"="*80)
        logger.info(f"QUERY: '{query}'")
        logger.info(f"Retrieving {initial_k} documents (will return top {k})...")
        if use_reranking and self.use_reranker:
            logger.info(f"Reranking enabled: will rerank {initial_k} → {k} documents")
        logger.info(f"="*80)
        
        # E5 models require "query: " prefix for queries
        # Documents should be embedded without prefix (they're already stored)
        query_text = f"query: {query}" if not query.startswith("query:") else query
        
        # Perform similarity search
        docs = self.vectordb.similarity_search_with_score(query_text, k=initial_k)
        
        # Format results
        retrieved_docs = []
        for i, (doc, score) in enumerate(docs, 1):
            chunk_info = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            retrieved_docs.append(chunk_info)
        
        # Apply reranking if enabled and we have more documents than needed
        if use_reranking and self.use_reranker and len(retrieved_docs) > k:
            retrieved_docs = self.rerank_documents(query, retrieved_docs, top_k=k)
            # Update scores to use rerank scores for logging
            for doc in retrieved_docs:
                doc['score'] = doc.get('rerank_score', doc['score'])
        
        # Log detailed information
        if verbose_logging:
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"CHUNK {i}/{len(retrieved_docs)}")
                logger.info(f"{'='*80}")
                logger.info(f"SIMILARITY SCORE: {doc['score']:.4f}")
                if 'rerank_score' in doc:
                    logger.info(f"RERANK SCORE: {doc['rerank_score']:.4f}")
                    logger.info(f"ORIGINAL SCORE: {doc.get('original_score', 'N/A')}")
                source_file = doc['metadata'].get('source_file') or doc['metadata'].get('filename', 'Unknown')
                logger.info(f"SOURCE FILE: {source_file}")
                logger.info(f"DOCUMENT TYPE: {doc['metadata'].get('doc_type', 'Unknown')}")
                logger.info(f"CHUNK ID: {doc['metadata'].get('chunk_id', 'N/A')}")
                logger.info(f"CHUNK LENGTH: {len(doc['content'])} characters")
                logger.info(f"\nFULL CHUNK CONTENT:")
                logger.info(f"{'-'*80}")
                logger.info(f"{doc['content']}")
                logger.info(f"{'-'*80}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RETRIEVAL SUMMARY: Retrieved {len(retrieved_docs)} chunks")
        if use_reranking and self.use_reranker and initial_k > k:
            logger.info(f"Reranking: {initial_k} → {len(retrieved_docs)} documents")
        logger.info(f"{'='*80}\n")
        
        return retrieved_docs
    
    def transform_query_with_history(
        self, 
        latest_user_query: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Transform a follow-up query into a standalone query using conversation history.
        
        This function rewrites queries like "What are the penalties for that?" into
        "What are the penalties for PMLA non-compliance?" by using context from
        previous conversation turns.
        
        Args:
            latest_user_query: The most recent user query (may contain references like "that", "it", etc.)
            chat_history: List of previous messages in format [{"role": "user", "content": "..."}, ...]
            
        Returns:
            Standalone query string that can be used for retrieval without context
        """
        # Handle empty history (first turn) - return query as-is
        if not chat_history or len(chat_history) == 0:
            print(f"\n🔍 QUERY TRANSFORMATION: No history (first turn)")
            print(f"   Using original query as-is: '{latest_user_query}'\n")
            logger.info("No conversation history available, using original query as-is")
            return latest_user_query.strip()
        
        # Format conversation history for the prompt
        # Include last 6 messages (3 exchanges) to provide context without overwhelming
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        formatted_history = ""
        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '').strip()
            if role == 'user':
                formatted_history += f"User: {content}\n"
            elif role == 'assistant':
                # Truncate very long assistant responses to keep prompt manageable
                # But keep enough context (1000 chars) to resolve pronouns like "its"
                content_truncated = content[:1000] + "..." if len(content) > 1000 else content
                formatted_history += f"Assistant: {content_truncated}\n"
        
        # Construct the transformation prompt
        transformation_prompt = f"""Given the following conversation history and a subsequent user query, rephrase the user query to be a standalone question that can be understood without the chat history.

IMPORTANT INSTRUCTIONS:
- Replace pronouns (it, its, that, this, they, them) with the specific entities or topics they refer to from the conversation history.
- Focus on the MOST RECENT topics discussed (especially from the last assistant response).
- Include specifically referenced entities, acts, regulations, procedures, or concepts from the history.
- If the query is already standalone and contains no pronouns or ambiguous references, return it unchanged.
- DO NOT answer the question. JUST rewrite the query to be clear and standalone.
- Your response should ONLY be the rewritten query, nothing else.

Chat History:
{formatted_history}

Latest User Query: {latest_user_query}

Standalone Query:"""

        print(f"\n🔍 QUERY TRANSFORMATION STARTING")
        print(f"   Original Query: '{latest_user_query}'")
        print(f"   History Messages: {len(recent_history)}")
        logger.info(f"Transforming query with {len(recent_history)} history messages")
        logger.debug(f"Original query: {latest_user_query}")
        
        try:
            print(f"   Calling LLM for transformation...")
            # Use Ollama to transform the query
            # Lower temperature for more consistent rewrites
            response = ollama.generate(
                model=self.model_name,
                prompt=transformation_prompt,
                options={
                    'temperature': 0.2,  # Slightly higher than answer generation for creativity in rewrites
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 150,  # Limit response length (queries should be concise)
                }
            )
            
            transformed_query = response['response'].strip()
            
            # Clean up common LLM artifacts
            # Remove quotes if the model wrapped the query in them
            if transformed_query.startswith('"') and transformed_query.endswith('"'):
                transformed_query = transformed_query[1:-1].strip()
            if transformed_query.startswith("'") and transformed_query.endswith("'"):
                transformed_query = transformed_query[1:-1].strip()
            
            # Validate that we got a reasonable response
            if not transformed_query or len(transformed_query) < 3:
                print(f"   ⚠️  WARNING: Transformed query too short, using original")
                logger.warning("Transformed query too short, using original query")
                return latest_user_query.strip()
            
            # Check if transformation actually changed the query significantly
            # If very similar, log it but still use the transformed version
            if transformed_query.lower() == latest_user_query.lower():
                print(f"   ℹ️  Query unchanged (already standalone)")
                print(f"   Result: '{transformed_query}'")
                logger.info("Query transformation returned original query (already standalone)")
            else:
                print(f"   ✅ TRANSFORMATION SUCCESS!")
                print(f"   Original: '{latest_user_query}'")
                print(f"   Transformed: '{transformed_query}'")
                logger.info(f"Query transformed: '{latest_user_query}' → '{transformed_query}'")
            
            print(f"🔍 QUERY TRANSFORMATION COMPLETE\n")
            return transformed_query
            
        except Exception as e:
            print(f"   ❌ ERROR during transformation: {str(e)}")
            print(f"   ⚠️  Falling back to original query")
            logger.error(f"Error transforming query: {str(e)}")
            logger.warning("Falling back to original query")
            return latest_user_query.strip()
    
    def generate_prompt(self, query: str, context_docs: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate prompt for LLM using retrieved documents and conversation history
        
        Args:
            query: User query
            context_docs: Retrieved documents
            conversation_history: List of previous messages in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            Formatted prompt string
        """
        # Combine context from retrieved documents with detailed source information
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # Support both 'source_file' (new) and 'filename' (legacy) metadata fields
            source_file = (doc['metadata'].get('source_file') or doc['metadata'].get('filename', 'Unknown')).replace('.txt', '')
            doc_type = doc['metadata'].get('doc_type', 'Official Document')
            # Add section info if available (from section-aware chunking)
            section = doc['metadata'].get('section')
            if section:
                source_file += f" | Section: {section}"
            
            # Format source citation for internal reference
            source_citation = f"Document: {source_file} | Type: {doc_type}"
            context_parts.append(f"[Source {i}: {source_citation}]\n{doc['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Add conversation history context if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Include last 5 exchanges to maintain context without overwhelming the prompt
            recent_history = conversation_history[-10:]  # Last 10 messages (5 exchanges)
            conversation_context = "\n\n### Previous Conversation Context:\n"
            for msg in recent_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    conversation_context += f"User: {content}\n"
                elif role == 'assistant':
                    conversation_context += f"Assistant: {content}\n"
            conversation_context += "\n---\n\n"
            conversation_context += "**IMPORTANT**: Use the conversation history above to understand context and provide coherent follow-up responses. Reference previous questions or answers when relevant.\n\n"
        
        # Create professional, government-grade prompt
        prompt = f"""You are FIU-Sahayak, an official AI assistant for the Financial Intelligence Unit of India (FIU-IND).

Your purpose is to provide accurate information about Indian financial regulations, PMLA compliance, money laundering prevention, and FIU-IND procedures.

### CRITICAL OPERATIONAL RULES:

**1. STRICT GROUNDING - NO HALLUCINATIONS:**
- You MUST answer using ONLY information from the Retrieved Context below
- If the answer is NOT in the context, respond: "I don't have sufficient information in my knowledge base to answer that question accurately. For authoritative guidance, please contact FIU-IND directly or visit the official FIU-IND website."
- NEVER use your general knowledge or make assumptions
- NEVER make up facts, statistics, dates, or procedures

**2. NO LEGAL OR FINANCIAL ADVICE:**
- You provide INFORMATION ONLY, not advice
- Never say "you should" or "I recommend" for legal/financial matters
- Instead say "According to [regulation], entities are required to..." or "The Act mandates that..."
- Always include: "This is informational guidance only. For specific legal or financial advice, please consult a qualified professional."

**3. FACTUAL & TERMINOLOGY GUARDRAILS (CRITICAL):**
- **OFFICIAL ENTITIES:** You must ONLY use the following official names: **Financial Intelligence Unit-India (FIU-IND)**, **Enforcement Directorate (ED)**, and **Reserve Bank of India (RBI)**.
- **DO NOT** invent or mention any other entity (e.g., "FRCA").
- **OFFICIAL LEGAL SECTIONS (FOR PENALTIES):**
    - The offense of money laundering is under **Section 4** of the PMLA.
    - Non-compliance by reporting entities (banks, etc.) is under **Section 13** of the PMLA.
- **NEVER** cite these sections (or any others) unless they are EXPLICITLY supported by the Retrieved Context. The "STRICT GROUNDING" rule (Rule 1) is your highest priority.

**4. CONTEXT-AWARE DETAILS (CONDITIONAL):**
- **IF, AND ONLY IF,** the "Retrieved Context" provides them, you should prioritize mentioning key operational details to provide a complete answer.
- These include: **STR (Suspicious Transaction Report)**, **CTR (Cash Transaction Report)**, **CBWTR (Cross-Border Wire Transfer Report)**, and **KYC (Know Your Customer)**.
- These also include specifics like thresholds (e.g., "CTRs for cash transactions exceeding ₹10 lakhs"), timelines, or procedures.
- **CRITICAL:** If these details are NOT in the context, DO NOT mention them. Adhere to Rule 1.

**5. SECURITY & PII HANDLING:**
- NEVER ask for personal information (PAN, Aadhaar, account numbers, etc.)
- If a user shares PII, ignore it and respond: "Please do not share personal information. I provide general information only and do not need personal details."
- Remind users: "Do not share sensitive personal or financial information in this chat."

**6. SYNTHESIS OVER CONCATENATION:**
- **CRITICAL**: Create a SINGLE COHERENT NARRATIVE from all sources
- DO NOT list isolated facts from each source separately
- SYNTHESIZE information into a flowing, logical explanation
- Weave facts together naturally - don't jump between disconnected points
- Prioritize information from higher-relevance sources
- Only include information that directly answers the question

**7. DIRECT AUTHORITATIVE TONE - ANSWER THE QUESTION FIRST:**
- ❌ NEVER WRITE: "Answering Your Question:", "Let me explain:", "According to [Source 1: ...]"
- ✅ ANSWER THE DIRECT QUESTION IMMEDIATELY in the first sentence
- ✅ START IMMEDIATELY with the answer: "The Prevention of Money Laundering Act (PMLA), 2002, is an Act of Parliament..."
- ✅ For "What is X?" → Answer what X is directly
- ✅ For "What is the penalty?" → State the penalty amount directly
- ✅ For "How do I report?" → State the reporting method directly
- Use direct, declarative statements as if you ARE the official authority
- NO conversational preambles, NO hedging, NO source citations in text
- **CRITICAL:** Answer the specific question asked FIRST, then provide additional context

**EXAMPLE OF WHAT NOT TO DO:**
"Answering Your Question: What is PMLA?
According to [Source 1: Document: PMLA 2002 | Type: General], PMLA stands for..."

**EXAMPLE OF CORRECT APPROACH (Direct Answer First):**
"The Prevention of Money Laundering Act (PMLA), 2002, is an Act of Parliament that came into force on July 1, 2005..."

**8. ABSOLUTELY NO INLINE CITATIONS:**
- ❌ NEVER: "[Source 1: ...]", "According to [Source X: ...]", "As per [Document Y]"
- ✅ INSTEAD: Simply state the facts directly and authoritatively
- The system will add sources at the end automatically
- Your job is ONLY to synthesize information into clean prose

**9. LIMITATIONS & ESCALATION:**
- Acknowledge when a question is too complex or ambiguous
- Clearly state what you CAN and CANNOT do
- For complex legal interpretation: "This matter may require legal expertise. Please consult FIU-IND or a compliance professional."
- For technical issues (website, forms, submissions): "For technical assistance, please contact FIU-IND support at [contact details]."

**10. RESPONSE STRUCTURE:**
- Opening: **Direct, immediate answer to the question asked** (no preamble, no delay)
- Body: 3-5 paragraphs of comprehensive, synthesized explanation that supports the direct answer
- **CRITICAL:** The first sentence must answer the specific question, then provide context
- Key terms: **Bold** important legislation, concepts, or requirements
- Lists: Use numbered lists ONLY for sequential steps or procedures
- Headers: Use ### only if genuinely organizing distinct major sections
- Closing: Brief disclaimer about consulting professionals if needed

{conversation_context}### Retrieved Context from Official Documents:
{context}

---

### User's Question:
{query}

---

### MANDATORY RESPONSE FORMAT:

**OPENING (First Sentence - CRITICAL):**
- **ANSWER THE DIRECT QUESTION IMMEDIATELY** in the first sentence
- Start DIRECTLY with the answer to what was asked
- Examples:
  - Question: "What is PMLA?" → Answer: "The Prevention of Money Laundering Act (PMLA), 2002, is..."
  - Question: "What is the penalty?" → Answer: "The penalty for [offense] is [amount] as specified in [section]..."
  - Question: "How do I report?" → Answer: "Reporting entities must submit [report type] through [method]..."
- NO preambles. NO "Answering your question". NO "According to [Source]".
- **The first sentence MUST directly answer the question asked**

**BODY (3-4 Paragraphs):**
1. **First paragraph**: Directly answer the question, then provide core definition and purpose
2. **Second paragraph**: Key operational mechanisms (STR, CTR, CBWTR, KYC) - ONLY if mentioned in context, with exact thresholds if provided
3. **Third paragraph**: Compliance requirements and procedures - ONLY if relevant to the question
4. **Fourth paragraph**: Consequences/penalties (ONLY if relevant to the question AND present in context)

**STRUCTURE:**
- Use **bold** for: Act names, key terms (STR, CTR, KYC, CBWTR)
- Use numbered lists ONLY for sequential steps
- Use ### headers ONLY if organizing truly distinct major sections
- Avoid repetition - say each fact once

**SPECIFIC REQUIREMENTS FOR PMLA QUERIES:**
- IF mentioned in Retrieved Context, prioritize mentioning: "Suspicious Transaction Reports (STRs)"
- IF mentioned in Retrieved Context, prioritize mentioning: "Cash Transaction Reports (CTRs)" with exact thresholds if provided
- IF mentioned in Retrieved Context, prioritize mentioning: "Know Your Customer (KYC)" procedures
- IF mentioned in Retrieved Context, prioritize mentioning: Financial Intelligence Unit-India (FIU-IND)
- **REMINDER:** Rule 1 (STRICT GROUNDING) and Rule 4 (CONTEXT-AWARE DETAILS) take precedence - only mention if in context

**CLOSING:**
"This is informational guidance only. For specific legal or financial advice, please consult a qualified professional."

**REMEMBER:**
❌ NO: "Answering Your Question:", "According to [Source]", inline citations, delaying the answer
✅ YES: Direct answer to the question in first sentence, then context, specific thresholds (if in context), operational terms (if in context)
✅ PRIORITY: Answer the direct question FIRST, then provide supporting details

Now provide your response following this EXACT format:"""
        
        return prompt
    
    def generate_response(self, prompt: str, query: str = "") -> Dict[str, Any]:
        """
        Generate response using Ollama LLM
        
        Args:
            prompt: Formatted prompt
            query: Original user query (for language detection logging)
            
        Returns:
            Response dictionary with text and metadata
        """
        logger.info(f"Generating response with model: {self.model_name}")

        """try:
            # ====================== SIMULATION MODE ==========================
            # Instead of calling Ollama (which is heavy), we simulate a realistic
            # answer so that the translation pipeline, frontend, and backend
            # can all be tested end-to-end without loading large LLMs.
            logger.warning("SIMULATION MODE ACTIVE: Skipping Ollama and generating fake response.")
            
            simulated_answer = (
                f"This is a simulated answer about '{query}'. "
                "In the real system, this text would be generated by the LLM "
                "based on the retrieved FIU-IND documents. "
                "Your translation and pipeline are working correctly!"
            )
        
            return {
                'text': simulated_answer,
                'model': f"{self.model_name} (simulated)",
                'tokens': len(simulated_answer.split()),
                'success': True
            }
            # =================================================================
        
        except Exception as e:
            logger.error(f"Error generating response (simulation): {str(e)}")
            return {
                'text': "I apologize, but I encountered an error generating a response. Please try again.",
                'model': self.model_name,
                'error': str(e),
                'success': False
            }"""

        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for factual responses
                    'top_p': 0.9,
                    'top_k': 40,
                }
            )
            
            return {
                'text': response['response'],
                'model': self.model_name,
                'tokens': response.get('eval_count', 0),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'text': "I apologize, but I encountered an error generating a response. Please try again.",
                'model': self.model_name,
                'error': str(e),
                'success': False
            }
    
    def generate_response_stream(self, prompt: str):
        """
        Generate streaming response using Ollama LLM
        
        Args:
            prompt: Formatted prompt
            
        Yields:
            Response chunks as they are generated
        """
        logger.info(f"Generating streaming response with model: {self.model_name}")
        
        try:
            # Call Ollama API with streaming
            stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': 0.1,  # Low temperature for factual responses
                    'top_p': 0.9,
                    'top_k': 40,
                }
            )
            
            for chunk in stream:
                if 'response' in chunk and chunk['response']:
                    yield chunk['response']
                if chunk.get('done', False):
                    break
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield "I apologize, but I encountered an error generating a response. Please try again."
    
    def query(self, user_question: str, k: int = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            user_question: User's question
            k: Number of documents to retrieve
            conversation_history: Previous conversation messages for context
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        import time
        start_time = time.time()
        
        # Step 0: Transform query to standalone if conversation history exists
        # This handles follow-up questions like "What are the penalties for that?"
        query_was_transformed = False
        standalone_query = user_question
        if self.enable_query_transformation:
            print(f"\n🚀 QUERY TRANSFORMATION ENABLED (non-streaming)")
            original_query = user_question
            standalone_query = self.transform_query_with_history(user_question, conversation_history)
            query_was_transformed = (standalone_query.lower().strip() != original_query.lower().strip())
            print(f"📊 Transformation Result: {'TRANSFORMED' if query_was_transformed else 'NO CHANGE'}\n")
        else:
            print(f"\n⚠️  QUERY TRANSFORMATION DISABLED - Using original query\n")
            logger.debug("Query transformation disabled, using original query")
        
        # Step 1: Retrieve relevant documents using the standalone query
        retrieved_docs = self.retrieve_documents(standalone_query, k=k)
        
        # Step 2: Generate prompt with conversation history
        # Use original user_question for response generation (maintains original intent)
        prompt = self.generate_prompt(user_question, retrieved_docs, conversation_history)
        
        # Step 3: Generate response
        llm_response = self.generate_response(prompt, user_question)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Format sources for frontend
        sources = []
        for doc in retrieved_docs:
            # Support both 'source_file' (new) and 'filename' (legacy) metadata fields
            source_file = doc['metadata'].get('source_file') or doc['metadata'].get('filename', 'Unknown')
            source_info = {
                'document': source_file,
                'doc_type': doc['metadata'].get('doc_type', 'General'),
                'chunk': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'score': doc['score']
            }
            # Add section info if available
            if doc['metadata'].get('section'):
                source_info['section'] = doc['metadata'].get('section')
            sources.append(source_info)
        
        # Build metadata
        metadata = {
            'model': self.model_name,
            'response_time': round(response_time, 2),
            'tokens_generated': llm_response.get('tokens', 0),
            'documents_retrieved': len(retrieved_docs),
            'success': llm_response['success']
        }
        
        # Add query transformation info if transformation was enabled
        if self.enable_query_transformation:
            metadata['query_transformation'] = {
                'enabled': True,
                'was_transformed': query_was_transformed,
                'original_query': user_question if query_was_transformed else None,
                'standalone_query': standalone_query if query_was_transformed else None
            }
        
        return {
            'answer': llm_response['text'],
            'sources': sources,
            'metadata': metadata
        }
    
    def query_stream(self, user_question: str, k: int = None, conversation_history: List[Dict[str, str]] = None):
        """
        Complete RAG pipeline with streaming: retrieve + generate stream
        
        Args:
            user_question: User's question
            k: Number of documents to retrieve
            conversation_history: Previous conversation messages for context
            
        Yields:
            Dictionary with chunk type and content
        """
        # Step 0: Transform query to standalone if conversation history exists
        # This handles follow-up questions like "What are the penalties for that?"
        query_was_transformed = False
        standalone_query = user_question
        if self.enable_query_transformation:
            print(f"\n🚀 QUERY TRANSFORMATION ENABLED (streaming)")
            original_query = user_question
            standalone_query = self.transform_query_with_history(user_question, conversation_history)
            query_was_transformed = (standalone_query.lower().strip() != original_query.lower().strip())
            print(f"📊 Transformation Result: {'TRANSFORMED' if query_was_transformed else 'NO CHANGE'}\n")
        else:
            print(f"\n⚠️  QUERY TRANSFORMATION DISABLED - Using original query\n")
            logger.debug("Query transformation disabled, using original query")
        
        # Step 1: Retrieve relevant documents using the standalone query
        retrieved_docs = self.retrieve_documents(standalone_query, k=k)
        
        # Step 2: Generate prompt with conversation history
        # Use original user_question for response generation (maintains original intent)
        prompt = self.generate_prompt(user_question, retrieved_docs, conversation_history)
        
        # Step 3: Format sources for frontend
        sources = []
        for doc in retrieved_docs:
            # Support both 'source_file' (new) and 'filename' (legacy) metadata fields
            source_file = doc['metadata'].get('source_file') or doc['metadata'].get('filename', 'Unknown')
            source_info = {
                'document': source_file,
                'doc_type': doc['metadata'].get('doc_type', 'General'),
                'chunk': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'score': doc['score']
            }
            # Add section info if available
            if doc['metadata'].get('section'):
                source_info['section'] = doc['metadata'].get('section')
            sources.append(source_info)
        
        # Send transformation metadata first (for debugging)
        if self.enable_query_transformation:
            yield {
                'type': 'transformation',
                'data': {
                    'was_transformed': query_was_transformed,
                    'original_query': user_question,
                    'standalone_query': standalone_query,
                    'history_length': len(conversation_history) if conversation_history else 0
                }
            }
        
        # Send sources metadata
        yield {
            'type': 'sources',
            'data': sources
        }
        
        # Step 4: Stream response chunks
        for chunk in self.generate_response_stream(prompt):
            yield {
                'type': 'chunk',
                'data': chunk
            }
        
        # Send done signal
        yield {
            'type': 'done',
            'data': None
        }
    
    def switch_model(self, model_name: str):
        """
        Switch to a different Ollama model
        
        Args:
            model_name: Name of the model (e.g., 'mistral:7b-instruct')
        """
        logger.info(f"Switching model from {self.model_name} to {model_name}")
        self.model_name = model_name


# Supported models
SUPPORTED_MODELS = {
    'llama3.2:3b': {
        'name': 'Llama 3.2 3B',
        'description': 'Fast and efficient, good for general queries',
        'size': '3B parameters'
    },
    'mistral:7b-instruct': {
        'name': 'Mistral 7B Instruct',
        'description': 'Excellent for factual Q&A and reasoning',
        'size': '7B parameters'
    },
    'gemma2:2b': {
        'name': 'Gemma 2 2B',
        'description': 'Google model, efficient and accurate',
        'size': '2B parameters'
    },
    'gemma2:9b': {
        'name': 'Gemma 2 9B',
        'description': 'Google model, larger capacity for complex prompts',
        'size': '9B parameters'
    },
    'phi3:3.8b': {
        'name': 'Phi-3',
        'description': 'Microsoft model, strong reasoning',
        'size': '3.8B parameters'
    },
    'qwen2.5:7b': {
        'name': 'Qwen 2.5 7B',
        'description': 'Excellent instruction following, great for RAG fine-tuning',
        'size': '7B parameters'
    }
}


if __name__ == "__main__":
    # Test the pipeline
    print("="*60)
    print("Testing RAG Pipeline")
    print("="*60)
    
    # Initialize pipeline
    rag = RAGPipeline(model_name="gemma2:9b")
    
    # Test queries (will use adaptive k automatically)
    test_queries = [
        "What is PMLA?",  # Simple factual → k=3-5
        "What is the penalty for PMLA non-compliance?",  # Factual → k=3-5
        "How can I report suspicious financial activity?",  # Procedural → k=5-7
        "Compare the penalties under Section 4 and Section 13 of PMLA"  # Complex → k=7-10
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*60)
        
        # k=None will use adaptive k based on query complexity
        result = rag.query(query, k=None)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"\nSources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['document']} (score: {source['score']:.3f})")
        print(f"\nMetadata: {result['metadata']}")
        print("="*60)


