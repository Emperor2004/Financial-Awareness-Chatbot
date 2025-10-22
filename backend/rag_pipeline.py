"""
RAG Pipeline for Financial Awareness Chatbot
Handles document retrieval and LLM response generation
"""

import os
import re
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for financial queries
    """
    
    def __init__(self, 
                 db_path: str = "db_e5",
                 model_name: str = "llama3.2:3b",
                 k: int = 5):
        """
        Initialize RAG pipeline
        
        Args:
            db_path: Path to ChromaDB database
            model_name: Ollama model name to use
            k: Number of documents to retrieve
        """
        self.db_path = db_path
        self.model_name = model_name
        self.k = k
        
        # Initialize embeddings
        logger.info("Loading E5-large-v2 embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': 'cpu'},  # Use CPU (CUDA not available)
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8  # Smaller batch for CPU
            }
        )
        
        # Load vector database
        logger.info(f"Loading vector database from {db_path}...")
        self.vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name="financial_regulations_e5"
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            k: Number of documents to retrieve (default: self.k)
            
        Returns:
            List of retrieved documents with metadata
        """
        if k is None:
            k = self.k
        
        logger.info(f"Retrieving {k} documents for query: '{query[:50]}...'")
        
        # Perform similarity search
        docs = self.vectordb.similarity_search_with_score(query, k=k)
        
        # Format results
        retrieved_docs = []
        for doc, score in docs:
            retrieved_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
    
    def generate_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate prompt for LLM using retrieved documents
        
        Args:
            query: User query
            context_docs: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        # Combine context from retrieved documents with detailed source information
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source_file = doc['metadata'].get('filename', 'Unknown').replace('.txt', '')
            doc_type = doc['metadata'].get('doc_type', 'Official Document')
            
            # Format source citation for internal reference
            source_citation = f"Document: {source_file} | Type: {doc_type}"
            context_parts.append(f"[Source {i}: {source_citation}]\n{doc['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
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

**3. SECURITY & PII HANDLING:**
- NEVER ask for personal information (PAN, Aadhaar, account numbers, etc.)
- If a user shares PII, ignore it and respond: "Please do not share personal information. I provide general information only and do not need personal details."
- Remind users: "Do not share sensitive personal or financial information in this chat."

**4. SYNTHESIS OVER CONCATENATION:**
- **CRITICAL**: Create a SINGLE COHERENT NARRATIVE from all sources
- DO NOT list isolated facts from each source separately
- SYNTHESIZE information into a flowing, logical explanation
- Weave facts together naturally - don't jump between disconnected points
- Prioritize information from higher-relevance sources
- Only include information that directly answers the question

**5. DIRECT AUTHORITATIVE TONE - CRITICAL:**
- ❌ NEVER WRITE: "Answering Your Question:", "Let me explain:", "According to [Source 1: ...]"
- ✅ START IMMEDIATELY: "The Prevention of Money Laundering Act (PMLA), 2002, is an Act of Parliament..."
- Use direct, declarative statements as if you ARE the official authority
- NO conversational preambles, NO hedging, NO source citations in text

**EXAMPLE OF WHAT NOT TO DO:**
"Answering Your Question: What is PMLA?
According to [Source 1: Document: PMLA 2002 | Type: General], PMLA stands for..."

**EXAMPLE OF CORRECT APPROACH:**
"The Prevention of Money Laundering Act (PMLA), 2002, is an Act of Parliament that came into force on July 1, 2005..."

**6. INCLUDE KEY OPERATIONAL DETAILS - MANDATORY:**
- For PMLA queries: ALWAYS mention STR (Suspicious Transaction Report) and CTR (Cash Transaction Report)
- Include specific thresholds: "CTRs for cash transactions exceeding ₹10 lakhs"
- Include CBWTR (Cross-Border Wire Transfer Report) when relevant
- Mention KYC (Know Your Customer) requirements
- Provide ACTIONABLE details with numbers, timelines, and specific requirements

**7. ABSOLUTELY NO INLINE CITATIONS:**
- ❌ NEVER: "[Source 1: ...]", "According to [Source X: ...]", "As per [Document Y]"
- ✅ INSTEAD: Simply state the facts directly and authoritatively
- The system will add sources at the end automatically
- Your job is ONLY to synthesize information into clean prose

**8. LIMITATIONS & ESCALATION:**
- Acknowledge when a question is too complex or ambiguous
- Clearly state what you CAN and CANNOT do
- For complex legal interpretation: "This matter may require legal expertise. Please consult FIU-IND or a compliance professional."
- For technical issues (website, forms, submissions): "For technical assistance, please contact FIU-IND support at [contact details]."

**9. RESPONSE STRUCTURE:**
- Opening: Direct, immediate answer (no preamble)
- Body: 3-5 paragraphs of comprehensive, synthesized explanation
- Key terms: **Bold** important legislation, concepts, or requirements
- Lists: Use numbered lists ONLY for sequential steps or procedures
- Headers: Use ### only if genuinely organizing distinct major sections
- Closing: Brief disclaimer about consulting professionals if needed

### Retrieved Context from Official Documents:
{context}

---

### User's Question:
{query}

---

### MANDATORY RESPONSE FORMAT:

**OPENING (First Sentence):**
Start DIRECTLY with: "The [Act/Regulation name], [year], is..." 
NO preambles. NO "Answering your question". NO "According to [Source]".

**BODY (3-4 Paragraphs):**
1. **First paragraph**: Core definition and purpose
2. **Second paragraph**: Key operational mechanisms (STR, CTR, CBWTR, KYC) with thresholds
3. **Third paragraph**: Compliance requirements and procedures
4. **Fourth paragraph**: Consequences/penalties (if relevant)

**STRUCTURE:**
- Use **bold** for: Act names, key terms (STR, CTR, KYC, CBWTR)
- Use numbered lists ONLY for sequential steps
- Use ### headers ONLY if organizing truly distinct major sections
- Avoid repetition - say each fact once

**SPECIFIC REQUIREMENTS FOR PMLA QUERIES:**
- MUST mention: "Suspicious Transaction Reports (STRs)"
- MUST mention: "Cash Transaction Reports (CTRs) for transactions exceeding ₹10 lakhs"
- MUST mention: "Know Your Customer (KYC)" procedures
- MUST mention: Financial Intelligence Unit-India (FIU-IND)

**CLOSING:**
"This is informational guidance only. For specific legal or financial advice, please consult a qualified professional."

**REMEMBER:**
❌ NO: "Answering Your Question:", "According to [Source]", inline citations
✅ YES: Direct statements, specific thresholds, operational terms (STR/CTR)

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
    
    def query(self, user_question: str, k: int = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            user_question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        import time
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(user_question, k=k)
        
        # Step 2: Generate prompt
        prompt = self.generate_prompt(user_question, retrieved_docs)
        
        # Step 3: Generate response
        llm_response = self.generate_response(prompt, user_question)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Format sources for frontend
        sources = [
            {
                'document': doc['metadata'].get('filename', 'Unknown'),
                'doc_type': doc['metadata'].get('doc_type', 'General'),
                'chunk': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'score': doc['score']
            }
            for doc in retrieved_docs
        ]
        
        return {
            'answer': llm_response['text'],
            'sources': sources,
            'metadata': {
                'model': self.model_name,
                'response_time': round(response_time, 2),
                'tokens_generated': llm_response.get('tokens', 0),
                'documents_retrieved': len(retrieved_docs),
                'success': llm_response['success']
            }
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
    'phi3:3.8b': {
        'name': 'Phi-3',
        'description': 'Microsoft model, strong reasoning',
        'size': '3.8B parameters'
    }
}


if __name__ == "__main__":
    # Test the pipeline
    print("="*60)
    print("Testing RAG Pipeline")
    print("="*60)
    
    # Initialize pipeline
    rag = RAGPipeline(model_name="llama3.2:3b")
    
    # Test queries
    test_queries = [
        "What is PMLA?",
        "What is the penalty for PMLA non-compliance?",
        "How can I report suspicious financial activity?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*60)
        
        result = rag.query(query, k=3)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"\nSources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['document']} (score: {source['score']:.3f})")
        print(f"\nMetadata: {result['metadata']}")
        print("="*60)


