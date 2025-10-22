"""
Evaluation Metrics for RAG System
Implements BLEU, ROUGE, F1, Precision, Recall, and Semantic Similarity
"""

import numpy as np
from typing import List, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not installed. BLEU score will not be available.")
    sentence_bleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    logger.warning("rouge-score not installed. ROUGE scores will not be available.")
    rouge_scorer = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    logger.warning("sentence-transformers not installed. Semantic similarity will not be available.")
    SentenceTransformer = None


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems
    """
    
    def __init__(self, semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize evaluator
        
        Args:
            semantic_model: Model to use for semantic similarity
        """
        # Initialize ROUGE scorer
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Initialize semantic similarity model
        if SentenceTransformer:
            logger.info(f"Loading semantic model: {semantic_model}")
            self.semantic_model = SentenceTransformer(semantic_model)
        else:
            self.semantic_model = None
        
        logger.info("RAGEvaluator initialized")
    
    # ==================== ANSWER QUALITY METRICS ====================
    
    def bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate
        
        Args:
            reference: Reference (ground truth) answer
            candidate: Generated answer
            
        Returns:
            BLEU score (0-1)
        """
        if not sentence_bleu:
            logger.warning("BLEU score not available (NLTK not installed)")
            return 0.0
        
        try:
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            
            # Use smoothing to avoid zero scores
            smoothing = SmoothingFunction().method1
            
            score = sentence_bleu(
                [ref_tokens], 
                cand_tokens, 
                smoothing_function=smoothing
            )
            
            return float(score)
        
        except Exception as e:
            logger.error(f"Error calculating BLEU: {str(e)}")
            return 0.0
    
    def rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            reference: Reference answer
            candidate: Generated answer
            
        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        if not self.rouge_scorer:
            logger.warning("ROUGE scores not available (rouge-score not installed)")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        
        except Exception as e:
            logger.error(f"Error calculating ROUGE: {str(e)}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
    
    def semantic_similarity(self, reference: str, candidate: str) -> float:
        """
        Calculate semantic similarity using embeddings
        
        Args:
            reference: Reference answer
            candidate: Generated answer
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not self.semantic_model:
            logger.warning("Semantic similarity not available (sentence-transformers not installed)")
            return 0.0
        
        try:
            ref_embedding = self.semantic_model.encode(reference, convert_to_tensor=True)
            cand_embedding = self.semantic_model.encode(candidate, convert_to_tensor=True)
            
            similarity = util.cos_sim(ref_embedding, cand_embedding).item()
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    # ==================== RETRIEVAL METRICS ====================
    
    def retrieval_metrics(self, 
                         retrieved_docs: List[str], 
                         relevant_docs: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 for retrieval
        
        Args:
            retrieved_docs: List of retrieved document IDs/names
            relevant_docs: List of relevant (ground truth) document IDs/names
            
        Returns:
            Dictionary with precision, recall, f1
        """
        try:
            retrieved_set = set(retrieved_docs)
            relevant_set = set(relevant_docs)
            
            if not retrieved_set:
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
            if not relevant_set:
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
            # Calculate true positives
            true_positives = len(retrieved_set.intersection(relevant_set))
            
            # Calculate metrics
            precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
            recall = true_positives / len(relevant_set) if relevant_set else 0.0
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    # ==================== COMBINED EVALUATION ====================
    
    def evaluate_response(self,
                         question: str,
                         generated_answer: str,
                         reference_answer: str,
                         retrieved_docs: List[str] = None,
                         relevant_docs: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single response
        
        Args:
            question: The user question
            generated_answer: Answer generated by the system
            reference_answer: Ground truth answer
            retrieved_docs: Documents retrieved by the system
            relevant_docs: Ground truth relevant documents
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating response for question: {question[:50]}...")
        
        results = {
            'question': question,
            'generated_answer': generated_answer,
            'reference_answer': reference_answer,
        }
        
        # Answer quality metrics
        results['bleu'] = self.bleu_score(reference_answer, generated_answer)
        results.update(self.rouge_scores(reference_answer, generated_answer))
        results['semantic_similarity'] = self.semantic_similarity(reference_answer, generated_answer)
        
        # Retrieval metrics (if provided)
        if retrieved_docs and relevant_docs:
            results['retrieval'] = self.retrieval_metrics(retrieved_docs, relevant_docs)
        
        # Calculate average score
        answer_scores = [
            results['bleu'],
            results['rouge1'],
            results['rouge2'],
            results['rougeL'],
            results['semantic_similarity']
        ]
        results['average_score'] = np.mean(answer_scores)
        
        return results
    
    def evaluate_batch(self,
                      test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple test cases
        
        Args:
            test_cases: List of test case dictionaries, each containing:
                - question
                - generated_answer
                - reference_answer
                - retrieved_docs (optional)
                - relevant_docs (optional)
                
        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        all_results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = self.evaluate_response(
                question=test_case['question'],
                generated_answer=test_case['generated_answer'],
                reference_answer=test_case['reference_answer'],
                retrieved_docs=test_case.get('retrieved_docs'),
                relevant_docs=test_case.get('relevant_docs')
            )
            
            all_results.append(result)
        
        # Calculate aggregate statistics
        aggregate = {
            'num_cases': len(all_results),
            'average_bleu': np.mean([r['bleu'] for r in all_results]),
            'average_rouge1': np.mean([r['rouge1'] for r in all_results]),
            'average_rouge2': np.mean([r['rouge2'] for r in all_results]),
            'average_rougeL': np.mean([r['rougeL'] for r in all_results]),
            'average_semantic_similarity': np.mean([r['semantic_similarity'] for r in all_results]),
        }
        
        # Add retrieval metrics if available
        retrieval_results = [r['retrieval'] for r in all_results if 'retrieval' in r]
        if retrieval_results:
            aggregate['average_precision'] = np.mean([r['precision'] for r in retrieval_results])
            aggregate['average_recall'] = np.mean([r['recall'] for r in retrieval_results])
            aggregate['average_f1'] = np.mean([r['f1'] for r in retrieval_results])
        
        return {
            'aggregate': aggregate,
            'individual_results': all_results
        }


# ==================== HELPER FUNCTIONS ====================

def calculate_accuracy(predictions: List[bool]) -> float:
    """
    Calculate accuracy from binary predictions
    
    Args:
        predictions: List of True/False predictions
        
    Returns:
        Accuracy score (0-1)
    """
    if not predictions:
        return 0.0
    return sum(predictions) / len(predictions)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*60)
    print("Testing RAG Evaluator")
    print("="*60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Test case
    question = "What is PMLA?"
    reference = "PMLA is an abbreviation for Prevention of Money Laundering Act, 2002 which is in force since 1st July 2005."
    candidate = "PMLA stands for Prevention of Money Laundering Act 2002, which came into force on July 1, 2005."
    
    # Evaluate
    result = evaluator.evaluate_response(
        question=question,
        generated_answer=candidate,
        reference_answer=reference,
        retrieved_docs=["PMLA 2002.txt", "Money Laundering.txt"],
        relevant_docs=["PMLA 2002.txt"]
    )
    
    # Print results
    print(f"\nQuestion: {question}")
    print(f"\nReference: {reference}")
    print(f"\nCandidate: {candidate}")
    print("\nScores:")
    print(f"  BLEU: {result['bleu']:.4f}")
    print(f"  ROUGE-1: {result['rouge1']:.4f}")
    print(f"  ROUGE-2: {result['rouge2']:.4f}")
    print(f"  ROUGE-L: {result['rougeL']:.4f}")
    print(f"  Semantic Similarity: {result['semantic_similarity']:.4f}")
    print(f"  Average Score: {result['average_score']:.4f}")
    
    if 'retrieval' in result:
        print(f"\nRetrieval Metrics:")
        print(f"  Precision: {result['retrieval']['precision']:.4f}")
        print(f"  Recall: {result['retrieval']['recall']:.4f}")
        print(f"  F1: {result['retrieval']['f1']:.4f}")
    
    print("="*60)


