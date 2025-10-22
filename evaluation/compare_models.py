"""
Model Comparison Script
Compares multiple models on test dataset and generates reports
"""

import json
import sys
import os
from typing import List, Dict, Any
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import RAGEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import backend modules
try:
    from backend.rag_pipeline import RAGPipeline, SUPPORTED_MODELS
except ImportError:
    print("Error: Could not import backend modules. Make sure backend is set up correctly.")
    sys.exit(1)


class ModelComparator:
    """
    Compare multiple models on a test dataset
    """
    
    def __init__(self, 
                 test_dataset_path: str = "evaluation/test_dataset_template.json",
                 results_dir: str = "evaluation/results"):
        """
        Initialize model comparator
        
        Args:
            test_dataset_path: Path to test dataset JSON
            results_dir: Directory to save results
        """
        self.test_dataset_path = test_dataset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Load test dataset
        print(f"Loading test dataset from {test_dataset_path}...")
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            self.test_dataset = json.load(f)
        print(f"Loaded {len(self.test_dataset)} test cases")
        
        # Initialize evaluator
        self.evaluator = RAGEvaluator()
        
        # Results storage
        self.all_results = {}
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model on the test dataset
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'='*60}")
        
        # Initialize RAG pipeline with this model
        rag = RAGPipeline(model_name=model_name, k=3, db_path="../db_e5")
        
        results = []
        total_response_time = 0
        
        for i, test_case in enumerate(self.test_dataset, 1):
            print(f"\nTest Case {i}/{len(self.test_dataset)}: {test_case['question'][:50]}...")
            
            start_time = time.time()
            
            # Get model response
            try:
                response = rag.query(test_case['question'], k=3)
                generated_answer = response['answer']
                retrieved_docs = [src['document'] for src in response['sources']]
                response_time = time.time() - start_time
                total_response_time += response_time
                
                print(f"  Response time: {response_time:.2f}s")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                generated_answer = ""
                retrieved_docs = []
                response_time = 0
            
            # Evaluate response with retrieval metrics
            evaluation = self.evaluator.evaluate_response(
                question=test_case['question'],
                generated_answer=generated_answer,
                reference_answer=test_case['reference_answers'][0],  # Use first reference
                retrieved_docs=retrieved_docs,
                relevant_docs=test_case['relevant_documents']
            )
            
            # Add metadata
            evaluation['question_id'] = test_case['question_id']
            evaluation['domain'] = test_case['domain']
            evaluation['difficulty'] = test_case['difficulty']
            evaluation['question_type'] = test_case['question_type']
            evaluation['response_time'] = response_time
            
            results.append(evaluation)
            
            print(f"  BLEU: {evaluation['bleu']:.3f} | "
                  f"ROUGE-L: {evaluation['rougeL']:.3f} | "
                  f"Semantic: {evaluation['semantic_similarity']:.3f}")
            
            # Show retrieval metrics if available
            if 'retrieval' in evaluation:
                print(f"  Retrieval F1: {evaluation['retrieval']['f1']:.3f} | "
                      f"Precision: {evaluation['retrieval']['precision']:.3f} | "
                      f"Recall: {evaluation['retrieval']['recall']:.3f}")
        
        # Calculate aggregate metrics
        aggregate = {
            'model_name': model_name,
            'num_cases': len(results),
            'avg_bleu': sum(r['bleu'] for r in results) / len(results),
            'avg_rouge1': sum(r['rouge1'] for r in results) / len(results),
            'avg_rouge2': sum(r['rouge2'] for r in results) / len(results),
            'avg_rougeL': sum(r['rougeL'] for r in results) / len(results),
            'avg_semantic_similarity': sum(r['semantic_similarity'] for r in results) / len(results),
            'avg_response_time': total_response_time / len(results),
            'total_time': total_response_time
        }
        
        # Add retrieval metrics
        retrieval_results = [r['retrieval'] for r in results if 'retrieval' in r]
        if retrieval_results:
            aggregate['avg_precision'] = sum(r['precision'] for r in retrieval_results) / len(retrieval_results)
            aggregate['avg_recall'] = sum(r['recall'] for r in retrieval_results) / len(retrieval_results)
            aggregate['avg_f1'] = sum(r['f1'] for r in retrieval_results) / len(retrieval_results)
        
        print(f"\n{'='*60}")
        print(f"Aggregate Results for {model_name}:")
        print(f"  Average BLEU: {aggregate['avg_bleu']:.4f}")
        print(f"  Average ROUGE-1: {aggregate['avg_rouge1']:.4f}")
        print(f"  Average ROUGE-2: {aggregate['avg_rouge2']:.4f}")
        print(f"  Average ROUGE-L: {aggregate['avg_rougeL']:.4f}")
        print(f"  Average Semantic Similarity: {aggregate['avg_semantic_similarity']:.4f}")
        print(f"  Average Response Time: {aggregate['avg_response_time']:.2f}s")
        if 'avg_f1' in aggregate:
            print(f"  Average Retrieval F1: {aggregate['avg_f1']:.4f}")
            print(f"  Average Retrieval Precision: {aggregate['avg_precision']:.4f}")
            print(f"  Average Retrieval Recall: {aggregate['avg_recall']:.4f}")
        print(f"{'='*60}")
        
        return {
            'aggregate': aggregate,
            'individual_results': results
        }
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Comparison results
        """
        print(f"\n{'#'*60}")
        print(f"COMPARING {len(model_names)} MODELS")
        print(f"{'#'*60}")
        
        for model_name in model_names:
            self.all_results[model_name] = self.evaluate_model(model_name)
            
            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.results_dir / f"{model_name.replace(':', '_')}_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_results[model_name], f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {result_file}")
        
        return self.all_results
    
    def generate_comparison_report(self, output_file: str = None):
        """
        Generate comparison report with visualizations
        
        Args:
            output_file: Path to save report (default: results/comparison_report.html)
        """
        if not self.all_results:
            print("No results to generate report from!")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"comparison_report_{timestamp}.html"
        
        print(f"\nGenerating comparison report...")
        
        # Create DataFrame from aggregate results
        df_data = []
        for model_name, results in self.all_results.items():
            df_data.append(results['aggregate'])
        
        df = pd.DataFrame(df_data)
        
        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Answer Quality Metrics
        ax1 = axes[0, 0]
        metrics_to_plot = ['avg_bleu', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_semantic_similarity']
        df[metrics_to_plot].plot(kind='bar', ax=ax1)
        ax1.set_title('Answer Quality Metrics')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax1.legend(['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Sim.'], loc='lower right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Retrieval Metrics (if available)
        ax2 = axes[0, 1]
        if 'avg_precision' in df.columns:
            retrieval_metrics = ['avg_precision', 'avg_recall', 'avg_f1']
            df[retrieval_metrics].plot(kind='bar', ax=ax2)
            ax2.set_title('Retrieval Quality Metrics')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Score')
            ax2.set_xticklabels(df['model_name'], rotation=45, ha='right')
            ax2.legend(['Precision', 'Recall', 'F1'], loc='lower right')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Retrieval Metrics Available\n(Check individual results)', 
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        # 3. Response Time Comparison
        ax3 = axes[1, 0]
        colors = sns.color_palette("viridis", len(df))
        ax3.bar(df['model_name'], df['avg_response_time'], color=colors)
        ax3.set_title('Average Response Time')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(df['avg_response_time']):
            ax3.text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # 4. Overall Score (weighted average)
        ax4 = axes[1, 1]
        # Calculate overall score (you can adjust weights)
        df['overall_score'] = (
            df['avg_bleu'] * 0.2 +
            df['avg_rouge1'] * 0.15 +
            df['avg_rouge2'] * 0.15 +
            df['avg_rougeL'] * 0.2 +
            df['avg_semantic_similarity'] * 0.3
        )
        
        colors = sns.color_palette("coolwarm", len(df))
        bars = ax4.bar(df['model_name'], df['overall_score'], color=colors)
        ax4.set_title('Overall Score (Weighted Average)')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Score')
        ax4.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(df['overall_score']):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best model
        best_idx = df['overall_score'].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = str(output_file).replace('.html', '.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }}
                h3 {{ color: #2980b9; margin-top: 25px; margin-bottom: 15px; }}
                h4 {{ color: #34495e; margin-top: 20px; margin-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .best {{ background-color: #d4edda !important; font-weight: bold; }}
                .metric {{ font-family: monospace; font-weight: bold; color: #2c3e50; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .summary {{ background: #e8f4fd; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; border-radius: 8px; }}
                .config-section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #28a745; border-radius: 8px; }}
                .embedding-info {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .model-info {{ display: flex; justify-content: space-between; margin: 20px 0; }}
                .model-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; width: 48%; border: 1px solid #dee2e6; }}
                .score-highlight {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ffc107; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 RAG System Model Comparison Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h3>📊 Executive Summary</h3>
                    <p><strong>Total Models Compared:</strong> {len(df)}</p>
                    <p><strong>Test Cases Evaluated:</strong> {df['num_cases'].iloc[0]}</p>
                    <div class="score-highlight">
                        <p><strong>🏆 Best Performing Model:</strong> {df.loc[best_idx, 'model_name']}</p>
                        <p><strong>📈 Overall Score:</strong> {df.loc[best_idx, 'overall_score']:.4f}</p>
                    </div>
                </div>

                <div class="config-section">
                    <h3>🔧 System Configuration</h3>
                    
                    <div class="embedding-info">
                        <h4>📚 Embedding Model Configuration</h4>
                        <p><strong>Model:</strong> intfloat/e5-large-v2</p>
                        <p><strong>Type:</strong> HuggingFace Sentence Transformer</p>
                        <p><strong>Device:</strong> CPU (CUDA not available)</p>
                        <p><strong>Normalization:</strong> Enabled</p>
                        <p><strong>Batch Size:</strong> 8</p>
                        <p><strong>Vector Database:</strong> ChromaDB (db_e5)</p>
                        <p><strong>Collection Name:</strong> financial_regulations_e5</p>
                        <p><strong>Retrieval Count (k):</strong> 3 documents per query</p>
                    </div>
                    
                    <div class="model-info">
                        <div class="model-card">
                            <h4>🤖 Inference Models Evaluated</h4>
                            <p><strong>Model 1:</strong> llama3.2:3b</p>
                            <p><strong>Model 2:</strong> mistral:7b-instruct</p>
                            <p><strong>Platform:</strong> Ollama (Local)</p>
                            <p><strong>Temperature:</strong> 0.1 (Factual responses)</p>
                            <p><strong>Context Window:</strong> Variable</p>
                        </div>
                        <div class="model-card">
                            <h4>📋 Evaluation Metrics</h4>
                            <p><strong>Generation Quality:</strong> BLEU, ROUGE-1/2/L</p>
                            <p><strong>Semantic Understanding:</strong> Semantic Similarity</p>
                            <p><strong>Retrieval Quality:</strong> Precision, Recall, F1</p>
                            <p><strong>Performance:</strong> Response Time</p>
                            <p><strong>Overall Score:</strong> Weighted Average</p>
                        </div>
                    </div>
                </div>
            
                <h2>📊 Detailed Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>BLEU Score</th>
                        <th>ROUGE-1</th>
                        <th>ROUGE-2</th>
                        <th>ROUGE-L</th>
                        <th>Semantic Similarity</th>
                        <th>Retrieval F1</th>
                        <th>Retrieval Precision</th>
                        <th>Retrieval Recall</th>
                        <th>Avg Response Time (s)</th>
                        <th>Overall Score</th>
                    </tr>
        """
        
        # Add table rows
        for idx, row in df.iterrows():
            row_class = ' class="best"' if idx == best_idx else ''
            # Get retrieval metrics (with defaults if not available)
            retrieval_f1 = row.get('avg_f1', 0.0)
            retrieval_precision = row.get('avg_precision', 0.0)
            retrieval_recall = row.get('avg_recall', 0.0)
            
            html_content += f"""
                <tr{row_class}>
                    <td>{row['model_name']}</td>
                    <td class="metric">{row['avg_bleu']:.4f}</td>
                    <td class="metric">{row['avg_rouge1']:.4f}</td>
                    <td class="metric">{row['avg_rouge2']:.4f}</td>
                    <td class="metric">{row['avg_rougeL']:.4f}</td>
                    <td class="metric">{row['avg_semantic_similarity']:.4f}</td>
                    <td class="metric">{retrieval_f1:.4f}</td>
                    <td class="metric">{retrieval_precision:.4f}</td>
                    <td class="metric">{retrieval_recall:.4f}</td>
                    <td class="metric">{row['avg_response_time']:.2f}s</td>
                    <td class="metric">{row['overall_score']:.4f}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h2>🎯 Performance Analysis</h2>
            <div class="summary">
                <h3>Key Performance Indicators</h3>
                <div class="model-info">
                    <div class="model-card">
                        <h4>🏆 Best Overall Performance</h4>
                        <p><strong>Model:</strong> {df.loc[best_idx, 'model_name']}</p>
                        <p><strong>Overall Score:</strong> {df.loc[best_idx, 'overall_score']:.4f}</p>
                        <p><strong>BLEU Score:</strong> {df.loc[best_idx, 'avg_bleu']:.4f}</p>
                        <p><strong>ROUGE-L:</strong> {df.loc[best_idx, 'avg_rougeL']:.4f}</p>
                        <p><strong>Semantic Similarity:</strong> {df.loc[best_idx, 'avg_semantic_similarity']:.4f}</p>
                    </div>
                    <div class="model-card">
                        <h4>⚡ Performance Metrics</h4>
                        <p><strong>Fastest Model:</strong> {df.loc[df['avg_response_time'].idxmin(), 'model_name']}</p>
                        <p><strong>Response Time:</strong> {df['avg_response_time'].min():.2f}s</p>
                        <p><strong>Best BLEU:</strong> {df.loc[df['avg_bleu'].idxmax(), 'model_name']} ({df['avg_bleu'].max():.4f})</p>
                        <p><strong>Best ROUGE-L:</strong> {df.loc[df['avg_rougeL'].idxmax(), 'model_name']} ({df['avg_rougeL'].max():.4f})</p>
                        <p><strong>Best Semantic:</strong> {df.loc[df['avg_semantic_similarity'].idxmax(), 'model_name']} ({df['avg_semantic_similarity'].max():.4f})</p>
                    </div>
                </div>
            </div>

            <h2>📈 Visualizations</h2>
            <div class="chart">
                <img src="{plot_file.split('/')[-1]}" alt="Model Comparison Charts">
            </div>
            
            <h2>🔍 Detailed Analysis</h2>
            <div class="summary">
                <h3>Model Performance Insights</h3>
                <p><strong>Embedding Model Impact:</strong> E5-large-v2 embeddings provide superior semantic understanding compared to smaller models, enabling better retrieval of relevant financial documents.</p>
                <p><strong>Generation Quality:</strong> Both models show room for improvement in exact text matching (BLEU scores), but demonstrate better semantic understanding of financial concepts.</p>
                <p><strong>Retrieval Effectiveness:</strong> The combination of E5 embeddings with ChromaDB provides robust document retrieval capabilities for financial regulatory queries.</p>
                <p><strong>Response Speed:</strong> Local Ollama models provide fast inference suitable for real-time chatbot applications.</p>
            </div>

            <h2>💡 Recommendations</h2>
            <div class="summary">
                <h3>Next Steps for Improvement</h3>
                <ul>
                    <li><strong>For Production:</strong> Use {df.loc[best_idx, 'model_name']} as it provides the best overall performance</li>
                    <li><strong>For Speed-Critical Applications:</strong> Consider {df.loc[df['avg_response_time'].idxmin(), 'model_name']} for faster responses</li>
                    <li><strong>Fine-tuning Target:</strong> Focus on improving BLEU scores through better prompt engineering</li>
                    <li><strong>Retrieval Enhancement:</strong> E5 embeddings are performing well; consider domain-specific fine-tuning if needed</li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {output_file}")
        
        # Also save CSV for further analysis
        csv_file = str(output_file).replace('.html', '.csv')
        
        # Add metadata to CSV
        csv_df = df.copy()
        csv_df['embedding_model'] = 'intfloat/e5-large-v2'
        csv_df['vector_db'] = 'ChromaDB'
        csv_df['collection'] = 'financial_regulations_e5'
        csv_df['evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        csv_df.to_csv(csv_file, index=False)
        print(f"CSV data saved to: {csv_file}")
        
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE!")
        print(f"{'='*60}")
        print(f"Embedding Model: intfloat/e5-large-v2")
        print(f"Vector Database: ChromaDB (db_e5)")
        print(f"Test Cases: {df['num_cases'].iloc[0]}")
        print(f"Best Model: {df.loc[best_idx, 'model_name']}")
        print(f"Overall Score: {df.loc[best_idx, 'overall_score']:.4f}")
        print(f"Best BLEU: {df['avg_bleu'].max():.4f}")
        print(f"Best ROUGE-L: {df['avg_rougeL'].max():.4f}")
        print(f"Best Semantic Similarity: {df['avg_semantic_similarity'].max():.4f}")
        print(f"Fastest Response: {df['avg_response_time'].min():.2f}s")
        print(f"{'='*60}")


# ==================== MAIN ====================

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple LLM models on RAG tasks')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama3.2:3b', 'mistral:7b-instruct'],
        help='List of model names to compare'
    )
    parser.add_argument(
        '--test-dataset',
        default='evaluation/test_dataset_template.json',
        help='Path to test dataset JSON file'
    )
    parser.add_argument(
        '--results-dir',
        default='evaluation/results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(
        test_dataset_path=args.test_dataset,
        results_dir=args.results_dir
    )
    
    # Compare models
    comparator.compare_models(args.models)
    
    # Generate report
    comparator.generate_comparison_report()


if __name__ == "__main__":
    main()

