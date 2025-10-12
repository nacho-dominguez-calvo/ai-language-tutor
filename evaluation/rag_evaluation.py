"""
RAG Evaluation Script - In-Memory Version with Full Metrics
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
import json
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

from app.rag_chain import ask_with_context
from app.vector_store import create_vectorstore_from_uploaded_files

load_dotenv()

# Test documents paths
TEST_DOCUMENTS_PATH = "data/documents"
TEST_DOCUMENTS = [
    "research_paper.txt",
    "internal_policy.txt",
    "project_alpha.txt",
    "client_records.txt",
    "experiment_log.txt"
]

EVAL_DATASET = [
    {
        "question": "What is the optimal wind speed for microturbine performance?",
        "expected_answer": "8.4 m/s average wind speed",
        "relevant_docs": ["research_paper.txt"]
    },
    {
        "question": "Who is the lead engineer for Project Alpha?",
        "expected_answer": "Laura Méndez",
        "relevant_docs": ["project_alpha.txt"]
    },
    {
        "question": "What is the primary contact for TechNova Industries?",
        "expected_answer": "Carlos Rivera",
        "relevant_docs": ["client_records.txt"]
    },
    {
        "question": "What happens if employees violate data access policy?",
        "expected_answer": "Immediate suspension pending review",
        "relevant_docs": ["internal_policy.txt"]
    },
    {
        "question": "What was the temperature drift observed in sensor S17B-09?",
        "expected_answer": "+0.04°C after 48 hours",
        "relevant_docs": ["experiment_log.txt"]
    },
]

class MockUploadedFile:
    """Mock Streamlit UploadedFile for evaluation."""
    def __init__(self, name, content):
        self.name = name
        self.content = content.encode('utf-8')
        self._position = 0
    
    def read(self):
        return self.content
    
    def seek(self, position):
        self._position = position

# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_context_precision(retrieved_sources: List[str], 
                                 relevant_doc_names: List[str]) -> float:
    """
    Context Precision: Proportion of retrieved docs that are relevant.
    """
    if not retrieved_sources:
        return 0.0
    
    relevant_count = sum(
        1 for source in retrieved_sources 
        if any(name in source for name in relevant_doc_names)
    )
    return relevant_count / len(retrieved_sources)

def calculate_context_recall(retrieved_sources: List[str],
                             relevant_doc_names: List[str]) -> float:
    """
    Context Recall: Proportion of relevant docs that were retrieved.
    """
    if not relevant_doc_names:
        return 1.0
    
    found_count = sum(
        1 for name in relevant_doc_names
        if any(name in source for source in retrieved_sources)
    )
    return found_count / len(relevant_doc_names)

def calculate_faithfulness_llm(question: str, answer: str, 
                               retrieved_sources: List[str], 
                               llm: ChatOpenAI) -> float:
    """
    Faithfulness: Is the answer supported by retrieved documents?
    """
    if not retrieved_sources:
        return 0.0
    
    context_summary = f"Retrieved from documents: {', '.join(retrieved_sources)}"
    
    prompt = f"""Rate the faithfulness of the answer on a scale of 0.0 to 1.0.
Faithfulness means all claims in the answer can be verified from the retrieved documents.

Question: {question}

Answer: {answer}

Retrieved sources: {context_summary}

Return ONLY a number between 0.0 and 1.0 (e.g., 0.8)"""
    
    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5

def calculate_answer_relevance_llm(question: str, answer: str, 
                                   llm: ChatOpenAI) -> float:
    """
    Answer Relevance: Does the answer address the question?
    """
    prompt = f"""Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.
1.0 means the answer directly addresses the question.

Question: {question}

Answer: {answer}

Return ONLY a number between 0.0 and 1.0 (e.g., 0.9)"""
    
    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5

def calculate_correctness_llm(expected_answer: str, actual_answer: str,
                              question: str, llm: ChatOpenAI) -> float:
    """
    Correctness: Does the answer match the expected answer semantically?
    """
    prompt = f"""Compare the actual answer with the expected answer and rate their semantic similarity on a scale of 0.0 to 1.0.
1.0 means they convey the same information.

Question: {question}

Expected answer: {expected_answer}

Actual answer: {actual_answer}

Return ONLY a number between 0.0 and 1.0 (e.g., 0.85)"""
    
    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5

def calculate_answer_length_compliance(answer: str, max_words: int = 50) -> float:
    """
    Checks if answer respects the length constraint.
    """
    word_count = len(answer.split())
    if word_count <= max_words:
        return 1.0
    else:
        return max(0.0, 1.0 - (word_count - max_words) / max_words)

# ============================================================
# LOAD TEST FILES
# ============================================================

def load_test_files():
    """Load test files as mock uploaded files."""
    mock_files = []
    for filename in TEST_DOCUMENTS:
        filepath = os.path.join(TEST_DOCUMENTS_PATH, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                mock_files.append(MockUploadedFile(filename, content))
        else:
            print(f"⚠️ Warning: {filepath} not found")
    return mock_files

# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():
    """Run evaluation with in-memory vector store."""
    print("=" * 60)
    print("RAG EVALUATION SCRIPT (IN-MEMORY)")
    print("=" * 60)
    
    # Load test files
    print("\n📄 Loading test documents...")
    mock_files = load_test_files()
    print(f"✅ Loaded {len(mock_files)} documents:")
    for f in mock_files:
        print(f"  - {f.name}")
    
    if not mock_files:
        print("❌ No test documents found!")
        return
    
    # Create in-memory vector store
    print("\n🔍 Creating in-memory vector store...")
    vectorstore = create_vectorstore_from_uploaded_files(mock_files)
    
    if vectorstore is None:
        print("❌ Failed to create vector store!")
        return
    
    print("✅ Vector store created in memory")
    
    # Initialize evaluation LLM
    print("\n🤖 Initializing evaluation LLM...")
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("✅ Evaluation LLM ready")
    
    # Run evaluation
    print("\n📊 Running evaluation on test dataset...")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(EVAL_DATASET, 1):
        print(f"\nTest Case {i}/{len(EVAL_DATASET)}")
        print(f"Question: {test_case['question']}")
        
        try:
            # Get answer from RAG system
            answer, sources = ask_with_context(test_case['question'], vectorstore)
            print(f"Answer: {answer[:80]}...")
            print(f"Sources: {sources}")
            
            # Calculate all metrics
            context_precision = calculate_context_precision(
                sources, 
                test_case['relevant_docs']
            )
            
            context_recall = calculate_context_recall(
                sources,
                test_case['relevant_docs']
            )
            
            faithfulness = calculate_faithfulness_llm(
                test_case['question'],
                answer,
                sources,
                eval_llm
            )
            
            answer_relevance = calculate_answer_relevance_llm(
                test_case['question'],
                answer,
                eval_llm
            )
            
            correctness = calculate_correctness_llm(
                test_case['expected_answer'],
                answer,
                test_case['question'],
                eval_llm
            )
            
            length_compliance = calculate_answer_length_compliance(answer, max_words=50)
            
            # Store result
            result = {
                "question": test_case['question'],
                "answer": answer,
                "expected_answer": test_case['expected_answer'],
                "sources": ", ".join(sources),
                "num_sources": len(sources),
                "context_precision": round(context_precision, 3),
                "context_recall": round(context_recall, 3),
                "faithfulness": round(faithfulness, 3),
                "answer_relevance": round(answer_relevance, 3),
                "correctness": round(correctness, 3),
                "length_compliance": round(length_compliance, 3)
            }
            results.append(result)
            
            # Print metrics
            print(f"  ├─ Context Precision: {context_precision:.3f}")
            print(f"  ├─ Context Recall: {context_recall:.3f}")
            print(f"  ├─ Faithfulness: {faithfulness:.3f}")
            print(f"  ├─ Answer Relevance: {answer_relevance:.3f}")
            print(f"  ├─ Correctness: {correctness:.3f}")
            print(f"  └─ Length Compliance: {length_compliance:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate average metrics
    print("\n" + "=" * 60)
    print("📈 OVERALL METRICS")
    print("=" * 60)
    
    if results:
        df = pd.DataFrame(results)
        
        avg_metrics = {
            "Context Precision": df["context_precision"].mean(),
            "Context Recall": df["context_recall"].mean(),
            "Faithfulness": df["faithfulness"].mean(),
            "Answer Relevance": df["answer_relevance"].mean(),
            "Correctness": df["correctness"].mean(),
            "Length Compliance": df["length_compliance"].mean()
        }
        
        for metric, value in avg_metrics.items():
            print(f"{metric:20s}: {value:.3f}")
        
        # Save results
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, "rag_evaluation_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\n💾 Detailed results saved to '{results_path}'")
        
        # Save summary
        summary_path = os.path.join(output_dir, "rag_evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "average_metrics": avg_metrics,
                "num_test_cases": len(EVAL_DATASET),
                "num_documents": len(TEST_DOCUMENTS),
                "timestamp": pd.Timestamp.now().isoformat()
            }, f, indent=2)
        print(f"💾 Summary saved to '{summary_path}'")
        
        return df, avg_metrics
    else:
        print("❌ No results to display!")
        return None, None

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results_df, metrics = run_evaluation()
    if results_df is not None:
        print("\n✅ Evaluation complete!")
    else:
        print("\n❌ Evaluation failed!")
