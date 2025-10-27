"""
Mistake Memory - Stores and retrieves enriched mistakes using Chroma vector DB.
"""

import os
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class MistakeMemory:
    """
    Persistent storage for language learning mistakes with semantic search.
    Uses Chroma for vector embeddings and retrieval.
    """
    
    def __init__(self, persist_directory: str = "data/mistakes_db"):
        """
        Initialize Mistake Memory with Chroma vector store.
        
        Args:
            persist_directory: Path to persist Chroma database
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        
        # Create directory if doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize or load Chroma
        self.vectorstore = Chroma(
            collection_name="mistakes",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def store_mistake(self, mistake: Dict) -> str:
        """
        Store a single enriched mistake.
        
        Args:
            mistake: Enriched mistake dict with all fields
        
        Returns:
            Document ID of stored mistake
        """
        # Create document from mistake
        doc = Document(
            page_content=mistake.get("searchable_text", ""),
            metadata={
                "student_input": mistake.get("student_input", ""),
                "corrected_answer": mistake.get("corrected_answer", ""),
                "error_type": mistake.get("error_type", ""),
                "error_category": mistake.get("error_category", ""),
                "difficulty": mistake.get("difficulty", ""),
                "timestamp": mistake.get("timestamp", ""),
                "concepts": ",".join(mistake.get("concepts", [])),
                "recurrence_risk": mistake.get("recurrence_risk", "")
            }
        )
        
        # Add to vectorstore
        ids = self.vectorstore.add_documents([doc])
        return ids[0] if ids else None
    
    def store_mistakes_batch(self, mistakes: List[Dict]) -> List[str]:
        """
        Store multiple mistakes at once (efficient).
        
        Args:
            mistakes: List of enriched mistake dicts
        
        Returns:
            List of document IDs
        """
        if not mistakes:
            return []
        
        docs = []
        for mistake in mistakes:
            doc = Document(
                page_content=mistake.get("searchable_text", ""),
                metadata={
                    "student_input": mistake.get("student_input", ""),
                    "corrected_answer": mistake.get("corrected_answer", ""),
                    "error_type": mistake.get("error_type", ""),
                    "error_category": mistake.get("error_category", ""),
                    "difficulty": mistake.get("difficulty", ""),
                    "timestamp": mistake.get("timestamp", ""),
                    "concepts": ",".join(mistake.get("concepts", [])),
                    "recurrence_risk": mistake.get("recurrence_risk", "")
                }
            )
            docs.append(doc)
        
        ids = self.vectorstore.add_documents(docs)
        return ids
    
    def retrieve_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve mistakes similar to query using semantic search.
        
        Args:
            query: Search query (e.g., "grammar mistakes")
            k: Number of results to return
        
        Returns:
            List of mistake dicts
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        return [self._doc_to_mistake(doc) for doc in docs]
    
    def retrieve_by_error_type(self, error_type: str, k: int = 10) -> List[Dict]:
        """
        Retrieve mistakes by specific error type.
        
        Args:
            error_type: Error type (e.g., "grammar_conjugation")
            k: Number of results
        
        Returns:
            List of mistakes matching error type
        """
        # Use metadata filter
        docs = self.vectorstore.similarity_search(
            query=error_type,
            k=k,
            filter={"error_type": error_type}
        )
        return [self._doc_to_mistake(doc) for doc in docs]
    
    def get_all_mistakes(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all stored mistakes.
        
        Args:
            limit: Maximum number to return (None = all)
        
        Returns:
            List of all mistakes
        """
        # Get all documents
        results = self.vectorstore.get()
        
        if not results or not results.get("documents"):
            return []
        
        mistakes = []
        docs = results["documents"]
        metadatas = results.get("metadatas", [])
        
        for i, doc_text in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) else {}
            mistake = {
                "searchable_text": doc_text,
                "student_input": metadata.get("student_input", ""),
                "corrected_answer": metadata.get("corrected_answer", ""),
                "error_type": metadata.get("error_type", ""),
                "error_category": metadata.get("error_category", ""),
                "difficulty": metadata.get("difficulty", ""),
                "timestamp": metadata.get("timestamp", ""),
                "concepts": metadata.get("concepts", "").split(",") if metadata.get("concepts") else [],
                "recurrence_risk": metadata.get("recurrence_risk", "")
            }
            mistakes.append(mistake)
        
        if limit:
            mistakes = mistakes[:limit]
        
        return mistakes
    
    def count_mistakes(self) -> int:
        """
        Count total mistakes stored.
        
        Returns:
            Number of mistakes
        """
        results = self.vectorstore.get()
        return len(results.get("documents", [])) if results else 0
    
    def clear_all(self):
        """
        Delete all mistakes (use with caution!).
        """
        # Delete collection
        self.vectorstore.delete_collection()
        
        # Recreate empty collection
        self.vectorstore = Chroma(
            collection_name="mistakes",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def _doc_to_mistake(self, doc: Document) -> Dict:
        """Convert Chroma document back to mistake dict."""
        metadata = doc.metadata
        return {
            "searchable_text": doc.page_content,
            "student_input": metadata.get("student_input", ""),
            "corrected_answer": metadata.get("corrected_answer", ""),
            "error_type": metadata.get("error_type", ""),
            "error_category": metadata.get("error_category", ""),
            "difficulty": metadata.get("difficulty", ""),
            "timestamp": metadata.get("timestamp", ""),
            "concepts": metadata.get("concepts", "").split(",") if metadata.get("concepts") else [],
            "recurrence_risk": metadata.get("recurrence_risk", "")
        }
