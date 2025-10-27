"""
Tests for MistakeMemory
"""

import pytest
import shutil
from datetime import datetime
from app.memory.mistake_memory import MistakeMemory
from dotenv import load_dotenv
load_dotenv() # Load environment variables for API keys, etc.
@pytest.fixture
def test_db_path(tmp_path):
    """Temporary database path for testing."""
    return str(tmp_path / "test_mistakes_db")


@pytest.fixture
def mistake_memory(test_db_path):
    """Create MistakeMemory instance with test database."""
    memory = MistakeMemory(persist_directory=test_db_path)
    yield memory
    # Cleanup after tests
    try:
        shutil.rmtree(test_db_path)
    except:
        pass


@pytest.fixture
def sample_mistake():
    """Sample enriched mistake."""
    return {
        "student_input": "I eat bananas with mie families",
        "corrected_answer": "I eat bananas with my family",
        "error_type": "spelling_and_grammar",
        "error_category": "mixed",
        "concepts": ["possessive_pronouns", "singular_plural"],
        "explanation": "Two errors: spelling of 'my' and plural/singular agreement",
        "difficulty": "beginner",
        "timestamp": datetime.now().isoformat(),
        "recurrence_risk": "medium",
        "searchable_text": "Student said 'I eat bananas with mie families' instead of 'I eat bananas with my family'. Error type: spelling and grammar."
    }


class TestMistakeMemory:
    
    def test_initialization(self, test_db_path):
        """Test memory initializes correctly."""
        memory = MistakeMemory(persist_directory=test_db_path)
        assert memory.vectorstore is not None
        assert memory.persist_directory == test_db_path
    
    def test_store_single_mistake(self, mistake_memory, sample_mistake):
        """Test storing a single mistake."""
        doc_id = mistake_memory.store_mistake(sample_mistake)
        
        assert doc_id is not None
        assert mistake_memory.count_mistakes() == 1
    
    def test_store_mistakes_batch(self, mistake_memory):
        """Test batch storage of multiple mistakes."""
        mistakes = [
            {
                "student_input": f"Mistake {i}",
                "corrected_answer": f"Correct {i}",
                "error_type": "grammar",
                "error_category": "grammar",
                "concepts": ["test"],
                "difficulty": "beginner",
                "timestamp": datetime.now().isoformat(),
                "recurrence_risk": "low",
                "searchable_text": f"Test mistake {i}"
            }
            for i in range(5)
        ]
        
        ids = mistake_memory.store_mistakes_batch(mistakes)
        
        assert len(ids) == 5
        assert mistake_memory.count_mistakes() == 5
    
    def test_retrieve_similar(self, mistake_memory, sample_mistake):
        """Test semantic similarity search."""
        # Store mistake
        mistake_memory.store_mistake(sample_mistake)
        
        # Search for similar
        results = mistake_memory.retrieve_similar("spelling mistakes", k=1)
        
        assert len(results) == 1
        assert results[0]["student_input"] == sample_mistake["student_input"]
    
    def test_retrieve_by_error_type(self, mistake_memory):
        """Test retrieval filtered by error type."""
        # Store mistakes with different types
        mistakes = [
            {
                "student_input": "Grammar error",
                "corrected_answer": "Corrected",
                "error_type": "grammar_conjugation",
                "error_category": "grammar",
                "concepts": ["test"],
                "difficulty": "beginner",
                "timestamp": datetime.now().isoformat(),
                "recurrence_risk": "low",
                "searchable_text": "Grammar error"
            },
            {
                "student_input": "Spelling error",
                "corrected_answer": "Corrected",
                "error_type": "spelling",
                "error_category": "spelling",
                "concepts": ["test"],
                "difficulty": "beginner",
                "timestamp": datetime.now().isoformat(),
                "recurrence_risk": "low",
                "searchable_text": "Spelling error"
            }
        ]
        
        mistake_memory.store_mistakes_batch(mistakes)
        
        # Retrieve only grammar errors
        grammar_errors = mistake_memory.retrieve_by_error_type("grammar_conjugation", k=10)
        
        assert len(grammar_errors) >= 1
        assert all(m["error_type"] == "grammar_conjugation" for m in grammar_errors)
    
    def test_get_all_mistakes(self, mistake_memory):
        """Test retrieving all mistakes."""
        # Store 3 mistakes
        mistakes = [
            {
                "student_input": f"Test {i}",
                "corrected_answer": f"Correct {i}",
                "error_type": "test",
                "error_category": "test",
                "concepts": ["test"],
                "difficulty": "beginner",
                "timestamp": datetime.now().isoformat(),
                "recurrence_risk": "low",
                "searchable_text": f"Test {i}"
            }
            for i in range(3)
        ]
        mistake_memory.store_mistakes_batch(mistakes)
        
        all_mistakes = mistake_memory.get_all_mistakes()
        
        assert len(all_mistakes) == 3
    
    def test_get_all_with_limit(self, mistake_memory):
        """Test get_all with limit parameter."""
        # Store 5 mistakes
        mistakes = [
            {
                "student_input": f"Test {i}",
                "corrected_answer": f"Correct {i}",
                "error_type": "test",
                "error_category": "test",
                "concepts": ["test"],
                "difficulty": "beginner",
                "timestamp": datetime.now().isoformat(),
                "recurrence_risk": "low",
                "searchable_text": f"Test {i}"
            }
            for i in range(5)
        ]
        mistake_memory.store_mistakes_batch(mistakes)
        
        limited = mistake_memory.get_all_mistakes(limit=2)
        
        assert len(limited) == 2
    
    def test_count_mistakes(self, mistake_memory, sample_mistake):
        """Test counting stored mistakes."""
        assert mistake_memory.count_mistakes() == 0
        
        mistake_memory.store_mistake(sample_mistake)
        assert mistake_memory.count_mistakes() == 1
        
        mistake_memory.store_mistake(sample_mistake)
        assert mistake_memory.count_mistakes() == 2
    
    def test_clear_all(self, mistake_memory, sample_mistake):
        """Test clearing all mistakes."""
        # Store mistakes
        mistake_memory.store_mistake(sample_mistake)
        assert mistake_memory.count_mistakes() > 0
        
        # Clear
        mistake_memory.clear_all()
        assert mistake_memory.count_mistakes() == 0
