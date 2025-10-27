"""
Unit tests for MistakeProcessor
"""

import pytest
from unittest.mock import Mock, MagicMock
from app.memory.processors.mistake_processor import MistakeProcessor


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid JSON."""
    llm = Mock()
    response = Mock()
    response.content = '''{
        "error_type": "grammar_conjugation",
        "error_category": "grammar",
        "concepts": ["present_simple", "third_person_singular"],
        "explanation": "Student forgot to add -s ending for third person.",
        "difficulty": "beginner",
        "suggested_practice": "Practice third-person verb endings",
        "recurrence_risk": "medium",
        "related_concepts": ["subject_verb_agreement"]
    }'''
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def sample_raw_mistake():
    """Sample raw mistake for testing."""
    return {
        "student_input": "She go to work",
        "correct_answer": "She goes to work",
        "timestamp": "2025-10-23T14:23:00",
        "topic": "present_simple"
    }


class TestMistakeProcessor:
    
    def test_initialization_with_llm(self, mock_llm):
        """Test processor initializes with provided LLM."""
        processor = MistakeProcessor(llm=mock_llm)
        assert processor.llm == mock_llm
    
    def test_initialization_without_llm(self):
        """Test processor creates default LLM."""
        processor = MistakeProcessor()
        assert processor.llm is not None
    
    def test_process_single_mistake_success(self, mock_llm, sample_raw_mistake):
        """Test successful single mistake processing."""
        processor = MistakeProcessor(llm=mock_llm)
        
        enriched = processor.process_single_mistake(sample_raw_mistake)
        
        # Check original data preserved
        assert enriched["student_input"] == "She go to work"
        assert enriched["correct_answer"] == "She goes to work"
        
        # Check enrichment added
        assert enriched["error_type"] == "grammar_conjugation"
        assert enriched["error_category"] == "grammar"
        assert "present_simple" in enriched["concepts"]
        assert len(enriched["explanation"]) > 0
        assert enriched["difficulty"] in ["beginner", "intermediate", "advanced"]
        
        # Check searchable text generated
        assert "searchable_text" in enriched
        assert "She go to work" in enriched["searchable_text"]
        
        # Verify LLM was called
        mock_llm.invoke.assert_called_once()
    
    def test_process_single_mistake_llm_failure(self, sample_raw_mistake):
        """Test processor handles LLM failure gracefully."""
        # Mock LLM that raises exception
        failing_llm = Mock()
        failing_llm.invoke.side_effect = Exception("LLM API error")
        
        processor = MistakeProcessor(llm=failing_llm)
        enriched = processor.process_single_mistake(sample_raw_mistake)
        
        # Should return default enrichment
        assert enriched["error_type"] == "unknown"
        assert enriched["error_category"] == "unknown"
        assert "searchable_text" in enriched
    
    def test_process_batch_mistakes_success(self, mock_llm):
        """Test batch processing of multiple mistakes."""
        # Mock batch response
        batch_response = Mock()
        batch_response.content = '''[
            {
                "error_type": "grammar_conjugation",
                "error_category": "grammar",
                "concepts": ["present_simple"],
                "explanation": "Forgot -s ending",
                "difficulty": "beginner",
                "suggested_practice": "Practice endings",
                "recurrence_risk": "medium",
                "related_concepts": []
            },
            {
                "error_type": "grammar_tense",
                "error_category": "grammar",
                "concepts": ["past_simple"],
                "explanation": "Used wrong past tense form",
                "difficulty": "beginner",
                "suggested_practice": "Review irregular verbs",
                "recurrence_risk": "high",
                "related_concepts": ["irregular_verbs"]
            }
        ]'''
        mock_llm.invoke.return_value = batch_response
        
        processor = MistakeProcessor(llm=mock_llm)
        
        raw_mistakes = [
            {
                "student_input": "She go",
                "correct_answer": "She goes",
                "timestamp": "2025-10-23T14:23:00"
            },
            {
                "student_input": "I goed",
                "correct_answer": "I went",
                "timestamp": "2025-10-23T14:25:00"
            }
        ]
        
        enriched_list = processor.process_batch_mistakes(raw_mistakes)
        
        assert len(enriched_list) == 2
        assert enriched_list[0]["error_type"] == "grammar_conjugation"
        assert enriched_list[1]["error_type"] == "grammar_tense"
        assert all("searchable_text" in m for m in enriched_list)
    
    def test_process_batch_empty_list(self, mock_llm):
        """Test batch processing with empty list."""
        processor = MistakeProcessor(llm=mock_llm)
        result = processor.process_batch_mistakes([])
        assert result == []
    
    def test_extract_error_pattern_multiple_mistakes(self):
        """Test pattern extraction from multiple mistakes."""
        processor = MistakeProcessor()
        
        mistakes = [
            {"error_type": "grammar_conjugation", "suggested_practice": "Practice verbs"},
            {"error_type": "grammar_conjugation", "suggested_practice": "Practice verbs"},
            {"error_type": "grammar_conjugation", "suggested_practice": "Practice verbs"},
            {"error_type": "vocabulary_choice", "suggested_practice": "Learn words"},
        ]
        
        pattern = processor.extract_error_pattern(mistakes)
        
        assert pattern["most_common_error_type"] == "grammar_conjugation"
        assert pattern["frequency"] == 3
        assert pattern["confidence"] == 0.75  # 3/4
        assert len(pattern["pattern_description"]) > 0
    
    def test_extract_error_pattern_empty_list(self):
        """Test pattern extraction with no mistakes."""
        processor = MistakeProcessor()
        pattern = processor.extract_error_pattern([])
        
        assert pattern["most_common_error_type"] == "none"
        assert pattern["frequency"] == 0
        assert pattern["confidence"] == 0.0
    
    def test_parse_llm_response_valid_json(self, mock_llm):
        """Test parsing valid JSON from LLM."""
        processor = MistakeProcessor(llm=mock_llm)
        
        valid_json = '{"error_type": "grammar", "error_category": "grammar", "concepts": [], "explanation": "test", "difficulty": "beginner", "suggested_practice": "test", "recurrence_risk": "low", "related_concepts": []}'
        
        parsed = processor._parse_llm_response(valid_json)
        assert parsed["error_type"] == "grammar"
    
    def test_parse_llm_response_markdown_wrapped(self, mock_llm):
        """Test parsing JSON wrapped in markdown code blocks."""
        processor = MistakeProcessor(llm=mock_llm)
        
        markdown_json = '''```
{"error_type": "grammar", "error_category": "grammar", "concepts": [], "explanation": "test", "difficulty": "beginner", "suggested_practice": "test", "recurrence_risk": "low", "related_concepts": []}
```'''
        
        parsed = processor._parse_llm_response(markdown_json)
        assert parsed["error_type"] == "grammar"
    
    def test_parse_llm_response_invalid_json(self, mock_llm):
        """Test handling of invalid JSON."""
        processor = MistakeProcessor(llm=mock_llm)
        
        invalid = "This is not JSON"
        parsed = processor._parse_llm_response(invalid)
        
        # Should return defaults
        assert parsed["error_type"] == "unknown"
        assert parsed["error_category"] == "unknown"
    
    def test_searchable_text_format(self, mock_llm, sample_raw_mistake):
        """Test searchable text contains key information."""
        processor = MistakeProcessor(llm=mock_llm)
        enriched = processor.process_single_mistake(sample_raw_mistake)
        
        searchable = enriched["searchable_text"]
        
        assert "She go to work" in searchable
        assert "She goes to work" in searchable
        assert "Error type" in searchable
        assert "Concepts" in searchable
