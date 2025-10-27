"""
Tests for ConversationAnalyzer
"""

import pytest
from unittest.mock import Mock
from app.conversation.conversation_analyzer import ConversationAnalyzer


@pytest.fixture
def mock_llm():
    """Mock LLM with valid response."""
    llm = Mock()
    response = Mock()
    response.content = '''[
        {
            "message_index": 0,
            "student_input": "I eat bananas with mie families",
            "corrected_answer": "I eat bananas with my family",
            "error_type": "spelling_and_grammar",
            "error_category": "mixed",
            "concepts": ["possessive_pronouns", "singular_plural"],
            "explanation": "Two errors: 'mie' should be 'my', and 'families' should be singular 'family'.",
            "difficulty": "beginner",
            "suggested_practice": "Practice possessive pronouns and singular/plural nouns",
            "recurrence_risk": "medium"
        },
        {
            "message_index": 3,
            "student_input": "I leaving at 8am",
            "corrected_answer": "I leave at 8am",
            "error_type": "grammar_tense",
            "error_category": "grammar",
            "concepts": ["present_simple", "continuous_vs_simple"],
            "explanation": "Used present continuous instead of present simple for habitual action.",
            "difficulty": "beginner",
            "suggested_practice": "Review present simple for routines and habits",
            "recurrence_risk": "high"
        }
    ]'''
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def sample_conversation():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "I eat bananas with mie families"},
        {"role": "assistant", "content": "Almost! Try: 'I eat bananas with my family'"},
        {"role": "user", "content": "I eat bananas with my family"},
        {"role": "assistant", "content": "Perfect! What time do you wake up?"},
        {"role": "user", "content": "I leaving at 8am"},
        {"role": "assistant", "content": "Try: 'I leave at 8am' for habitual actions"}
    ]


class TestConversationAnalyzer:
    
    def test_initialization(self, mock_llm):
        """Test analyzer initializes correctly."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        assert analyzer.llm == mock_llm
    
    def test_analyze_conversation_success(self, mock_llm, sample_conversation):
        """Test successful conversation analysis."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        
        mistakes = analyzer.analyze_conversation(sample_conversation)
        
        # Should find 2 mistakes (indexes 0 and 3 - only user messages with errors)
        assert len(mistakes) == 2
        
        # Check first mistake
        assert mistakes[0]["student_input"] == "I eat bananas with mie families"
        assert mistakes[0]["corrected_answer"] == "I eat bananas with my family"
        assert mistakes[0]["error_type"] == "spelling_and_grammar"
        assert "possessive_pronouns" in mistakes[0]["concepts"]
        
        # Check second mistake
        assert mistakes[1]["student_input"] == "I leaving at 8am"
        assert mistakes[1]["error_type"] == "grammar_tense"
        
        # Verify metadata added
        assert all("timestamp" in m for m in mistakes)
        assert all("searchable_text" in m for m in mistakes)
    
    def test_analyze_empty_conversation(self, mock_llm):
        """Test handling empty conversation."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        mistakes = analyzer.analyze_conversation([])
        assert mistakes == []
    
    def test_analyze_no_user_messages(self, mock_llm):
        """Test conversation with no user messages."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        messages = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "assistant", "content": "How are you?"}
        ]
        mistakes = analyzer.analyze_conversation(messages)
        assert mistakes == []
    
    def test_parse_response_valid_json(self, mock_llm):
        """Test parsing valid JSON response."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        
        valid_json = '[{"message_index": 0, "student_input": "test"}]'
        result = analyzer._parse_response(valid_json)
        
        assert isinstance(result, list)
        assert len(result) == 1
    

    
    def test_parse_response_invalid_json(self, mock_llm):
        """Test handling invalid JSON."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        
        invalid = "not json"
        result = analyzer._parse_response(invalid)
        
        assert result == []
    
    def test_searchable_text_format(self, mock_llm, sample_conversation):
        """Test searchable text contains key information."""
        analyzer = ConversationAnalyzer(llm=mock_llm)
        
        mistakes = analyzer.analyze_conversation(sample_conversation)
        
        for mistake in mistakes:
            text = mistake["searchable_text"]
            assert mistake["student_input"] in text
            assert mistake["corrected_answer"] in text
            assert "Error type" in text
            assert "Concepts" in text
