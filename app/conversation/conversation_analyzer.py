"""
Conversation Analyzer - Analyzes completed conversations to extract mistakes.
Uses OpenAI Structured Outputs for 100% reliable JSON.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class MistakeSchema(BaseModel):
    """Pydantic model for a single mistake."""
    message_index: int = Field(description="Index of the message (1-based)")
    student_input: str = Field(description="Exact text student wrote")
    corrected_answer: str = Field(description="Correct version")
    error_type: str = Field(description="Type of error: grammar_conjugation, spelling, word_order, etc")
    error_category: str = Field(description="Category: grammar, vocabulary, or spelling")
    concepts: List[str] = Field(description="List of concepts involved")
    explanation: str = Field(description="Brief explanation (1-2 sentences)")
    difficulty: str = Field(description="beginner, intermediate, or advanced")
    suggested_practice: str = Field(description="Practice recommendation")
    recurrence_risk: str = Field(description="low, medium, or high")


class MistakesResponse(BaseModel):
    """Container for multiple mistakes."""
    mistakes: List[MistakeSchema] = Field(description="List of detected mistakes")


class ConversationAnalyzer:
    """
    Analyzes entire conversations to identify and enrich mistakes.
    Uses OpenAI Structured Outputs for guaranteed valid JSON.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize with LLM client."""
        if llm is None:
            from app.llm_client import llm as default_llm
            # Use gpt-4o-mini or gpt-4o for structured outputs
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )
        else:
            self.llm = llm
    
    def analyze_conversation(self, messages: List[Dict]) -> List[Dict]:
        """
        Batch analyze conversation to find and enrich mistakes.
        
        Args:
            messages: List of {role: "user"/"assistant", content: "..."} dicts
        
        Returns:
            List of enriched mistake dicts
        """
        # Extract only user messages
        user_messages = [
            {"index": i+1, "content": msg["content"]}
            for i, msg in enumerate(messages)
            if msg.get("role") == "user"
        ]
        
        if not user_messages:
            return []
        
        try:
            # Build prompt
            prompt = self._build_prompt(user_messages)
            
            # Use structured output with Pydantic
            structured_llm = self.llm.with_structured_output(MistakesResponse)
            
            # Get response
            response = structured_llm.invoke(prompt)
            
            # Convert to dict format
            mistakes = []
            for mistake in response.mistakes:
                mistake_dict = {
                    "message_index": mistake.message_index,
                    "student_input": mistake.student_input,
                    "corrected_answer": mistake.corrected_answer,
                    "error_type": mistake.error_type,
                    "error_category": mistake.error_category,
                    "concepts": mistake.concepts,
                    "explanation": mistake.explanation,
                    "difficulty": mistake.difficulty,
                    "suggested_practice": mistake.suggested_practice,
                    "recurrence_risk": mistake.recurrence_risk,
                    "timestamp": datetime.now().isoformat(),
                }
                mistake_dict["searchable_text"] = self._build_searchable_text(mistake_dict)
                mistakes.append(mistake_dict)
            
            return mistakes
        
        except Exception as e:
            print(f"Error analyzing conversation: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_prompt(self, user_messages: List[Dict]) -> List:
        """Build prompt for analysis."""
        messages_text = "\n".join([
            f"{msg['index']}. {msg['content']}"
            for msg in user_messages
        ])
        
        system_msg = """You are an expert English language error analyzer.
Identify grammatical and vocabulary errors in student messages.
Return ONLY messages that contain errors. Skip correct messages."""
        
        user_msg = f"""Analyze these English student messages for errors:

{messages_text}

For each message WITH errors:
- Identify the specific error
- Provide the correct version
- Explain the mistake pedagogically
- Classify the error type and difficulty

Skip messages with no errors."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", user_msg)
        ])
        
        return template.format_messages()
    
    def _build_searchable_text(self, mistake: Dict) -> str:
        """Build text for vector embedding."""
        student_input = mistake.get('student_input', '')
        corrected = mistake.get('corrected_answer', '')
        error_type = mistake.get('error_type', 'unknown').replace('_', ' ')
        explanation = mistake.get('explanation', '')
        concepts = ', '.join(mistake.get('concepts', []))
        difficulty = mistake.get('difficulty', 'unknown')
        
        return f"""Student said: "{student_input}" instead of "{corrected}".
Error type: {error_type}.
{explanation}
Concepts: {concepts}.
Difficulty: {difficulty}."""
