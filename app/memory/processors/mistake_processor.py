"""
Mistake Processor - Enriches raw language learning mistakes with LLM analysis.

Transforms raw student errors into structured, searchable data with pedagogical insights.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class MistakeProcessor:
    """
    Processes language learning mistakes to extract error patterns,
    generate explanations, and suggest remediation strategies.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize the mistake processor.
        
        Args:
            llm: Optional ChatOpenAI instance. If None, creates default instance.
        """
        if llm is None:
            from app.llm_client import llm as default_llm
            self.llm = default_llm
        else:
            self.llm = llm
    
    def process_single_mistake(self, raw_mistake: Dict) -> Dict:
        """
        Process a single mistake with LLM enrichment.
        
        Args:
            raw_mistake: Dict with keys:
                - student_input (str): What student wrote
                - correct_answer (str): Correct version
                - timestamp (str): ISO format timestamp
                - conversation_context (List[Dict], optional): Recent messages
                - topic (str, optional): Topic being practiced
                - language_pair (str, optional): e.g., "en->es"
        
        Returns:
            Enriched mistake dict with LLM-generated analysis
        """
        prompt = self._build_enrichment_prompt(raw_mistake)
        
        try:
            response = self.llm.invoke(prompt)
            enriched_data = self._parse_llm_response(response.content)
        except Exception as e:
            print(f"Warning: LLM enrichment failed: {e}. Using defaults.")
            enriched_data = self._get_default_enrichment()
        
        # Merge original data with enrichment
        enriched_mistake = {
            **raw_mistake,
            **enriched_data,
            "searchable_text": self._build_searchable_text(raw_mistake, enriched_data)
        }
        
        return enriched_mistake
    
    def process_batch_mistakes(self, raw_mistakes: List[Dict]) -> List[Dict]:
        """
        Process multiple mistakes in a single LLM call (cost-efficient).
        
        Args:
            raw_mistakes: List of raw mistake dicts
        
        Returns:
            List of enriched mistake dicts
        """
        if not raw_mistakes:
            return []
        
        prompt = self._build_batch_enrichment_prompt(raw_mistakes)
        
        try:
            response = self.llm.invoke(prompt)
            enriched_list = self._parse_batch_llm_response(response.content, len(raw_mistakes))
        except Exception as e:
            print(f"Warning: Batch LLM enrichment failed: {e}. Using defaults.")
            enriched_list = [self._get_default_enrichment() for _ in raw_mistakes]
        
        # Merge original data with enrichments
        result = []
        for raw, enriched_data in zip(raw_mistakes, enriched_list):
            enriched_mistake = {
                **raw,
                **enriched_data,
                "searchable_text": self._build_searchable_text(raw, enriched_data)
            }
            result.append(enriched_mistake)
        
        return result
    
    def extract_error_pattern(self, mistakes: List[Dict]) -> Dict:
        """
        Analyze multiple mistakes to detect recurring patterns.
        
        Args:
            mistakes: List of enriched mistake dicts
        
        Returns:
            Pattern analysis dict with most common error and recommendations
        """
        if not mistakes:
            return {
                "most_common_error_type": "none",
                "frequency": 0,
                "pattern_description": "No mistakes to analyze",
                "confidence": 0.0,
                "recommendation": "Continue practicing"
            }
        
        # Count error types
        error_counts = {}
        for mistake in mistakes:
            error_type = mistake.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Find most common
        most_common = max(error_counts, key=error_counts.get)
        frequency = error_counts[most_common]
        confidence = frequency / len(mistakes)
        
        # Generate description
        sample_mistake = next(m for m in mistakes if m.get("error_type") == most_common)
        
        return {
            "most_common_error_type": most_common,
            "frequency": frequency,
            "pattern_description": f"Student struggles with {most_common.replace('_', ' ')}",
            "confidence": round(confidence, 2),
            "recommendation": sample_mistake.get("suggested_practice", "Review this topic")
        }
    
    # Private helper methods
    
    def _build_enrichment_prompt(self, raw_mistake: Dict) -> str:
        """Build LLM prompt for single mistake analysis."""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert language learning analyst. 
            Analyze mistakes and extract pedagogical insights.
            Return ONLY valid JSON, no markdown, no explanations."""),
            ("user", """Analyze this language learning mistake:

            Student wrote: "{student_input}"
            Correct answer: "{correct_answer}"
            Topic: {topic}

            Extract and return JSON with these fields:
            {{
                "error_type": "one of: grammar_conjugation, grammar_agreement, vocabulary_choice, spelling, word_order, article_usage, preposition_usage, other",
                "error_category": "one of: grammar, vocabulary, spelling, pronunciation, syntax",
                "concepts": ["list of grammatical concepts involved"],
                "explanation": "pedagogical explanation for student (1-2 sentences)",
                "difficulty": "one of: beginner, intermediate, advanced",
                "suggested_practice": "specific practice recommendation",
                "recurrence_risk": "one of: low, medium, high",
                "related_concepts": ["list of related topics to review"]
            }}

            Return ONLY the JSON object.""")
                    ])
        
        return template.format_messages(
            student_input=raw_mistake.get("student_input", ""),
            correct_answer=raw_mistake.get("correct_answer", ""),
            topic=raw_mistake.get("topic", "general practice")
        )
    
    def _build_batch_enrichment_prompt(self, raw_mistakes: List[Dict]) -> str:
        """Build LLM prompt for batch mistake analysis."""
        mistakes_text = "\n".join([
            f"{i+1}. Student: \"{m.get('student_input', '')}\" | Correct: \"{m.get('correct_answer', '')}\""
            for i, m in enumerate(raw_mistakes)
        ])
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert language learning analyst. Return ONLY valid JSON array."),
            ("user", f"""Analyze these {len(raw_mistakes)} mistakes. Return JSON array with one object per mistake.

            Mistakes:
            {mistakes_text}

            For each, extract:
            - error_type, error_category, concepts, explanation, difficulty, suggested_practice, recurrence_risk, related_concepts

            Return JSON array of {len(raw_mistakes)} objects.""")
                    ])
        
        return template.format_messages()
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse and validate LLM JSON response."""
        try:
            # Remove markdown if present
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response.strip())
            
            # Validate required fields
            required = ["error_type", "error_category", "concepts", "explanation", 
                       "difficulty", "suggested_practice", "recurrence_risk", "related_concepts"]
            
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            return self._get_default_enrichment()
    
    def _parse_batch_llm_response(self, response: str, expected_count: int) -> List[Dict]:
        """Parse batch LLM response into list of dicts."""
        try:
            # Remove markdown
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response.strip())
            
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")
            
            if len(data) != expected_count:
                print(f"Warning: Expected {expected_count} items, got {len(data)}")
            
            return data
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse batch LLM response: {e}")
            return [self._get_default_enrichment() for _ in range(expected_count)]
    
    def _get_default_enrichment(self) -> Dict:
        """Return default enrichment when LLM fails."""
        return {
            "error_type": "unknown",
            "error_category": "unknown",
            "concepts": ["general"],
            "explanation": "Error analysis unavailable. Please review the topic.",
            "difficulty": "beginner",
            "suggested_practice": "Review topic and try similar exercises",
            "recurrence_risk": "low",
            "related_concepts": []
        }
    
    def _build_searchable_text(self, raw: Dict, enriched: Dict) -> str:
        """Build searchable text for vector embedding."""
        return f"""Student wrote: "{raw.get('student_input', '')}" instead of "{raw.get('correct_answer', '')}".
        Error type: {enriched.get('error_type', 'unknown').replace('_', ' ')}.
        {enriched.get('explanation', '')}
        Concepts: {', '.join(enriched.get('concepts', []))}.
        """
