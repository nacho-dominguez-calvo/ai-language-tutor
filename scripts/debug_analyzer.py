"""
Debug script to test conversation analyzer directly
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from app.conversation.conversation_analyzer import ConversationAnalyzer
from app.memory.mistake_memory import MistakeMemory

load_dotenv()

def test_analyzer():
    """Test the analyzer with sample conversation."""
    
    print("\n" + "="*50)
    print("🔍 TESTING CONVERSATION ANALYZER")
    print("="*50 + "\n")
    
    # Sample conversation with intentional mistakes
    messages = [
        {"role": "user", "content": "Hello, I want to practice English"},
        {"role": "assistant", "content": "Great! Let's practice."},
        {"role": "user", "content": "I eat bananas with mie families"},
        {"role": "assistant", "content": "Almost! Try: 'I eat bananas with my family'"},
        {"role": "user", "content": "She go to work every day"},
        {"role": "assistant", "content": "Almost! Try: 'She goes to work every day'"},
        {"role": "user", "content": "I have went there yesterday"},
        {"role": "assistant", "content": "Try: 'I went there yesterday'"}
    ]
    
    print("📝 INPUT MESSAGES:")
    for i, msg in enumerate(messages):
        print(f"  {i+1}. [{msg['role']}] {msg['content']}")
    
    print("\n🔄 Analyzing conversation...")
    
    # Initialize analyzer
    analyzer = ConversationAnalyzer()
    
    # Analyze
    mistakes = analyzer.analyze_conversation(messages)
    
    print(f"\n✅ ANALYSIS COMPLETE: Found {len(mistakes)} mistakes\n")
    
    if mistakes:
        print("📋 DETECTED MISTAKES:")
        print("-" * 50)
        for i, mistake in enumerate(mistakes, 1):
            print(f"\n{i}. ERROR TYPE: {mistake.get('error_type', 'unknown')}")
            print(f"   Student said: \"{mistake.get('student_input', '')}\"")
            print(f"   Correct: \"{mistake.get('corrected_answer', '')}\"")
            print(f"   Explanation: {mistake.get('explanation', '')}")
            print(f"   Concepts: {', '.join(mistake.get('concepts', []))}")
            print(f"   Difficulty: {mistake.get('difficulty', '')}")
    else:
        print("⚠️ WARNING: No mistakes detected!")
        print("This should have found 3 mistakes.")
    
    print("\n" + "="*50)
    print("🧪 TEST COMPLETE")
    print("="*50 + "\n")
    
    return mistakes


def test_storage():
    """Test storing mistakes."""
    print("\n📦 TESTING MISTAKE STORAGE")
    print("-" * 50)
    
    analyzer = ConversationAnalyzer()
    mistake_memory = MistakeMemory()
    
    # Test conversation
    messages = [
        {"role": "user", "content": "I eat bananas with mie families"},
        {"role": "assistant", "content": "Almost! Try: 'my family'"}
    ]
    
    mistakes = analyzer.analyze_conversation(messages)
    
    if mistakes:
        print(f"✅ Storing {len(mistakes)} mistakes...")
        mistake_memory.store_mistakes_batch(mistakes)
        
        # Verify storage
        stored = mistake_memory.get_all_mistakes()
        print(f"✅ Verified: {len(stored)} mistakes in database")
    else:
        print("❌ No mistakes to store")
    
    print("-" * 50 + "\n")


if __name__ == "__main__":
    # Test analyzer
    mistakes = test_analyzer()
    
    # If mistakes found, test storage
    if mistakes:
        test_storage()
    else:
        print("⚠️ DEBUGGING NEEDED: Analyzer not detecting mistakes correctly")
