"""
Short Term Memory - Stores current conversation session in RAM.
"""

from collections import deque
from typing import List, Dict


class ShortTermMemory:
    """
    In-memory storage for current conversation.
    Lost when session ends (that's when we analyze and store in LTM).
    """
    
    def __init__(self, max_messages: int = 50):
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum messages to keep in memory
        """
        self.messages = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def get_messages(self, last_n: int = None) -> List[Dict]:
        """
        Get conversation messages.
        
        Args:
            last_n: Get only last N messages (None = all)
        
        Returns:
            List of message dicts
        """
        if last_n:
            return list(self.messages)[-last_n:]
        return list(self.messages)
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
    
    def count(self) -> int:
        """Get message count."""
        return len(self.messages)
