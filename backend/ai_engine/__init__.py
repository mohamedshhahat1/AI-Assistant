"""
AI Engine Package
=================
Core AI components for the AI Assistant:
- IntentDetector: Classifies user messages into intents
- ResponseEngine: Generates appropriate responses
- MemorySystem: Remembers users across conversations
"""

from .intent_model import IntentDetector
from .response_engine import ResponseEngine
from .memory import MemorySystem
