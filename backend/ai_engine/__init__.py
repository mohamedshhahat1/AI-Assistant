"""
AI Engine Package
=================
Core AI components for the AI Assistant:
- IntentDetector: Classifies user messages into intents
- ResponseEngine: Generates appropriate responses (Hybrid Mode)
- HybridEngine: Core decision engine (RAG + intent + fallback)
- RAGEngine: Retrieval Augmented Generation engine (dynamic, self-learning knowledge)
- MemorySystem: Remembers users across conversations
- Analytics: Provides usage analytics and reporting
- ToolDispatcher: Routes messages to tool handlers (calculator, notes, reminders, etc.)
"""

from .intent_model import IntentDetector
from .response_engine import ResponseEngine
from .hybrid_engine import HybridEngine
from .rag_engine import RAGEngine
from .memory import MemorySystem
from .analytics import Analytics
from .tools import ToolDispatcher
