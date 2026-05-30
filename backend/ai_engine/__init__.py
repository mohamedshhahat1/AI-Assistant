"""
AI Engine Package
=================
Core AI components for the AI Assistant:
- IntentDetector: Classifies user messages into intents
- ResponseEngine: Generates appropriate responses (Hybrid Mode)
- HybridEngine: Core decision engine (embeddings + intent + fallback)
- MemorySystem: Remembers users across conversations
- Analytics: Provides usage analytics and reporting
- ToolDispatcher: Routes messages to tool handlers (calculator, notes, reminders, etc.)
"""

from .intent_model import IntentDetector
from .response_engine import ResponseEngine
from .hybrid_engine import HybridEngine
from .memory import MemorySystem
from .analytics import Analytics
from .tools import ToolDispatcher
