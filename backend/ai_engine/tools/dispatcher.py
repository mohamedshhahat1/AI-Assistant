"""
Tool Dispatcher - Single entry point for the tool execution system.

The ToolDispatcher is the main interface that the AI engine uses to
route user messages to the appropriate tool. It manages the registry,
initializes all tools, and handles dispatch logic.
"""

import sqlite3
from typing import Dict, List, Optional

from .tool_registry import ToolRegistry
from .calculator import CalculatorTool
from .notes import NotesTool
from .reminders import ReminderTool
from .datetime_tool import DateTimeTool
from .dictionary_tool import DictionaryTool


class ToolDispatcher:
    """
    Central dispatcher for the tool execution system.

    Acts as the single entry point for processing user messages through
    the tool system. It:
        1. Initializes the tool registry and all available tools
        2. Creates necessary database tables for persistent tools
        3. Routes messages to the matching tool
        4. Returns structured results or None if no tool matches

    Usage:
        dispatcher = ToolDispatcher(db_path="assistant.db")
        result = dispatcher.dispatch("what is 5 + 3", user_id="user123")
        # result = {"result": "5 + 3 = 8", "tool_used": "calculator", "answer": 8}

        result = dispatcher.dispatch("hello there")
        # result = None (no tool matched, use normal chat response)
    """

    def __init__(self, db_path: str = "ai_assistant.db"):
        """
        Initialize the ToolDispatcher.

        Creates the tool registry, registers all available tools, and
        ensures the SQLite database tables exist for tools that need
        persistent storage (notes, reminders).

        Args:
            db_path: Path to the SQLite database file for persistent tools.
        """
        self.db_path = db_path
        self.registry = ToolRegistry()

        # Ensure the database and tables exist
        self._init_database()

        # Register all available tools
        self._register_tools()

    def _init_database(self) -> None:
        """
        Initialize the SQLite database with all required tables.

        Creates tables for notes and reminders if they don't already exist.
        Uses check_same_thread=False for multi-threaded access.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Create notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create reminders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL,
                remind_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def _register_tools(self) -> None:
        """
        Register all available tools in the registry.

        Each tool is instantiated with the appropriate configuration
        (e.g., database path for persistent tools) and added to the registry.
        """
        # Calculator - no persistent storage needed
        self.registry.register(CalculatorTool())

        # Notes - uses SQLite for persistence
        self.registry.register(NotesTool(db_path=self.db_path))

        # Reminders - uses SQLite for persistence
        self.registry.register(ReminderTool(db_path=self.db_path))

        # DateTime - no persistent storage needed
        self.registry.register(DateTimeTool())

        # Dictionary - no persistent storage needed
        self.registry.register(DictionaryTool())

    def dispatch(self, message: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """
        Dispatch a user message to the appropriate tool.

        Attempts to match the message to a registered tool. If a match
        is found, the tool is executed and its result is returned.
        If no tool matches, returns None to indicate the message should
        be handled by normal chat/conversation logic.

        Args:
            message: The user's input message.
            user_id: Optional user identifier for user-specific operations.

        Returns:
            A dictionary with the tool's result if a tool matched,
            or None if no tool was appropriate for the message.
        """
        if not message or not message.strip():
            return None

        # Try to find a matching tool
        tool = self.registry.match_tool(message)

        if tool is None:
            return None

        # Execute the matched tool and return its result
        try:
            result = tool.execute(message, user_id=user_id)
            return result
        except Exception as e:
            # If a tool fails, return an error result rather than crashing
            return {
                "result": f"Tool '{tool.name}' encountered an error: {str(e)}",
                "tool_used": tool.name,
                "error": True
            }

    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Get information about all available tools.

        Returns a list of dictionaries describing each registered tool,
        useful for displaying capabilities to the user or for help text.

        Returns:
            A list of dictionaries with 'name' and 'description' keys
            for each registered tool.
        """
        return self.registry.list_tools()
