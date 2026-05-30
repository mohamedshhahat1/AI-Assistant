"""
Tool Registry - Base class and registry for all tools.

Provides the abstract BaseTool class that all tools must inherit from,
and the ToolRegistry that manages tool registration and matching.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Every tool must define:
        - name: unique identifier for the tool
        - description: human-readable description of what the tool does
        - keywords: list of words that trigger this tool
        - patterns: list of regex patterns that trigger this tool
        - execute(): the method that performs the tool's action
    """

    name: str = ""
    description: str = ""
    keywords: List[str] = []
    patterns: List[str] = []

    @abstractmethod
    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Execute the tool's action.

        Args:
            message: The user's input message that triggered this tool.
            user_id: Optional user identifier for user-specific operations.
            args: Optional additional arguments for the tool.

        Returns:
            A dictionary containing at minimum:
                - result: human-readable result string
                - tool_used: name of the tool that was used
        """
        pass


class ToolRegistry:
    """
    Registry that manages all available tools.

    Handles tool registration, lookup by name, listing all tools,
    and matching user messages to the appropriate tool based on
    keywords and regex patterns.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: An instance of a BaseTool subclass to register.
        """
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by its name.

        Args:
            name: The unique name of the tool.

        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all registered tools with their names and descriptions.

        Returns:
            A list of dictionaries with 'name' and 'description' keys.
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def match_tool(self, message: str) -> Optional[BaseTool]:
        """
        Find the best matching tool for a given user message.

        Matching is done in two passes:
        1. Regex pattern matching (higher priority, more specific)
        2. Keyword matching (fallback, counts keyword hits)

        Args:
            message: The user's input message.

        Returns:
            The best matching tool instance, or None if no tool matches.
        """
        message_lower = message.lower().strip()

        # First pass: try regex pattern matching (most specific)
        for tool in self._tools.values():
            for pattern in tool.patterns:
                if re.search(pattern, message_lower):
                    return tool

        # Second pass: keyword matching with scoring
        best_tool = None
        best_score = 0

        for tool in self._tools.values():
            score = 0
            for keyword in tool.keywords:
                # Check if the keyword appears in the message
                if keyword.lower() in message_lower:
                    # Longer keywords get higher scores (more specific)
                    score += len(keyword)

            if score > best_score:
                best_score = score
                best_tool = tool

        return best_tool
