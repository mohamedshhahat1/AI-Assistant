"""
DateTime Tool - Get current date, time, and day information.

Provides the current date and time in human-readable formats.
Handles queries about the current time, date, day of week, etc.
"""

from datetime import datetime
from typing import Dict, Optional

from .tool_registry import BaseTool


class DateTimeTool(BaseTool):
    """
    DateTime tool for answering questions about the current date and time.

    Supports queries like:
        - "what time is it" -> "It's 2:30 PM"
        - "what day is today" -> "Today is Friday, January 15, 2024"
        - "what is the date" -> "Today's date is January 15, 2024"
        - "what day of the week" -> "Today is Friday"
    """

    name = "datetime"
    description = "Get the current date, time, or day of the week"
    keywords = [
        "time", "date", "day", "today", "clock",
        "what time", "what day", "what date", "current time",
        "current date", "now", "tonight", "this morning"
    ]
    patterns = [
        r"what\s+(?:time|day|date)\s+is\s+it",
        r"what\s+is\s+(?:the\s+)?(?:current\s+)?(?:time|date|day)",
        r"what\s+(?:day|date)\s+is\s+(?:it\s+)?today",
        r"(?:tell\s+me\s+)?the\s+(?:current\s+)?(?:time|date|day)",
        r"what\s+day\s+of\s+the\s+week",
    ]

    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Return the current date/time information based on the query.

        Analyzes the message to determine if the user wants the time,
        date, day of week, or a combination.

        Args:
            message: The user's input message.
            user_id: Optional user identifier (not used).
            args: Optional additional arguments.

        Returns:
            Dictionary with formatted date/time result.
        """
        message_lower = message.lower().strip()
        now = datetime.now()

        # Determine what information the user wants
        if self._wants_time(message_lower):
            # Format time in 12-hour format
            time_str = now.strftime("%I:%M %p").lstrip("0")
            result = f"It's {time_str}"
        elif self._wants_day(message_lower):
            # Full day with date
            day_str = now.strftime("%A, %B %d, %Y")
            result = f"Today is {day_str}"
        elif self._wants_date(message_lower):
            # Just the date
            date_str = now.strftime("%B %d, %Y")
            result = f"Today's date is {date_str}"
        else:
            # Default: provide both date and time
            date_str = now.strftime("%A, %B %d, %Y")
            time_str = now.strftime("%I:%M %p").lstrip("0")
            result = f"It's {time_str} on {date_str}"

        return {
            "result": result,
            "tool_used": "datetime",
            "timestamp": now.isoformat()
        }

    def _wants_time(self, message: str) -> bool:
        """Check if the user is asking specifically about the time."""
        time_indicators = [
            "what time", "current time", "the time", "clock",
            "tell me the time", "time is it"
        ]
        return any(indicator in message for indicator in time_indicators)

    def _wants_day(self, message: str) -> bool:
        """Check if the user is asking about the day of the week."""
        day_indicators = [
            "what day", "which day", "day is it", "day is today",
            "day of the week"
        ]
        return any(indicator in message for indicator in day_indicators)

    def _wants_date(self, message: str) -> bool:
        """Check if the user is asking specifically about the date."""
        date_indicators = [
            "what date", "the date", "current date", "today's date",
            "date is it", "date today"
        ]
        return any(indicator in message for indicator in date_indicators)
