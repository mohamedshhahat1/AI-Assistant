"""
Reminders Tool - Set, list, and complete reminders.

Provides persistent reminder storage using SQLite. Each reminder
is associated with a user_id and can be marked as completed.
"""

import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional

from .tool_registry import BaseTool


class ReminderTool(BaseTool):
    """
    Reminder tool for setting, listing, and completing reminders.

    Stores reminders in a SQLite database with the following schema:
        - id: auto-incrementing primary key
        - user_id: identifies which user owns the reminder
        - content: what to be reminded about
        - remind_at: when to remind (timestamp or None)
        - created_at: timestamp when the reminder was created
        - completed: boolean flag (0 or 1)

    Supported actions:
        - "remind me to call mom" -> creates a reminder
        - "set reminder: buy groceries" -> creates a reminder
        - "show reminders" / "list reminders" -> lists active reminders
        - "complete reminder 1" / "done reminder 2" -> marks as completed
    """

    name = "reminders"
    description = "Set, list, and complete reminders"
    keywords = [
        "remind", "reminder", "reminders", "set reminder",
        "show reminders", "list reminders", "my reminders",
        "complete reminder", "done reminder", "finish reminder",
        "remind me", "don't forget", "dont forget"
    ]
    patterns = [
        r"remind\s+me\s+(?:to\s+)?.+",
        r"(?:set|create|add|new)\s+(?:a\s+)?reminder[\s:]+.+",
        r"(?:show|list|display|get|view)\s+(?:my\s+)?reminders",
        r"(?:complete|done|finish|mark)\s+reminder\s+\d+",
        r"don'?t\s+(?:let\s+me\s+)?forget\s+(?:to\s+)?.+",
    ]

    def __init__(self, db_path: str = "ai_assistant.db"):
        """
        Initialize the Reminders tool with a database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the reminders table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
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

    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Execute a reminder action based on the user's message.

        Determines the action (create, list, or complete) from the message
        and performs the corresponding database operation.

        Args:
            message: The user's input message.
            user_id: Identifier for the user (defaults to 'default').
            args: Optional additional arguments.

        Returns:
            Dictionary with result, tool_used, and action performed.
        """
        if user_id is None:
            user_id = "default"

        message_lower = message.lower().strip()

        # Determine the action
        if self._is_complete_action(message_lower):
            return self._complete_reminder(message_lower, user_id)
        elif self._is_list_action(message_lower):
            return self._list_reminders(user_id)
        else:
            return self._create_reminder(message, user_id)

    def _is_complete_action(self, message: str) -> bool:
        """Check if the message is requesting to complete a reminder."""
        return bool(re.search(r"(?:complete|done|finish|mark)\s+reminder\s+\d+", message))

    def _is_list_action(self, message: str) -> bool:
        """Check if the message is requesting to list reminders."""
        list_patterns = [
            r"(?:show|list|display|get|view)\s+(?:my\s+)?reminders",
            r"my\s+reminders",
            r"(?:all|the|active)\s+reminders",
        ]
        return any(re.search(p, message) for p in list_patterns)

    def _create_reminder(self, message: str, user_id: str) -> Dict:
        """
        Create a new reminder from the user's message.

        Extracts the reminder content by removing command prefixes
        like "remind me to", "set reminder:", etc.

        Args:
            message: The user's input containing the reminder content.
            user_id: The user's identifier.

        Returns:
            Dictionary confirming reminder creation.
        """
        # Extract reminder content by removing command prefixes
        content = re.sub(
            r"^(?:remind\s+me\s+(?:to\s+)?|"
            r"(?:set|create|add|new)\s+(?:a\s+)?reminder[\s:]*|"
            r"don'?t\s+(?:let\s+me\s+)?forget\s+(?:to\s+)?)",
            "",
            message,
            flags=re.IGNORECASE
        ).strip()

        if not content:
            return {
                "result": "Please provide content for the reminder.",
                "tool_used": "reminders",
                "action": "error"
            }

        # Try to extract a time reference (basic parsing)
        remind_at = self._parse_time(content)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (user_id, content, remind_at, created_at, completed) VALUES (?, ?, ?, ?, 0)",
            (user_id, content, remind_at, datetime.now().isoformat())
        )
        conn.commit()
        reminder_id = cursor.lastrowid
        conn.close()

        result_text = f"Reminder set: \"{content}\" (ID: {reminder_id})"
        if remind_at:
            result_text += f" - Scheduled for: {remind_at}"

        return {
            "result": result_text,
            "tool_used": "reminders",
            "action": "create",
            "reminder_id": reminder_id
        }

    def _list_reminders(self, user_id: str) -> Dict:
        """
        List all active (non-completed) reminders for a given user.

        Args:
            user_id: The user's identifier.

        Returns:
            Dictionary with formatted list of active reminders.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, remind_at, created_at, completed FROM reminders "
            "WHERE user_id = ? AND completed = 0 ORDER BY created_at DESC",
            (user_id,)
        )
        reminders = cursor.fetchall()
        conn.close()

        if not reminders:
            return {
                "result": "You have no active reminders.",
                "tool_used": "reminders",
                "action": "list",
                "reminders": []
            }

        # Format reminders for display
        reminder_list = []
        formatted_lines = []
        for rem_id, content, remind_at, created_at, completed in reminders:
            reminder_list.append({
                "id": rem_id,
                "content": content,
                "remind_at": remind_at,
                "created_at": created_at,
                "completed": bool(completed)
            })
            line = f"  [{rem_id}] {content}"
            if remind_at:
                line += f" (due: {remind_at})"
            formatted_lines.append(line)

        result_text = f"Active reminders ({len(reminders)}):\n" + "\n".join(formatted_lines)

        return {
            "result": result_text,
            "tool_used": "reminders",
            "action": "list",
            "reminders": reminder_list
        }

    def _complete_reminder(self, message: str, user_id: str) -> Dict:
        """
        Mark a specific reminder as completed.

        Args:
            message: The user's message containing the reminder ID.
            user_id: The user's identifier.

        Returns:
            Dictionary confirming completion or error if not found.
        """
        # Extract the reminder ID from the message
        id_match = re.search(r"(?:complete|done|finish|mark)\s+reminder\s+(\d+)", message)
        if not id_match:
            return {
                "result": "Please specify the reminder ID to complete (e.g., 'complete reminder 1').",
                "tool_used": "reminders",
                "action": "error"
            }

        reminder_id = int(id_match.group(1))

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Check if reminder exists and belongs to user
        cursor.execute(
            "SELECT id, content FROM reminders WHERE id = ? AND user_id = ? AND completed = 0",
            (reminder_id, user_id)
        )
        reminder = cursor.fetchone()

        if not reminder:
            conn.close()
            return {
                "result": f"Reminder {reminder_id} not found or already completed.",
                "tool_used": "reminders",
                "action": "error"
            }

        # Mark as completed
        cursor.execute(
            "UPDATE reminders SET completed = 1 WHERE id = ? AND user_id = ?",
            (reminder_id, user_id)
        )
        conn.commit()
        conn.close()

        return {
            "result": f"Reminder {reminder_id} completed: \"{reminder[1]}\"",
            "tool_used": "reminders",
            "action": "complete",
            "completed_id": reminder_id
        }

    def _parse_time(self, content: str) -> Optional[str]:
        """
        Attempt to parse a time reference from the reminder content.

        This is a basic implementation that handles common patterns.
        A production version would use a library like dateparser.

        Args:
            content: The reminder content text.

        Returns:
            An ISO format timestamp string, or None if no time found.
        """
        # Look for "at HH:MM" patterns
        time_match = re.search(r"at\s+(\d{1,2}):(\d{2})\s*(am|pm)?", content.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            period = time_match.group(3)

            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0

            now = datetime.now()
            remind_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If the time has already passed today, set for tomorrow
            if remind_time < now:
                from datetime import timedelta
                remind_time += timedelta(days=1)

            return remind_time.isoformat()

        return None
