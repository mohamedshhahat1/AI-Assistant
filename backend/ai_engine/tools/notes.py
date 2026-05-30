"""
Notes Tool - Create, list, and delete personal notes.

Provides persistent note storage using SQLite. Each note is associated
with a user_id so users only see their own notes.
"""

import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from .tool_registry import BaseTool


class NotesTool(BaseTool):
    """
    Notes tool for creating, listing, and deleting user notes.

    Stores notes in a SQLite database with the following schema:
        - id: auto-incrementing primary key
        - user_id: identifies which user owns the note
        - title: short title for the note
        - content: full content of the note
        - created_at: timestamp when the note was created

    Supported actions:
        - "save note: buy milk" -> creates a new note
        - "note: meeting at 3pm" -> creates a new note
        - "show my notes" / "list notes" -> lists all user's notes
        - "delete note 1" / "remove note 3" -> deletes a specific note
    """

    name = "notes"
    description = "Create, list, and delete personal notes"
    keywords = [
        "note", "notes", "save note", "add note", "create note",
        "show notes", "list notes", "my notes", "delete note",
        "remove note", "write down", "remember this", "jot down"
    ]
    patterns = [
        r"(?:save|add|create|new)\s+(?:a\s+)?note[\s:]+.+",
        r"note[\s:]+.+",
        r"(?:show|list|display|get)\s+(?:my\s+)?notes",
        r"(?:delete|remove|erase)\s+note\s+\d+",
        r"(?:write\s+down|jot\s+down|remember\s+this)[\s:]+.+",
    ]

    def __init__(self, db_path: str = "ai_assistant.db"):
        """
        Initialize the Notes tool with a database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the notes table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Execute a notes action based on the user's message.

        Determines the action (create, list, or delete) from the message
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
        if self._is_delete_action(message_lower):
            return self._delete_note(message_lower, user_id)
        elif self._is_list_action(message_lower):
            return self._list_notes(user_id)
        else:
            return self._create_note(message, user_id)

    def _is_delete_action(self, message: str) -> bool:
        """Check if the message is requesting note deletion."""
        return bool(re.search(r"(?:delete|remove|erase)\s+note\s+\d+", message))

    def _is_list_action(self, message: str) -> bool:
        """Check if the message is requesting to list notes."""
        list_patterns = [
            r"(?:show|list|display|get|view)\s+(?:my\s+)?notes",
            r"my\s+notes",
            r"(?:all|the)\s+notes",
        ]
        return any(re.search(p, message) for p in list_patterns)

    def _create_note(self, message: str, user_id: str) -> Dict:
        """
        Create a new note from the user's message.

        Extracts the note content from the message by removing
        command prefixes like "save note:", "note:", etc.

        Args:
            message: The user's input containing the note content.
            user_id: The user's identifier.

        Returns:
            Dictionary confirming note creation.
        """
        # Extract note content by removing command prefixes
        content = re.sub(
            r"^(?:save|add|create|new|write\s+down|jot\s+down|remember\s+this)\s*(?:a\s+)?(?:note)?[\s:]*",
            "",
            message,
            flags=re.IGNORECASE
        ).strip()

        # If still has "note:" prefix, remove it
        content = re.sub(r"^note[\s:]+", "", content, flags=re.IGNORECASE).strip()

        if not content:
            return {
                "result": "Please provide content for the note.",
                "tool_used": "notes",
                "action": "error"
            }

        # Use first few words as title, full text as content
        title = content[:50] + ("..." if len(content) > 50 else "")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO notes (user_id, title, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, title, content, datetime.now().isoformat())
        )
        conn.commit()
        note_id = cursor.lastrowid
        conn.close()

        return {
            "result": f"Note saved: \"{content}\" (ID: {note_id})",
            "tool_used": "notes",
            "action": "create",
            "note_id": note_id
        }

    def _list_notes(self, user_id: str) -> Dict:
        """
        List all notes for a given user.

        Args:
            user_id: The user's identifier.

        Returns:
            Dictionary with formatted list of notes.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, content, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        notes = cursor.fetchall()
        conn.close()

        if not notes:
            return {
                "result": "You have no saved notes.",
                "tool_used": "notes",
                "action": "list",
                "notes": []
            }

        # Format notes for display
        note_list = []
        formatted_lines = []
        for note_id, title, content, created_at in notes:
            note_list.append({
                "id": note_id,
                "title": title,
                "content": content,
                "created_at": created_at
            })
            formatted_lines.append(f"  [{note_id}] {content}")

        result_text = f"Your notes ({len(notes)}):\n" + "\n".join(formatted_lines)

        return {
            "result": result_text,
            "tool_used": "notes",
            "action": "list",
            "notes": note_list
        }

    def _delete_note(self, message: str, user_id: str) -> Dict:
        """
        Delete a specific note by its ID.

        Args:
            message: The user's message containing the note ID to delete.
            user_id: The user's identifier.

        Returns:
            Dictionary confirming deletion or error if not found.
        """
        # Extract the note ID from the message
        id_match = re.search(r"(?:delete|remove|erase)\s+note\s+(\d+)", message)
        if not id_match:
            return {
                "result": "Please specify the note ID to delete (e.g., 'delete note 1').",
                "tool_used": "notes",
                "action": "error"
            }

        note_id = int(id_match.group(1))

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Check if note exists and belongs to user
        cursor.execute(
            "SELECT id, content FROM notes WHERE id = ? AND user_id = ?",
            (note_id, user_id)
        )
        note = cursor.fetchone()

        if not note:
            conn.close()
            return {
                "result": f"Note {note_id} not found.",
                "tool_used": "notes",
                "action": "error"
            }

        # Delete the note
        cursor.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
        conn.commit()
        conn.close()

        return {
            "result": f"Note {note_id} deleted successfully.",
            "tool_used": "notes",
            "action": "delete",
            "deleted_id": note_id
        }
