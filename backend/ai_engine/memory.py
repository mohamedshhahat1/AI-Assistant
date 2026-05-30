"""
Memory System for AI Assistant
===============================
This module handles persistence and memory - it remembers users across conversations.
It uses SQLite to store user profiles, conversation history, and preferences.

Usage:
    memory = MemorySystem()
    memory.save_interaction("user123", "Hello!", "Hi there!", intent="greeting")
    context = memory.get_context("user123")
"""

import sqlite3
import json
import os
import re
from datetime import datetime


class MemorySystem:
    """
    Manages long-term memory for the AI Assistant.
    Stores user profiles, conversation history, and preferences in SQLite.
    Thread-safe for use in web servers.
    """

    def __init__(self, db_path="backend/data/memory.db"):
        """
        Initialize the memory system with a SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
                           Defaults to 'backend/data/memory.db'.
        """
        # Create the data directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Connect to database with thread-safety enabled
        # check_same_thread=False allows multiple threads to use this connection
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Create tables if they don't exist
        self._create_tables()

    def _create_tables(self):
        """Create the database tables if they don't already exist."""
        cursor = self.conn.cursor()

        # Users table - stores user profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                preferences TEXT
            )
        """)

        # History table - stores conversation messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                message TEXT,
                intent TEXT,
                timestamp TEXT
            )
        """)

        self.conn.commit()

    def get_user_memory(self, user_id):
        """
        Retrieve complete memory for a user, including profile and recent history.

        If the user doesn't exist yet, a new entry is created automatically.

        Args:
            user_id (str): Unique identifier for the user.

        Returns:
            dict: User memory containing:
                - user_id (str): The user's ID
                - name (str or None): The user's name if known
                - history (list): Last 20 messages as dicts with role, message, intent, timestamp
                - preferences (dict): User preferences
                - message_count (int): Total number of messages from this user
        """
        cursor = self.conn.cursor()

        try:
            # Try to find the user
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()

            # If user doesn't exist, create a new entry
            if user is None:
                self._create_user(user_id)
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                user = cursor.fetchone()

            # Get the last 20 messages for this user
            cursor.execute("""
                SELECT role, message, intent, timestamp
                FROM history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 20
            """, (user_id,))
            rows = cursor.fetchall()

            # Convert rows to list of dicts (reverse to get chronological order)
            history = [
                {
                    "role": row["role"],
                    "message": row["message"],
                    "intent": row["intent"],
                    "timestamp": row["timestamp"]
                }
                for row in reversed(rows)
            ]

            # Parse preferences from JSON string
            preferences = {}
            if user["preferences"]:
                try:
                    preferences = json.loads(user["preferences"])
                except json.JSONDecodeError:
                    preferences = {}

            # Count total messages from this user
            cursor.execute(
                "SELECT COUNT(*) as count FROM history WHERE user_id = ?",
                (user_id,)
            )
            message_count = cursor.fetchone()["count"]

            return {
                "user_id": user_id,
                "name": user["name"],
                "history": history,
                "preferences": preferences,
                "message_count": message_count
            }

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error getting user memory: {e}")
            return {
                "user_id": user_id,
                "name": None,
                "history": [],
                "preferences": {},
                "message_count": 0
            }

    def _create_user(self, user_id):
        """
        Create a new user entry in the database.

        Args:
            user_id (str): Unique identifier for the new user.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, name, created_at, preferences) VALUES (?, ?, ?, ?)",
            (user_id, None, datetime.now().isoformat(), json.dumps({}))
        )
        self.conn.commit()

    def save_interaction(self, user_id, user_message, assistant_response, intent=None):
        """
        Save a complete interaction (user message + assistant response) to history.

        Also auto-detects if the user mentions their name (e.g., "my name is Ahmed")
        and stores it in their profile.

        Args:
            user_id (str): The user's unique identifier.
            user_message (str): What the user said.
            assistant_response (str): What the assistant replied.
            intent (str, optional): The detected intent of the user's message.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        try:
            # Make sure the user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if cursor.fetchone() is None:
                self._create_user(user_id)

            # Save the user's message
            cursor.execute(
                "INSERT INTO history (user_id, role, message, intent, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, "user", user_message, intent, now)
            )

            # Save the assistant's response
            cursor.execute(
                "INSERT INTO history (user_id, role, message, intent, timestamp) VALUES (?, ?, ?, ?, ?)",
                (user_id, "assistant", assistant_response, intent, now)
            )

            self.conn.commit()

            # Auto-detect name from message (e.g., "my name is Ahmed" or "I'm Ahmed")
            self._detect_and_store_name(user_id, user_message)

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error saving interaction: {e}")

    def _detect_and_store_name(self, user_id, message):
        """
        Auto-detect if the user mentions their name and store it.

        Patterns detected:
            - "my name is X"
            - "I'm X"
            - "I am X"
            - "call me X"

        Args:
            user_id (str): The user's unique identifier.
            message (str): The user's message to check for name mentions.
        """
        # Common patterns for name introduction
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                name = match.group(1).capitalize()
                self.update_user_name(user_id, name)
                break

    def update_user_name(self, user_id, name):
        """
        Update the user's name in their profile.

        Args:
            user_id (str): The user's unique identifier.
            name (str): The new name to store.
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                "UPDATE users SET name = ? WHERE user_id = ?",
                (name, user_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"[MemorySystem] Error updating user name: {e}")

    def update_preferences(self, user_id, preferences_dict):
        """
        Update user preferences by merging with existing ones.

        New keys are added, existing keys are overwritten.

        Args:
            user_id (str): The user's unique identifier.
            preferences_dict (dict): Dictionary of preferences to merge.

        Example:
            memory.update_preferences("user123", {"language": "ar", "theme": "dark"})
        """
        cursor = self.conn.cursor()

        try:
            # Get existing preferences
            cursor.execute(
                "SELECT preferences FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if row is None:
                # User doesn't exist, create them first
                self._create_user(user_id)
                existing = {}
            else:
                # Parse existing preferences
                try:
                    existing = json.loads(row["preferences"]) if row["preferences"] else {}
                except json.JSONDecodeError:
                    existing = {}

            # Merge new preferences with existing ones
            existing.update(preferences_dict)

            # Save back to database
            cursor.execute(
                "UPDATE users SET preferences = ? WHERE user_id = ?",
                (json.dumps(existing), user_id)
            )
            self.conn.commit()

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error updating preferences: {e}")

    def get_context(self, user_id):
        """
        Generate a formatted context string for the response engine.

        This provides the AI with relevant context about the user to make
        responses more personalized and relevant.

        Args:
            user_id (str): The user's unique identifier.

        Returns:
            str: A formatted context string, e.g.:
                 "User name: Ahmed. Previous topics: AI, python. Last message: what is ML?"
        """
        cursor = self.conn.cursor()

        try:
            # Get user profile
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()

            if user is None:
                return "New user. No previous context."

            # Build context parts
            parts = []

            # Add name if known
            if user["name"]:
                parts.append(f"User name: {user['name']}")
            else:
                parts.append("User name: unknown")

            # Get last 5 interactions for context summary
            cursor.execute("""
                SELECT role, message, intent
                FROM history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 10
            """, (user_id,))
            rows = cursor.fetchall()

            # Extract topics from intents
            intents = set()
            last_user_message = None

            for row in rows:
                if row["intent"]:
                    intents.add(row["intent"])
                if row["role"] == "user" and last_user_message is None:
                    last_user_message = row["message"]

            # Add previous topics
            if intents:
                parts.append(f"Previous topics: {', '.join(intents)}")

            # Add last message
            if last_user_message:
                # Truncate if too long
                if len(last_user_message) > 100:
                    last_user_message = last_user_message[:100] + "..."
                parts.append(f"Last message: {last_user_message}")

            return ". ".join(parts) + "."

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error getting context: {e}")
            return "Error retrieving context."

    def clear_memory(self, user_id):
        """
        Delete all data for a user (history and profile).

        This is a permanent action and cannot be undone.

        Args:
            user_id (str): The user's unique identifier.
        """
        cursor = self.conn.cursor()

        try:
            # Delete all history for the user
            cursor.execute("DELETE FROM history WHERE user_id = ?", (user_id,))

            # Delete the user profile
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

            self.conn.commit()

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error clearing memory: {e}")

    def get_all_users(self):
        """
        Get a list of all users with their message counts.

        Returns:
            list: List of dicts, each containing:
                - user_id (str): The user's unique identifier
                - message_count (int): Total number of messages from this user
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT u.user_id, COUNT(h.id) as message_count
                FROM users u
                LEFT JOIN history h ON u.user_id = h.user_id
                GROUP BY u.user_id
            """)
            rows = cursor.fetchall()

            return [
                {
                    "user_id": row["user_id"],
                    "message_count": row["message_count"]
                }
                for row in rows
            ]

        except sqlite3.Error as e:
            print(f"[MemorySystem] Error getting all users: {e}")
            return []
