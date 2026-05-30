"""
Analytics System for AI Assistant
==================================
This module provides analytics and reporting capabilities by querying
the same SQLite database used by the MemorySystem.

It answers questions like:
- How many users do we have?
- What are the most common intents?
- When are users most active?
- What are the recent conversations?

Usage:
    analytics = Analytics()
    overview = analytics.get_overview()
    print(f"Total users: {overview['total_users']}")
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta


class Analytics:
    """
    Provides analytics and reporting on user interactions.

    Connects to the same SQLite database as MemorySystem to provide
    insights about usage patterns, popular intents, and user activity.

    Thread-safe for use in web servers (FastAPI, etc.).
    """

    def __init__(self, db_path="backend/data/memory.db"):
        """
        Initialize the Analytics system with a connection to the SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
                           Defaults to 'backend/data/memory.db' (same as MemorySystem).
        """
        # Connect to the database with thread-safety enabled
        # check_same_thread=False allows multiple threads to use this connection
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        # This lets us access columns by name (e.g., row["user_id"])
        self.conn.row_factory = sqlite3.Row

    def get_overview(self) -> dict:
        """
        Get general overview statistics about the system.

        Returns:
            dict: Overview stats including:
                - total_users (int): Number of registered users
                - total_messages (int): Total messages in the system
                - total_conversations (int): Estimated conversations (messages / 2)
                - avg_messages_per_user (float): Average messages per user
                - active_today (int): Users who sent messages today
                - active_this_week (int): Users who sent messages this week
        """
        cursor = self.conn.cursor()

        try:
            # Count total users
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()["count"]

            # Count total messages (both user and assistant roles)
            cursor.execute("SELECT COUNT(*) as count FROM history")
            total_messages = cursor.fetchone()["count"]

            # Each interaction creates 2 messages (user + assistant),
            # so total conversations = total_messages / 2
            total_conversations = total_messages // 2

            # Calculate average messages per user (avoid division by zero)
            avg_messages_per_user = round(total_messages / total_users, 1) if total_users > 0 else 0.0

            # Count users active today
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as count
                FROM history
                WHERE timestamp LIKE ?
            """, (f"{today}%",))
            active_today = cursor.fetchone()["count"]

            # Count users active this week (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as count
                FROM history
                WHERE timestamp >= ?
            """, (week_ago,))
            active_this_week = cursor.fetchone()["count"]

            return {
                "total_users": total_users,
                "total_messages": total_messages,
                "total_conversations": total_conversations,
                "avg_messages_per_user": avg_messages_per_user,
                "active_today": active_today,
                "active_this_week": active_this_week
            }

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting overview: {e}")
            # Return zeros on error so the app doesn't crash
            return {
                "total_users": 0,
                "total_messages": 0,
                "total_conversations": 0,
                "avg_messages_per_user": 0.0,
                "active_today": 0,
                "active_this_week": 0
            }

    def get_intent_stats(self) -> list:
        """
        Get intent frequency statistics.

        Only counts "user" role messages (not assistant responses).
        Excludes messages with null/empty intents.
        Results are sorted by frequency (most common first).

        Returns:
            list: List of dicts, each containing:
                - intent (str): The intent name (e.g., "greeting")
                - count (int): How many times this intent was detected
                - percentage (float): What percentage of all intents this represents
        """
        cursor = self.conn.cursor()

        try:
            # Count total user messages with a valid intent
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM history
                WHERE role = 'user' AND intent IS NOT NULL AND intent != ''
            """)
            total = cursor.fetchone()["total"]

            # If no messages exist, return empty list
            if total == 0:
                return []

            # Count each intent, sorted by frequency
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM history
                WHERE role = 'user' AND intent IS NOT NULL AND intent != ''
                GROUP BY intent
                ORDER BY count DESC
            """)
            rows = cursor.fetchall()

            # Build the result list with percentages
            result = []
            for row in rows:
                result.append({
                    "intent": row["intent"],
                    "count": row["count"],
                    "percentage": round((row["count"] / total) * 100, 1)
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting intent stats: {e}")
            return []

    def get_active_users(self, limit=10) -> list:
        """
        Get the most active users sorted by message count.

        Args:
            limit (int): Maximum number of users to return (default: 10).

        Returns:
            list: List of dicts, each containing:
                - user_id (str): The user's unique identifier
                - name (str or None): The user's name if known
                - message_count (int): Total messages from this user
                - last_active (str): ISO timestamp of their most recent message
        """
        cursor = self.conn.cursor()

        try:
            # Join users and history tables to get name + message stats
            cursor.execute("""
                SELECT
                    u.user_id,
                    u.name,
                    COUNT(h.id) as message_count,
                    MAX(h.timestamp) as last_active
                FROM users u
                LEFT JOIN history h ON u.user_id = h.user_id
                GROUP BY u.user_id
                ORDER BY message_count DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

            # Convert rows to list of dicts
            result = []
            for row in rows:
                result.append({
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "message_count": row["message_count"],
                    "last_active": row["last_active"]
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting active users: {e}")
            return []

    def get_hourly_activity(self) -> list:
        """
        Get message count grouped by hour of day (0-23).

        This shows when users are most active throughout the day.
        Useful for understanding peak usage hours.

        Returns:
            list: List of 24 dicts (one per hour), each containing:
                - hour (int): Hour of the day (0-23)
                - count (int): Number of messages sent during that hour
        """
        cursor = self.conn.cursor()

        try:
            # Extract the hour from the timestamp and count messages per hour
            # SQLite timestamps are in ISO format: "2024-01-15T14:30:00"
            # We extract characters at positions 11-12 for the hour
            cursor.execute("""
                SELECT
                    CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM history
                WHERE timestamp IS NOT NULL AND LENGTH(timestamp) >= 13
                GROUP BY hour
                ORDER BY hour
            """)
            rows = cursor.fetchall()

            # Create a dict of existing hours for quick lookup
            hour_counts = {row["hour"]: row["count"] for row in rows}

            # Build a complete list for all 24 hours (fill missing hours with 0)
            result = []
            for h in range(24):
                result.append({
                    "hour": h,
                    "count": hour_counts.get(h, 0)
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting hourly activity: {e}")
            # Return all zeros on error
            return [{"hour": h, "count": 0} for h in range(24)]

    def get_daily_activity(self, days=30) -> list:
        """
        Get message count by date for the last N days.

        Args:
            days (int): Number of days to look back (default: 30).

        Returns:
            list: List of dicts (one per day), each containing:
                - date (str): The date in "YYYY-MM-DD" format
                - count (int): Number of messages sent on that date
              Sorted by date descending (most recent first).
        """
        cursor = self.conn.cursor()

        try:
            # Calculate the start date (N days ago)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Extract date from ISO timestamp and count messages per day
            # ISO timestamps look like "2024-01-15T14:30:00"
            # SUBSTR(timestamp, 1, 10) gives us "2024-01-15"
            cursor.execute("""
                SELECT
                    SUBSTR(timestamp, 1, 10) as date,
                    COUNT(*) as count
                FROM history
                WHERE timestamp >= ?
                GROUP BY date
                ORDER BY date DESC
            """, (start_date,))
            rows = cursor.fetchall()

            # Convert rows to list of dicts
            result = []
            for row in rows:
                result.append({
                    "date": row["date"],
                    "count": row["count"]
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting daily activity: {e}")
            return []

    def get_recent_conversations(self, limit=20) -> list:
        """
        Get the most recent user messages across all users.

        Only returns "user" role messages (not assistant responses)
        to show what users are asking about.

        Args:
            limit (int): Maximum number of messages to return (default: 20).

        Returns:
            list: List of dicts, each containing:
                - user_id (str): Who sent the message
                - name (str or None): The user's name if known
                - message (str): The message text
                - intent (str): The detected intent
                - timestamp (str): When the message was sent
        """
        cursor = self.conn.cursor()

        try:
            # Join with users table to get the name
            # Only get "user" role messages (not assistant responses)
            cursor.execute("""
                SELECT
                    h.user_id,
                    u.name,
                    h.message,
                    h.intent,
                    h.timestamp
                FROM history h
                LEFT JOIN users u ON h.user_id = u.user_id
                WHERE h.role = 'user'
                ORDER BY h.id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

            # Convert rows to list of dicts
            result = []
            for row in rows:
                result.append({
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "message": row["message"],
                    "intent": row["intent"],
                    "timestamp": row["timestamp"]
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting recent conversations: {e}")
            return []

    def get_user_stats(self, user_id) -> dict:
        """
        Get detailed statistics for a specific user.

        Args:
            user_id (str): The user's unique identifier.

        Returns:
            dict: User statistics including:
                - user_id (str): The user's ID
                - name (str or None): The user's name
                - total_messages (int): Total messages from this user
                - first_seen (str): Timestamp of their first message
                - last_active (str): Timestamp of their most recent message
                - top_intents (list): Most common intents for this user
                - preferences (dict): User preferences
        """
        cursor = self.conn.cursor()

        try:
            # Get user profile information
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()

            # If user doesn't exist, return empty stats
            if user is None:
                return {
                    "user_id": user_id,
                    "name": None,
                    "total_messages": 0,
                    "first_seen": None,
                    "last_active": None,
                    "top_intents": [],
                    "preferences": {}
                }

            # Count total messages for this user
            cursor.execute(
                "SELECT COUNT(*) as count FROM history WHERE user_id = ?",
                (user_id,)
            )
            total_messages = cursor.fetchone()["count"]

            # Get first and last message timestamps
            cursor.execute(
                "SELECT MIN(timestamp) as first_seen, MAX(timestamp) as last_active FROM history WHERE user_id = ?",
                (user_id,)
            )
            time_row = cursor.fetchone()
            first_seen = time_row["first_seen"]
            last_active = time_row["last_active"]

            # Get top intents for this user (only from user messages)
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM history
                WHERE user_id = ? AND role = 'user' AND intent IS NOT NULL AND intent != ''
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 5
            """, (user_id,))
            intent_rows = cursor.fetchall()

            top_intents = [
                {"intent": row["intent"], "count": row["count"]}
                for row in intent_rows
            ]

            # Parse user preferences from JSON string
            preferences = {}
            if user["preferences"]:
                try:
                    preferences = json.loads(user["preferences"])
                except json.JSONDecodeError:
                    preferences = {}

            return {
                "user_id": user_id,
                "name": user["name"],
                "total_messages": total_messages,
                "first_seen": first_seen,
                "last_active": last_active,
                "top_intents": top_intents,
                "preferences": preferences
            }

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting user stats: {e}")
            return {
                "user_id": user_id,
                "name": None,
                "total_messages": 0,
                "first_seen": None,
                "last_active": None,
                "top_intents": [],
                "preferences": {}
            }
