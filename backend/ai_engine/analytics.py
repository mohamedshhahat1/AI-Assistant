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

    Uses the `short_term_memory` table (from the Advanced Memory System)
    and the `users` and `sessions` tables for all analytics queries.
    """

    def __init__(self, db_path="backend/data/memory.db"):
        """
        Initialize the Analytics system with a connection to the SQLite database.

        Args:
            db_path (str): Path to the SQLite database file.
                           Defaults to 'backend/data/memory.db' (same as MemorySystem).
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_overview(self) -> dict:
        """
        Get general overview statistics about the system.

        Returns:
            dict: Overview stats including total_users, total_messages,
                  total_conversations, avg_messages_per_user,
                  active_today, active_this_week.
        """
        cursor = self.conn.cursor()

        try:
            # Count total users
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()["count"]

            # Count total messages from short_term_memory
            cursor.execute("SELECT COUNT(*) as count FROM short_term_memory")
            total_messages = cursor.fetchone()["count"]

            # Each interaction creates 2 messages (user + assistant)
            total_conversations = total_messages // 2

            # Average messages per user
            avg_messages_per_user = round(total_messages / total_users, 1) if total_users > 0 else 0.0

            # Count users active today
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as count
                FROM short_term_memory
                WHERE timestamp LIKE ?
            """, (f"{today}%",))
            active_today = cursor.fetchone()["count"]

            # Count users active this week (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as count
                FROM short_term_memory
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

        Returns:
            list: List of dicts with intent, count, percentage.
        """
        cursor = self.conn.cursor()

        try:
            # Count total user messages with a valid intent
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM short_term_memory
                WHERE role = 'user' AND intent IS NOT NULL AND intent != ''
            """)
            total = cursor.fetchone()["total"]

            if total == 0:
                return []

            # Count each intent, sorted by frequency
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM short_term_memory
                WHERE role = 'user' AND intent IS NOT NULL AND intent != ''
                GROUP BY intent
                ORDER BY count DESC
            """)
            rows = cursor.fetchall()

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

        Returns:
            list: List of dicts with user_id, name, message_count, last_active.
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    u.user_id,
                    u.name,
                    COUNT(stm.id) as message_count,
                    MAX(stm.timestamp) as last_active
                FROM users u
                LEFT JOIN short_term_memory stm ON u.user_id = stm.user_id
                GROUP BY u.user_id
                ORDER BY message_count DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

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

        Returns:
            list: List of 24 dicts with hour and count.
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM short_term_memory
                WHERE timestamp IS NOT NULL AND LENGTH(timestamp) >= 13
                GROUP BY hour
                ORDER BY hour
            """)
            rows = cursor.fetchall()

            hour_counts = {row["hour"]: row["count"] for row in rows}

            result = []
            for h in range(24):
                result.append({
                    "hour": h,
                    "count": hour_counts.get(h, 0)
                })

            return result

        except sqlite3.Error as e:
            print(f"[Analytics] Error getting hourly activity: {e}")
            return [{"hour": h, "count": 0} for h in range(24)]

    def get_daily_activity(self, days=30) -> list:
        """
        Get message count by date for the last N days.

        Returns:
            list: List of dicts with date and count.
        """
        cursor = self.conn.cursor()

        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor.execute("""
                SELECT
                    SUBSTR(timestamp, 1, 10) as date,
                    COUNT(*) as count
                FROM short_term_memory
                WHERE timestamp >= ?
                GROUP BY date
                ORDER BY date DESC
            """, (start_date,))
            rows = cursor.fetchall()

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

        Returns:
            list: List of dicts with user_id, name, message, intent, timestamp.
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    stm.user_id,
                    u.name,
                    stm.message,
                    stm.intent,
                    stm.timestamp
                FROM short_term_memory stm
                LEFT JOIN users u ON stm.user_id = u.user_id
                WHERE stm.role = 'user'
                ORDER BY stm.id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

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

        Returns:
            dict: User stats with user_id, name, total_messages, first_seen,
                  last_active, top_intents, preferences.
        """
        cursor = self.conn.cursor()

        try:
            # Get user profile
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()

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

            # Count total messages
            cursor.execute(
                "SELECT COUNT(*) as count FROM short_term_memory WHERE user_id = ?",
                (user_id,)
            )
            total_messages = cursor.fetchone()["count"]

            # Get first and last timestamps
            cursor.execute(
                "SELECT MIN(timestamp) as first_seen, MAX(timestamp) as last_active FROM short_term_memory WHERE user_id = ?",
                (user_id,)
            )
            time_row = cursor.fetchone()
            first_seen = time_row["first_seen"]
            last_active = time_row["last_active"]

            # Get top intents
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM short_term_memory
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

            # Parse preferences
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
