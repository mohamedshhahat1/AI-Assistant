"""
Advanced Memory System for AI Assistant
========================================

This module implements a cognitive-inspired memory architecture with two main components:

SHORT-TERM MEMORY (STM):
    - Holds the current conversation context (like human working memory)
    - Limited to the active session (~50 messages)
    - Fast access, temporary storage
    - Cleared/archived when session ends

LONG-TERM MEMORY (LTM):
    - Persistent knowledge about the user across sessions
    - Stores facts, preferences, topics of interest, and events
    - Each memory has an importance score (0-1)
    - Accessed less frequently but retained indefinitely
    - Memories are reinforced (access_count) when retrieved

USER PROFILE EVOLUTION:
    - The system learns about the user over time
    - Personality tags, interests, and communication style evolve
    - Profile is updated at the end of each session

SESSION TRACKING:
    - Each conversation is tracked as a session
    - Sessions have start/end times, message counts, mood, and summaries
    - Enables understanding of user engagement patterns

Usage:
    memory = AdvancedMemorySystem()
    session_id = memory.start_session("user123")
    memory.save_to_stm(session_id, "user123", "user", "Hello!", intent="greeting")
    memory.save_to_stm(session_id, "user123", "assistant", "Hi there!")
    context = memory.get_context("user123", session_id)
    memory.end_session(session_id)

Backward-compatible with the old MemorySystem interface:
    memory.save_interaction("user123", "Hello!", "Hi there!", intent="greeting")
    data = memory.load_memory("user123")
"""

import sqlite3
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from collections import Counter


class AdvancedMemorySystem:
    """
    A cognitive-inspired memory system that combines short-term and long-term memory
    with user profile evolution and session tracking.

    This class replaces the old MemorySystem while maintaining full backward
    compatibility with the previous interface (load_memory, save_interaction,
    clear_memory, get_all_users).
    """

    def __init__(self, db_path="backend/data/memory.db"):
        """
        Initialize the Advanced Memory System.

        Creates the SQLite database with the upgraded schema including tables for:
        - users (evolved profile with personality, interests, communication style)
        - sessions (conversation session tracking)
        - short_term_memory (current conversation context)
        - long_term_memory (persistent knowledge about the user)

        If old tables exist from the previous system, data is migrated automatically.

        Args:
            db_path (str): Path to the SQLite database file.
                           Defaults to 'backend/data/memory.db'.
        """
        # Create the data directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate_if_needed()

    def _create_tables(self):
        """Create the upgraded schema with STM, LTM, sessions, and user profile tables."""
        cursor = self.conn.cursor()

        # Users table — stores evolved user profile
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                personality_tags TEXT DEFAULT '[]',
                interests TEXT DEFAULT '[]',
                communication_style TEXT DEFAULT 'casual',
                created_at TEXT,
                last_seen TEXT,
                total_sessions INTEGER DEFAULT 0,
                preferences TEXT DEFAULT '{}'
            )
        """)

        # Sessions table — tracks each conversation session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                message_count INTEGER DEFAULT 0,
                summary TEXT,
                mood TEXT
            )
        """)

        # Short-term memory — current conversation context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,
                message TEXT,
                intent TEXT,
                emotion TEXT,
                timestamp TEXT
            )
        """)

        # Long-term memory — persistent knowledge about the user
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                memory_type TEXT,
                content TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)

        self.conn.commit()

    def _migrate_if_needed(self):
        """
        Migrate data from old tables if they exist.
        Maintains backward compatibility with the previous single-table schema
        that used 'users' and 'history' tables.
        """
        cursor = self.conn.cursor()

        # Check if old 'history' table exists (from previous MemorySystem version)
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='history'
        """)
        old_table = cursor.fetchone()

        if old_table:
            try:
                # Migrate old users — add new columns if they don't have them
                cursor.execute("PRAGMA table_info(users)")
                existing_columns = [col[1] for col in cursor.fetchall()]

                # The old users table had: user_id, name, created_at, preferences
                # We need to add: personality_tags, interests, communication_style,
                #                  last_seen, total_sessions
                new_columns = {
                    "personality_tags": "TEXT DEFAULT '[]'",
                    "interests": "TEXT DEFAULT '[]'",
                    "communication_style": "TEXT DEFAULT 'casual'",
                    "last_seen": "TEXT",
                    "total_sessions": "INTEGER DEFAULT 0"
                }

                for col_name, col_type in new_columns.items():
                    if col_name not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
                        except sqlite3.OperationalError:
                            pass  # Column already exists

                # Migrate old history into short_term_memory
                cursor.execute("SELECT DISTINCT user_id FROM history")
                old_users = cursor.fetchall()

                for row in old_users:
                    user_id = row[0]
                    now = datetime.now().isoformat()

                    # Update user's last_seen
                    cursor.execute("""
                        UPDATE users SET last_seen = ? WHERE user_id = ?
                    """, (now, user_id))

                    # Create a legacy session for migrated messages
                    legacy_session_id = f"legacy_{user_id}"
                    cursor.execute("""
                        SELECT session_id FROM sessions WHERE session_id = ?
                    """, (legacy_session_id,))

                    if not cursor.fetchone():
                        cursor.execute("""
                            INSERT INTO sessions (session_id, user_id, started_at, ended_at, summary)
                            VALUES (?, ?, ?, ?, ?)
                        """, (legacy_session_id, user_id, now, now, "Migrated from legacy system"))

                        # Move messages from history to short_term_memory
                        cursor.execute("""
                            SELECT role, message, intent, timestamp FROM history
                            WHERE user_id = ?
                            ORDER BY id ASC
                        """, (user_id,))
                        messages = cursor.fetchall()

                        for msg in messages:
                            timestamp = msg[3] if msg[3] else now
                            cursor.execute("""
                                INSERT INTO short_term_memory
                                (session_id, user_id, role, message, intent, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (legacy_session_id, user_id, msg[0], msg[1], msg[2], timestamp))

                self.conn.commit()
            except Exception:
                # If migration fails, continue with fresh tables
                pass

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, user_id):
        """
        Start a new conversation session for a user.

        - Creates a new session entry with a UUID
        - Ends any previous active session for the user
        - Increments user's total_sessions counter
        - Updates user's last_seen timestamp

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            str: A new UUID session_id.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        session_id = str(uuid.uuid4())

        # End any previous active session for this user
        cursor.execute("""
            UPDATE sessions SET ended_at = ?
            WHERE user_id = ? AND ended_at IS NULL
        """, (now, user_id))

        # Create user if they don't exist
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, created_at, last_seen, total_sessions)
            VALUES (?, ?, ?, 0)
        """, (user_id, now, now))

        # Increment total sessions and update last_seen
        cursor.execute("""
            UPDATE users SET total_sessions = total_sessions + 1, last_seen = ?
            WHERE user_id = ?
        """, (now, user_id))

        # Create new session
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, started_at, message_count)
            VALUES (?, ?, ?, 0)
        """, (session_id, user_id, now))

        self.conn.commit()
        return session_id

    def end_session(self, session_id):
        """
        End a session and perform post-session processing.

        This method:
        1. Marks the session as ended (sets ended_at timestamp)
        2. Generates a summary from the STM messages
        3. Extracts learned facts and stores them in LTM
        4. Evolves the user profile based on session content

        Args:
            session_id (str): The session to end.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        # Get session info
        cursor.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return
        user_id = row[0]

        # Mark session as ended
        cursor.execute("""
            UPDATE sessions SET ended_at = ? WHERE session_id = ?
        """, (now, session_id))

        # Generate session summary from STM messages
        summary = self._generate_session_summary(session_id)
        cursor.execute("""
            UPDATE sessions SET summary = ? WHERE session_id = ?
        """, (summary, session_id))

        # Extract facts from session and store in LTM
        self._extract_facts_to_ltm(session_id, user_id)

        self.conn.commit()

        # Evolve user profile based on accumulated data
        self.evolve_profile(user_id)

    def get_active_session(self, user_id):
        """
        Get the active session for a user.

        A session is considered "active" if:
        - ended_at is NULL (not yet ended)
        - started less than 30 minutes ago

        Args:
            user_id (str): The user to check.

        Returns:
            str or None: The active session_id, or None if no active session exists.
        """
        cursor = self.conn.cursor()
        threshold = (datetime.now() - timedelta(minutes=30)).isoformat()

        cursor.execute("""
            SELECT session_id FROM sessions
            WHERE user_id = ? AND ended_at IS NULL AND started_at > ?
            ORDER BY started_at DESC LIMIT 1
        """, (user_id, threshold))

        row = cursor.fetchone()
        return row[0] if row else None

    # =========================================================================
    # SHORT-TERM MEMORY (Current Conversation)
    # =========================================================================

    def save_to_stm(self, session_id, user_id, role, message, intent=None, emotion=None):
        """
        Save a message to short-term memory (current conversation context).

        STM holds up to ~50 messages for the active session. Older messages
        are automatically pruned when the limit is exceeded.

        Args:
            session_id (str): The current session ID.
            user_id (str): The user ID.
            role (str): "user" or "assistant".
            message (str): The message content.
            intent (str, optional): Detected intent (e.g., "greeting", "question_ai").
            emotion (str, optional): Detected emotion (e.g., "curious", "happy").
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO short_term_memory (session_id, user_id, role, message, intent, emotion, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, role, message, intent, emotion, now))

        # Update session message count
        cursor.execute("""
            UPDATE sessions SET message_count = message_count + 1
            WHERE session_id = ?
        """, (session_id,))

        # Keep STM limited to ~50 messages per session
        cursor.execute("""
            DELETE FROM short_term_memory
            WHERE id NOT IN (
                SELECT id FROM short_term_memory
                WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT 50
            ) AND session_id = ?
        """, (session_id, session_id))

        self.conn.commit()

    def get_stm(self, session_id, limit=20):
        """
        Retrieve recent messages from the current session's short-term memory.

        Messages are returned in chronological order (oldest first).

        Args:
            session_id (str): The session to retrieve messages from.
            limit (int): Maximum number of messages to return (default 20).

        Returns:
            list: Messages as dicts with keys:
                - role (str): "user" or "assistant"
                - message (str): The message content
                - intent (str or None): Detected intent
                - emotion (str or None): Detected emotion
                - timestamp (str): ISO format timestamp
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, message, intent, emotion, timestamp
            FROM short_term_memory
            WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        # Return in chronological order (oldest first)
        messages = []
        for row in reversed(rows):
            messages.append({
                "role": row[0],
                "message": row[1],
                "intent": row[2],
                "emotion": row[3],
                "timestamp": row[4]
            })
        return messages

    # =========================================================================
    # LONG-TERM MEMORY (Persistent Knowledge)
    # =========================================================================

    def save_to_ltm(self, user_id, memory_type, content, importance=0.5):
        """
        Save a memory to long-term storage.

        If the same content already exists, its importance is updated (max of old/new)
        and access_count is incremented instead of creating a duplicate.

        Args:
            user_id (str): The user this memory belongs to.
            memory_type (str): One of "fact", "preference", "topic", "event".
                - fact: "User is a computer science student"
                - preference: "User prefers Python over JavaScript"
                - topic: "User is interested in AI and machine learning"
                - event: "User mentioned they have an exam next week"
            importance (float): 0.0 to 1.0, higher = more important to remember.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        # Check for duplicate content to avoid storing the same fact twice
        cursor.execute("""
            SELECT id FROM long_term_memory
            WHERE user_id = ? AND content = ?
        """, (user_id, content))

        if cursor.fetchone():
            # Update existing memory's importance and access time
            cursor.execute("""
                UPDATE long_term_memory
                SET importance = MAX(importance, ?), last_accessed = ?, access_count = access_count + 1
                WHERE user_id = ? AND content = ?
            """, (importance, now, user_id, content))
        else:
            cursor.execute("""
                INSERT INTO long_term_memory
                (user_id, memory_type, content, importance, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (user_id, memory_type, content, importance, now, now))

        self.conn.commit()

    def get_ltm(self, user_id, memory_type=None, limit=10):
        """
        Retrieve long-term memories for a user.

        Memories are sorted by importance (descending), then by last_accessed
        (descending). When memories are retrieved, their access_count and
        last_accessed fields are updated (reinforcement).

        Args:
            user_id (str): The user to retrieve memories for.
            memory_type (str, optional): Filter by type — "fact", "preference",
                                         "topic", or "event". None returns all types.
            limit (int): Maximum number of memories to return (default 10).

        Returns:
            list: Memories as dicts with keys:
                - id (int): Memory ID
                - memory_type (str): Type of memory
                - content (str): The memory content
                - importance (float): Importance score 0-1
                - created_at (str): When the memory was first stored
                - last_accessed (str): When last retrieved
                - access_count (int): How many times retrieved
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        if memory_type:
            cursor.execute("""
                SELECT id, memory_type, content, importance, created_at, last_accessed, access_count
                FROM long_term_memory
                WHERE user_id = ? AND memory_type = ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """, (user_id, memory_type, limit))
        else:
            cursor.execute("""
                SELECT id, memory_type, content, importance, created_at, last_accessed, access_count
                FROM long_term_memory
                WHERE user_id = ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """, (user_id, limit))

        rows = cursor.fetchall()
        memories = []
        ids_to_update = []

        for row in rows:
            memories.append({
                "id": row[0],
                "memory_type": row[1],
                "content": row[2],
                "importance": row[3],
                "created_at": row[4],
                "last_accessed": row[5],
                "access_count": row[6]
            })
            ids_to_update.append(row[0])

        # Update access_count and last_accessed for retrieved memories (reinforcement)
        if ids_to_update:
            placeholders = ",".join(["?" for _ in ids_to_update])
            cursor.execute(f"""
                UPDATE long_term_memory
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders})
            """, [now] + ids_to_update)
            self.conn.commit()

        return memories

    def search_ltm(self, user_id, query):
        """
        Search long-term memory by keyword.

        Performs a simple keyword-based search across all LTM content for a user.
        Results are sorted by relevance (keyword match ratio * 0.7 + importance * 0.3).

        Args:
            user_id (str): The user whose memories to search.
            query (str): Search query string (space-separated keywords).

        Returns:
            list: Matching memories sorted by relevance, each with an additional
                  'relevance' field (0-1 score).
        """
        cursor = self.conn.cursor()
        keywords = query.lower().split()

        cursor.execute("""
            SELECT id, memory_type, content, importance, created_at, last_accessed, access_count
            FROM long_term_memory
            WHERE user_id = ?
        """, (user_id,))

        rows = cursor.fetchall()
        results = []

        for row in rows:
            content_lower = row[2].lower()
            # Calculate relevance: how many keywords match
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count > 0:
                # Relevance score combines keyword matches and importance
                relevance = (match_count / len(keywords)) * 0.7 + row[3] * 0.3
                results.append({
                    "id": row[0],
                    "memory_type": row[1],
                    "content": row[2],
                    "importance": row[3],
                    "created_at": row[4],
                    "last_accessed": row[5],
                    "access_count": row[6],
                    "relevance": relevance
                })

        # Sort by relevance descending
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results

    # =========================================================================
    # USER PROFILE EVOLUTION
    # =========================================================================

    def get_user_profile(self, user_id):
        """
        Get the evolved user profile with all computed fields.

        Returns a comprehensive profile that includes both stored data and
        computed fields like top_intents and recent_topics derived from
        conversation history.

        Args:
            user_id (str): The user to get profile for.

        Returns:
            dict: Complete user profile:
                - user_id (str): User identifier
                - name (str or None): User's name
                - personality_tags (list): e.g. ["curious", "technical", "friendly"]
                - interests (list): e.g. ["AI", "Python", "machine learning"]
                - communication_style (str): "formal", "casual", or "technical"
                - total_sessions (int): Number of sessions
                - total_messages (int): Total messages across all sessions
                - first_seen (str): First interaction timestamp
                - last_seen (str): Most recent interaction timestamp
                - top_intents (list): Most frequent intents
                - preferences (dict): User preferences
                - recent_topics (list): Recently discussed topics from LTM
        """
        cursor = self.conn.cursor()

        # Get base user info
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if not row:
            return {
                "user_id": user_id,
                "name": None,
                "personality_tags": [],
                "interests": [],
                "communication_style": "casual",
                "total_sessions": 0,
                "total_messages": 0,
                "first_seen": None,
                "last_seen": None,
                "top_intents": [],
                "preferences": {},
                "recent_topics": []
            }

        # Parse JSON fields safely
        try:
            personality_tags = json.loads(row["personality_tags"]) if row["personality_tags"] else []
        except (json.JSONDecodeError, TypeError):
            personality_tags = []

        try:
            interests = json.loads(row["interests"]) if row["interests"] else []
        except (json.JSONDecodeError, TypeError):
            interests = []

        try:
            preferences = json.loads(row["preferences"]) if row["preferences"] else {}
        except (json.JSONDecodeError, TypeError):
            preferences = {}

        # Calculate total messages across all sessions
        cursor.execute("""
            SELECT COUNT(*) FROM short_term_memory WHERE user_id = ?
        """, (user_id,))
        total_messages = cursor.fetchone()[0]

        # Get top intents from STM history
        cursor.execute("""
            SELECT intent, COUNT(*) as cnt FROM short_term_memory
            WHERE user_id = ? AND intent IS NOT NULL AND intent != ''
            GROUP BY intent ORDER BY cnt DESC LIMIT 5
        """, (user_id,))
        top_intents = [r[0] for r in cursor.fetchall()]

        # Get recent topics from LTM
        cursor.execute("""
            SELECT content FROM long_term_memory
            WHERE user_id = ? AND memory_type = 'topic'
            ORDER BY last_accessed DESC LIMIT 5
        """, (user_id,))
        recent_topics = [r[0] for r in cursor.fetchall()]

        return {
            "user_id": user_id,
            "name": row["name"],
            "personality_tags": personality_tags,
            "interests": interests,
            "communication_style": row["communication_style"],
            "total_sessions": row["total_sessions"],
            "total_messages": total_messages,
            "first_seen": row["created_at"],
            "last_seen": row["last_seen"],
            "top_intents": top_intents,
            "preferences": preferences,
            "recent_topics": recent_topics
        }

    def update_profile(self, user_id, updates):
        """
        Merge updates into the user profile fields.

        Only the specified fields are updated; other fields remain unchanged.

        Args:
            user_id (str): The user to update.
            updates (dict): Fields to update. Supported keys:
                - name (str): User's display name
                - personality_tags (list): List of personality trait strings
                - interests (list): List of interest strings
                - communication_style (str): "formal", "casual", or "technical"
                - preferences (dict): Key-value pairs merged with existing preferences
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        # Ensure user exists
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, created_at, last_seen, total_sessions)
            VALUES (?, ?, ?, 0)
        """, (user_id, now, now))

        # Update each supported field
        if "name" in updates:
            cursor.execute("UPDATE users SET name = ? WHERE user_id = ?",
                           (updates["name"], user_id))

        if "personality_tags" in updates:
            cursor.execute("UPDATE users SET personality_tags = ? WHERE user_id = ?",
                           (json.dumps(updates["personality_tags"]), user_id))

        if "interests" in updates:
            cursor.execute("UPDATE users SET interests = ? WHERE user_id = ?",
                           (json.dumps(updates["interests"]), user_id))

        if "communication_style" in updates:
            cursor.execute("UPDATE users SET communication_style = ? WHERE user_id = ?",
                           (updates["communication_style"], user_id))

        if "preferences" in updates:
            # Merge preferences (don't overwrite existing ones not in update)
            cursor.execute("SELECT preferences FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            try:
                existing_prefs = json.loads(row[0]) if row and row[0] else {}
            except (json.JSONDecodeError, TypeError):
                existing_prefs = {}
            existing_prefs.update(updates["preferences"])
            cursor.execute("UPDATE users SET preferences = ? WHERE user_id = ?",
                           (json.dumps(existing_prefs), user_id))

        self.conn.commit()

    def evolve_profile(self, user_id):
        """
        Automatically analyze user's history and evolve their profile.

        This method performs three types of analysis:
        1. Interest detection — from frequent LTM topics and preferences
        2. Communication style detection — from message length and vocabulary
        3. Personality tag detection — from behavioral patterns

        Called automatically at the end of each session via end_session().

        Args:
            user_id (str): The user whose profile to evolve.
        """
        cursor = self.conn.cursor()

        # --- Detect interests from LTM topics and frequent content ---
        cursor.execute("""
            SELECT content FROM long_term_memory
            WHERE user_id = ? AND memory_type IN ('topic', 'preference')
            ORDER BY importance DESC, access_count DESC
            LIMIT 20
        """, (user_id,))
        topic_rows = cursor.fetchall()
        interests = list(set([row[0] for row in topic_rows]))[:10]

        if interests:
            cursor.execute("UPDATE users SET interests = ? WHERE user_id = ?",
                           (json.dumps(interests), user_id))

        # --- Detect communication style from message patterns ---
        cursor.execute("""
            SELECT message FROM short_term_memory
            WHERE user_id = ? AND role = 'user'
            ORDER BY timestamp DESC LIMIT 50
        """, (user_id,))
        messages = [row[0] for row in cursor.fetchall()]

        if messages:
            style = self._detect_communication_style(messages)
            cursor.execute("UPDATE users SET communication_style = ? WHERE user_id = ?",
                           (style, user_id))

        # --- Detect personality tags ---
        personality_tags = self._detect_personality_tags(messages, user_id)
        if personality_tags:
            cursor.execute("UPDATE users SET personality_tags = ? WHERE user_id = ?",
                           (json.dumps(personality_tags), user_id))

        self.conn.commit()

    # =========================================================================
    # CONTEXT GENERATION (for response engine)
    # =========================================================================

    def get_context(self, user_id, session_id=None):
        """
        Generate rich context for the response engine.

        This is the primary method the response engine should call to understand
        who it's talking to and what's been discussed. It combines user profile
        data, recent conversation messages, and relevant long-term memories.

        Args:
            user_id (str): The user to generate context for.
            session_id (str, optional): Specific session ID. If None, uses the
                                        currently active session.

        Returns:
            dict: Rich context containing:
                - user_name (str or None): User's name
                - personality (list): Personality tags
                - interests (list): User's interests
                - communication_style (str): How the user communicates
                - session_messages (list): Last 5 STM messages
                - relevant_memories (list): Top 3 LTM memories by importance
                - session_mood (str): "positive", "negative", "neutral", or "mixed"
                - is_returning_user (bool): Whether this user has had previous sessions
                - sessions_count (int): Total number of sessions
        """
        profile = self.get_user_profile(user_id)

        # Get session messages
        session_messages = []
        active_session_id = session_id
        if session_id:
            session_messages = self.get_stm(session_id, limit=5)
        else:
            active_session_id = self.get_active_session(user_id)
            if active_session_id:
                session_messages = self.get_stm(active_session_id, limit=5)

        # Get relevant long-term memories (most important ones)
        relevant_memories = self.get_ltm(user_id, limit=3)

        # Determine session mood from recent emotions
        session_mood = self._determine_session_mood(active_session_id) if active_session_id else "neutral"

        # Check if returning user
        is_returning_user = (profile["total_sessions"] or 0) > 1

        return {
            "user_name": profile["name"],
            "personality": profile["personality_tags"],
            "interests": profile["interests"],
            "communication_style": profile["communication_style"],
            "session_messages": session_messages,
            "relevant_memories": [m["content"] for m in relevant_memories],
            "session_mood": session_mood,
            "is_returning_user": is_returning_user,
            "sessions_count": profile["total_sessions"]
        }

    # =========================================================================
    # BACKWARD COMPATIBILITY
    # =========================================================================

    def load_memory(self, user_id):
        """
        Backward-compatible method to load user memory.

        Maintains the old MemorySystem interface while using the new system
        internally. Code that previously called MemorySystem.load_memory()
        will continue to work unchanged.

        Args:
            user_id (str): The user to load memory for.

        Returns:
            dict: Contains:
                - user_id (str): The user's ID
                - name (str or None): The user's name if known
                - history (list): Recent messages as dicts with role, message, timestamp
                - preferences (dict): User preferences
                - message_count (int): Total number of messages
        """
        cursor = self.conn.cursor()

        # Get user info
        cursor.execute("SELECT name, preferences FROM users WHERE user_id = ?", (user_id,))
        user_row = cursor.fetchone()

        name = user_row["name"] if user_row else None
        try:
            preferences = json.loads(user_row["preferences"]) if user_row and user_row["preferences"] else {}
        except (json.JSONDecodeError, TypeError):
            preferences = {}

        # Get message history (most recent messages across all sessions)
        cursor.execute("""
            SELECT role, message, timestamp FROM short_term_memory
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT 50
        """, (user_id,))
        rows = cursor.fetchall()
        history = [{"role": r["role"], "message": r["message"], "timestamp": r["timestamp"]} for r in reversed(rows)]

        # Get total message count
        cursor.execute("SELECT COUNT(*) FROM short_term_memory WHERE user_id = ?", (user_id,))
        message_count = cursor.fetchone()[0]

        return {
            "user_id": user_id,
            "name": name,
            "history": history,
            "preferences": preferences,
            "message_count": message_count
        }

    def get_user_memory(self, user_id):
        """
        Alias for load_memory() — backward compatibility with old interface.

        Args:
            user_id (str): The user to load memory for.

        Returns:
            dict: Same as load_memory().
        """
        return self.load_memory(user_id)

    def save_interaction(self, user_id, user_message, assistant_response, intent=None):
        """
        Backward-compatible method to save a conversation interaction.

        Automatically:
        - Creates a session if one isn't active
        - Saves both messages to STM
        - Detects and stores the user's name if mentioned
        - Extracts facts for LTM if the message contains useful info

        Args:
            user_id (str): The user ID.
            user_message (str): What the user said.
            assistant_response (str): What the assistant replied.
            intent (str, optional): Detected intent of the user's message.
        """
        # Get or create active session
        session_id = self.get_active_session(user_id)
        if not session_id:
            session_id = self.start_session(user_id)

        # Save user message to STM
        self.save_to_stm(session_id, user_id, "user", user_message, intent=intent)

        # Save assistant response to STM
        self.save_to_stm(session_id, user_id, "assistant", assistant_response)

        # Auto-detect name from message
        self._detect_and_store_name(user_id, user_message)

        # Try to extract facts from user message for LTM
        self._extract_facts_from_message(user_id, user_message)

    def clear_memory(self, user_id):
        """
        Clear all memory (STM, LTM, sessions, profile) for a user.

        WARNING: This is destructive and irreversible. Removes all data
        associated with the user from every table.

        Args:
            user_id (str): The user whose memory to clear.
        """
        cursor = self.conn.cursor()

        # Clear all tables for this user
        cursor.execute("DELETE FROM short_term_memory WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM long_term_memory WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

        self.conn.commit()

    def get_all_users(self):
        """
        Backward-compatible method to get all users with message counts.

        Returns:
            list: List of dicts, each containing:
                - user_id (str): The user's unique identifier
                - name (str or None): The user's name
                - message_count (int): Total number of messages from this user
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT u.user_id, u.name, COUNT(stm.id) as message_count
            FROM users u
            LEFT JOIN short_term_memory stm ON u.user_id = stm.user_id
            GROUP BY u.user_id
        """)

        users = []
        for row in cursor.fetchall():
            users.append({
                "user_id": row[0],
                "name": row[1],
                "message_count": row[2]
            })
        return users

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _detect_and_store_name(self, user_id, message):
        """
        Auto-detect if the user mentions their name and store it in their profile.

        Patterns detected:
            - "my name is X"
            - "I'm X"
            - "I am X"
            - "call me X"

        Args:
            user_id (str): The user's unique identifier.
            message (str): The user's message to check for name mentions.
        """
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
                # Filter out common false positives
                false_positives = ["a", "the", "not", "very", "so", "too", "just",
                                   "really", "here", "there", "fine", "good", "bad"]
                if name.lower() not in false_positives:
                    self.update_profile(user_id, {"name": name})
                break

    def _generate_session_summary(self, session_id):
        """
        Generate a brief summary of the session from STM messages.

        Extracts unique intents, counts messages, and includes the first
        user message as a topic indicator.

        Args:
            session_id (str): The session to summarize.

        Returns:
            str: A brief summary of what was discussed.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, message, intent FROM short_term_memory
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        rows = cursor.fetchall()

        if not rows:
            return "Empty session"

        # Collect intents and key user messages for summary
        intents = [r[2] for r in rows if r[2]]
        user_messages = [r[1] for r in rows if r[0] == "user"]

        # Build a simple summary
        parts = []
        if intents:
            unique_intents = list(set(intents))
            parts.append(f"Intents: {', '.join(unique_intents[:5])}")

        msg_count = len(rows)
        parts.append(f"{msg_count} messages exchanged")

        # Include first user message as topic indicator
        if user_messages:
            first_msg = user_messages[0][:100]
            parts.append(f"Started with: '{first_msg}'")

        return ". ".join(parts)

    def _extract_facts_to_ltm(self, session_id, user_id):
        """
        Extract meaningful facts from session messages and store in LTM.

        Scans all user messages in the session for self-referential statements
        that reveal personal information, preferences, or events.

        Args:
            session_id (str): The session to extract facts from.
            user_id (str): The user the facts belong to.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message, intent FROM short_term_memory
            WHERE session_id = ? AND user_id = ? AND role = 'user'
            ORDER BY timestamp ASC
        """, (session_id, user_id))

        for row in cursor.fetchall():
            message = row[0]
            self._extract_facts_from_message(user_id, message)

    def _extract_facts_from_message(self, user_id, message):
        """
        Extract facts from a single user message and save to LTM.

        Uses keyword heuristics to detect:
        - Personal facts ("I am...", "My name is...")
        - Preferences ("I like...", "I prefer...")
        - Events ("I have an exam...", "Tomorrow I...")
        - Topics (from detected technical/topic keywords)

        Args:
            user_id (str): The user ID.
            message (str): The message to analyze.
        """
        if not message or len(message) < 5:
            return

        msg_lower = message.lower()

        # Detect personal facts
        fact_indicators = ["i am ", "i'm ", "my name is ", "i work ", "i study ",
                           "i live ", "my job ", "i have a ", "i'm a "]
        for indicator in fact_indicators:
            if indicator in msg_lower:
                self.save_to_ltm(user_id, "fact", message.strip(), importance=0.7)
                break

        # Detect preferences
        pref_indicators = ["i like ", "i love ", "i prefer ", "i enjoy ",
                           "i hate ", "i don't like ", "my favorite "]
        for indicator in pref_indicators:
            if indicator in msg_lower:
                self.save_to_ltm(user_id, "preference", message.strip(), importance=0.6)
                break

        # Detect events
        event_indicators = ["tomorrow ", "next week ", "today i ", "yesterday ",
                            "i have an exam", "i'm going to ", "i will ",
                            "next month ", "this weekend "]
        for indicator in event_indicators:
            if indicator in msg_lower:
                self.save_to_ltm(user_id, "event", message.strip(), importance=0.5)
                break

        # Detect topic interests from keywords
        topic_keywords = {
            "python": "Python programming",
            "javascript": "JavaScript",
            "machine learning": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "deep learning": "Deep Learning",
            "web development": "Web Development",
            "data science": "Data Science",
            "nlp": "Natural Language Processing",
            "neural network": "Neural Networks",
            "algorithm": "Algorithms",
            "database": "Databases",
            "cloud computing": "Cloud Computing",
            "cybersecurity": "Cybersecurity",
            "mobile development": "Mobile Development",
        }
        for keyword, topic in topic_keywords.items():
            if keyword in msg_lower:
                self.save_to_ltm(user_id, "topic", topic, importance=0.4)

    def _detect_communication_style(self, messages):
        """
        Detect the user's communication style from their messages.

        Analyzes message patterns to classify as:
        - "formal": Long sentences, polite language, proper punctuation
        - "casual": Short messages, informal language, abbreviations
        - "technical": Contains technical jargon, code-like patterns

        Args:
            messages (list): List of user message strings.

        Returns:
            str: "formal", "casual", or "technical".
        """
        if not messages:
            return "casual"

        # Analyze message characteristics
        avg_length = sum(len(m) for m in messages) / len(messages)
        total_text = " ".join(messages).lower()

        # Technical indicators
        technical_words = ["function", "variable", "algorithm", "api", "database",
                           "framework", "compile", "debug", "code", "class",
                           "import", "def ", "return", "error", "bug", "server",
                           "frontend", "backend", "deploy"]
        tech_count = sum(1 for w in technical_words if w in total_text)

        # Formal indicators
        formal_words = ["please", "could you", "would you", "thank you",
                        "i would appreciate", "kindly", "regards", "sincerely",
                        "i was wondering", "if you don't mind"]
        formal_count = sum(1 for w in formal_words if w in total_text)

        # Casual indicators
        casual_words = ["hey", "lol", "haha", "gonna", "wanna", "btw",
                        "omg", "bruh", "nah", "yep", "cool", "yo",
                        "sup", "tbh", "imo"]
        casual_count = sum(1 for w in casual_words if w in total_text)

        # Determine style based on weighted scores
        scores = {
            "technical": tech_count * 2 + (1 if avg_length > 80 else 0),
            "formal": formal_count * 2 + (1 if avg_length > 60 else 0),
            "casual": casual_count * 2 + (1 if avg_length < 40 else 0)
        }

        return max(scores, key=scores.get)

    def _detect_personality_tags(self, messages, user_id):
        """
        Detect personality tags from user message patterns and behavior.

        Analyzes message content and patterns to assign personality traits
        like "curious", "technical", "friendly", "detail-oriented", etc.

        Args:
            messages (list): List of user messages.
            user_id (str): The user ID for additional context.

        Returns:
            list: Personality tags (max 5), e.g. ["curious", "technical", "friendly"].
        """
        tags = []

        if not messages:
            return tags

        total_text = " ".join(messages).lower()
        question_count = sum(1 for m in messages if "?" in m)

        # Curious: asks many questions
        if question_count > len(messages) * 0.3:
            tags.append("curious")

        # Technical: uses technical language
        technical_words = ["code", "function", "algorithm", "data", "system",
                           "programming", "software", "api", "debug", "compile"]
        if sum(1 for w in technical_words if w in total_text) >= 3:
            tags.append("technical")

        # Friendly: uses greetings and positive language
        friendly_words = ["thanks", "thank you", "please", "hello", "hi",
                          "great", "awesome", "nice", "appreciate", "love"]
        if sum(1 for w in friendly_words if w in total_text) >= 2:
            tags.append("friendly")

        # Detail-oriented: writes longer messages
        avg_length = sum(len(m) for m in messages) / len(messages)
        if avg_length > 100:
            tags.append("detail-oriented")

        # Concise: writes short messages
        if avg_length < 30:
            tags.append("concise")

        # Creative: uses varied vocabulary and expressive language
        creative_words = ["imagine", "what if", "idea", "create", "design",
                          "build", "innovative", "unique", "experiment"]
        if sum(1 for w in creative_words if w in total_text) >= 2:
            tags.append("creative")

        # Patient: writes polite, measured responses
        patient_words = ["no problem", "take your time", "whenever", "no rush",
                         "i understand", "that's okay"]
        if sum(1 for w in patient_words if w in total_text) >= 1:
            tags.append("patient")

        return tags[:5]  # Limit to 5 tags

    def _determine_session_mood(self, session_id):
        """
        Determine the overall mood of a session from emotions in STM.

        Counts positive vs negative emotions to classify the session mood.

        Args:
            session_id (str): The session to analyze.

        Returns:
            str: "positive", "negative", "neutral", or "mixed".
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT emotion FROM short_term_memory
            WHERE session_id = ? AND emotion IS NOT NULL AND emotion != ''
        """, (session_id,))

        emotions = [row[0] for row in cursor.fetchall()]

        if not emotions:
            return "neutral"

        # Count positive vs negative emotions
        positive = {"happy", "excited", "grateful", "curious", "satisfied",
                    "positive", "joy", "enthusiastic", "hopeful"}
        negative = {"sad", "angry", "frustrated", "confused", "disappointed",
                    "negative", "anxious", "annoyed", "upset"}

        pos_count = sum(1 for e in emotions if e.lower() in positive)
        neg_count = sum(1 for e in emotions if e.lower() in negative)

        if pos_count > neg_count * 2:
            return "positive"
        elif neg_count > pos_count * 2:
            return "negative"
        elif pos_count > 0 and neg_count > 0:
            return "mixed"
        else:
            return "neutral"

    def __del__(self):
        """Clean up database connection on object destruction."""
        try:
            self.conn.close()
        except Exception:
            pass


# Backward-compatible alias so existing code that imports MemorySystem still works
MemorySystem = AdvancedMemorySystem
