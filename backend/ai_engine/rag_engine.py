"""
RAG (Retrieval Augmented Generation) Engine
=============================================

This module implements a DYNAMIC, self-learning knowledge system using the RAG pattern.

What is RAG?
  RAG stands for Retrieval Augmented Generation. Instead of relying solely on a
  fixed model or static responses, RAG systems:
    1. RETRIEVE relevant information from a knowledge base
    2. AUGMENT the context with that retrieved information
    3. GENERATE a response informed by the retrieved knowledge

  Think of it like an open-book exam: rather than memorizing everything, the system
  looks up the most relevant information before answering.

Why RAG?
  - DYNAMIC: Knowledge can be added, updated, or removed without retraining
  - SCALABLE: Can handle thousands of knowledge chunks efficiently
  - ACCURATE: Retrieves specific, relevant information rather than guessing
  - SELF-LEARNING: Can absorb new knowledge from user interactions
  - TRANSPARENT: You can see exactly which knowledge was used to answer

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                         RAG Engine                                    │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────┐  │
  │  │ Auto Chunker│──>│ Quality Gate │──>│ Semantic Indexer (TF-IDF)│  │
  │  └─────────────┘   └──────────────┘   └─────────────────────────┘  │
  │         │                                         │                  │
  │         v                                         v                  │
  │  ┌─────────────┐                     ┌──────────────────────┐      │
  │  │ Topic Detect│                     │ SQLite Knowledge DB   │      │
  │  └─────────────┘                     └──────────────────────┘      │
  │                                               │                      │
  │                                               v                      │
  │                                      ┌────────────────┐             │
  │                                      │ Retrieval/Search│             │
  │                                      └────────────────┘             │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

5 Core Components:
  1. Auto Chunking     — Split text into semantic chunks
  2. Semantic Indexing  — Embed each chunk for fast retrieval (TF-IDF)
  3. Dynamic Updates    — Learn new knowledge from user interactions
  4. Quality Filtering  — Only store high-quality, useful information
  5. Retrieval          — Find relevant chunks for any query

Storage:
  Uses SQLite for persistent storage. Knowledge survives restarts.
  TF-IDF vectors are rebuilt in-memory on startup for fast search.
"""

import os
import re
import sqlite3
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim


class RAGEngine:
    """
    Retrieval Augmented Generation Engine.

    A dynamic, self-learning knowledge system that stores information as
    semantic chunks, indexes them for fast retrieval, and can learn new
    knowledge from user interactions.

    The engine uses TF-IDF (Term Frequency-Inverse Document Frequency) for
    vectorization. TF-IDF converts text into numerical vectors where each
    dimension represents a word's importance — words that appear often in one
    document but rarely across all documents get high scores.

    Attributes:
        db_path (str): Path to the SQLite database file.
        conn (sqlite3.Connection): Database connection.
        vectorizer (TfidfVectorizer): Scikit-learn TF-IDF vectorizer.
        chunk_vectors (numpy.ndarray): Pre-computed TF-IDF vectors for all chunks.
        chunk_ids (list): Database IDs corresponding to each vector row.
        chunk_contents (list): Text content corresponding to each vector row.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, db_path="backend/data/knowledge.db"):
        """
        Initialize the RAG Engine.

        Sets up the SQLite database, creates tables if they don't exist,
        seeds the static knowledge base, and builds the TF-IDF search index.

        Args:
            db_path (str): Path to the SQLite database file. The directory
                will be created if it doesn't exist.

        How it works:
            1. Create/connect to SQLite database
            2. Create tables for chunks and learning log
            3. Seed with static knowledge (if DB is empty)
            4. Build TF-IDF index for fast similarity search
        """
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path

        # Connect to SQLite with check_same_thread=False for multi-threaded access
        # (FastAPI runs async, so multiple threads may access the DB)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Create database tables
        self._create_tables()

        # Seed with static knowledge if database is empty
        self._seed_static_knowledge()

        # Build the TF-IDF search index from all stored chunks
        self.vectorizer = None
        self.chunk_vectors = None
        self.chunk_ids = []
        self.chunk_contents = []
        self.rebuild_index()

        # Print initialization stats
        stats = self.get_stats()
        print(f"[RAGEngine] Initialized with {stats['total_chunks']} knowledge chunks")
        print(f"[RAGEngine] Sources: {stats['sources']}")
        print(f"[RAGEngine] Topics: {list(stats['topics'].keys())}")
        print("[RAGEngine] Mode: TF-IDF retrieval with dynamic learning")

    def _create_tables(self):
        """
        Create the database tables if they don't already exist.

        Tables:
            knowledge_chunks: Stores each piece of knowledge with metadata.
                - content: The actual text of the knowledge chunk
                - embedding: Numpy array stored as bytes (for optional dense vectors)
                - source: Where this knowledge came from (static, user_learned, system, tool_output)
                - topic: Auto-detected category tag
                - quality_score: How reliable/useful this chunk is (0 to 1)
                - access_count: How often this chunk has been retrieved
                - created_at: When it was added
                - last_accessed: When it was last retrieved

            learning_log: Tracks what the system has learned from users.
                - user_id: Who taught us this
                - original_message: What the user said
                - extracted_knowledge: What we extracted and stored
                - accepted: Whether the knowledge passed quality filters
                - timestamp: When it happened
        """
        cursor = self.conn.cursor()

        # Main knowledge storage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                source TEXT DEFAULT 'static',
                topic TEXT DEFAULT 'general',
                quality_score REAL DEFAULT 0.7,
                access_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_accessed TEXT
            )
        """)

        # Learning log — tracks what we've learned from users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                original_message TEXT,
                extracted_knowledge TEXT,
                accepted BOOLEAN,
                timestamp TEXT
            )
        """)

        self.conn.commit()

    # =========================================================================
    # AUTO CHUNKING
    # =========================================================================

    def chunk_text(self, text, max_chunk_size=200):
        """
        Split text into semantic chunks.

        Chunking is important because:
          - Smaller chunks are more specific and retrieve better
          - Each chunk should be a self-contained piece of knowledge
          - Chunks that are too long dilute the meaning when searching

        Strategy:
          1. Split on sentence boundaries (., !, ?)
          2. If a sentence is too long, split on commas
          3. If still too long, split at max_chunk_size characters
          4. Strip whitespace and filter empty chunks

        Args:
            text (str): The text to split into chunks.
            max_chunk_size (int): Maximum characters per chunk (default 200).

        Returns:
            list: List of chunk strings, each a self-contained piece of info.

        Example:
            >>> engine.chunk_text("ML is great. It uses data to learn patterns.")
            ["ML is great", "It uses data to learn patterns"]
        """
        if not text or not text.strip():
            return []

        # Step 1: Split on sentence boundaries (period, exclamation, question mark)
        # We use a regex that splits on punctuation followed by a space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Step 2: If the sentence fits within max_chunk_size, keep it as-is
            if len(sentence) <= max_chunk_size:
                # Remove trailing punctuation for cleaner storage
                clean = sentence.rstrip('.!?').strip()
                if clean:
                    chunks.append(clean)
            else:
                # Step 3: Sentence is too long — split on commas
                parts = sentence.split(',')
                current_chunk = ""

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Check if adding this part would exceed the limit
                    if current_chunk and len(current_chunk) + len(part) + 2 > max_chunk_size:
                        # Save current chunk and start a new one
                        clean = current_chunk.rstrip('.!?,').strip()
                        if clean:
                            chunks.append(clean)
                        current_chunk = part
                    elif not current_chunk:
                        current_chunk = part
                    else:
                        current_chunk += ", " + part

                # Don't forget the last piece
                if current_chunk:
                    # Step 4: If still too long, hard-split at max_chunk_size
                    if len(current_chunk) > max_chunk_size:
                        for i in range(0, len(current_chunk), max_chunk_size):
                            piece = current_chunk[i:i + max_chunk_size].strip()
                            clean = piece.rstrip('.!?,').strip()
                            if clean:
                                chunks.append(clean)
                    else:
                        clean = current_chunk.rstrip('.!?,').strip()
                        if clean:
                            chunks.append(clean)

        return chunks

    # =========================================================================
    # SEMANTIC INDEXING
    # =========================================================================

    def _index_chunk(self, content, source, topic=None, quality_score=0.7):
        """
        Index a single knowledge chunk into the database.

        This stores the chunk with all its metadata and triggers an index rebuild
        so the new chunk becomes searchable immediately.

        Args:
            content (str): The text content to store.
            source (str): Where it came from ('static', 'user_learned', 'system', 'tool_output').
            topic (str, optional): Category tag. Auto-detected if None.
            quality_score (float): Reliability score from 0 to 1.

        How it works:
            1. Auto-detect topic if not provided
            2. Optionally compute embedding (numpy array stored as bytes)
            3. Insert into SQLite with metadata
            4. Rebuild the TF-IDF index to include this new chunk
        """
        # Auto-detect topic if not provided
        if topic is None:
            topic = self._detect_topic(content)

        # Create embedding placeholder (numpy zeros stored as bytes)
        # This is for future use with dense embeddings (e.g., sentence-transformers)
        embedding_placeholder = np.zeros(10, dtype=np.float32).tobytes()

        # Get current timestamp
        now = datetime.now().isoformat()

        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge_chunks (content, embedding, source, topic, quality_score, access_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
        """, (content, embedding_placeholder, source, topic, quality_score, now, now))
        self.conn.commit()

        # Rebuild the search index to include the new chunk
        self.rebuild_index()

    def rebuild_index(self):
        """
        Rebuild the TF-IDF search index from all chunks in the database.

        This is called:
          - On initialization (to build the initial index)
          - After adding new knowledge (to make it searchable)

        How TF-IDF works:
          - TF (Term Frequency): How often a word appears in a chunk
          - IDF (Inverse Document Frequency): How rare a word is across all chunks
          - TF-IDF = TF * IDF — words that are frequent in one chunk but rare overall
            get the highest scores, making them good discriminators.

        The vectorizer transforms text into sparse vectors. We then use cosine
        similarity to find chunks with similar word distributions to the query.
        """
        # Load all chunks from the database
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content FROM knowledge_chunks ORDER BY id")
        rows = cursor.fetchall()

        if not rows:
            # No chunks yet — nothing to index
            self.vectorizer = None
            self.chunk_vectors = None
            self.chunk_ids = []
            self.chunk_contents = []
            return

        # Extract IDs and content
        self.chunk_ids = [row["id"] for row in rows]
        self.chunk_contents = [row["content"] for row in rows]

        # Build TF-IDF vectorizer
        # - ngram_range=(1,2): Consider single words AND word pairs (bigrams)
        #   e.g., "machine learning" is a bigram that carries more meaning than
        #   "machine" and "learning" separately
        # - sublinear_tf=True: Apply log scaling to term frequency, reducing the
        #   impact of very common words within a document
        # - max_features=10000: Limit vocabulary size for performance
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=10000,
            sublinear_tf=True,
        )

        # Fit the vectorizer on all chunk contents and transform to vectors
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunk_contents)

    # =========================================================================
    # RETRIEVAL (Search)
    # =========================================================================

    def search(self, query, top_k=5, min_score=0.2):
        """
        Search the knowledge base for chunks relevant to the query.

        This is the core retrieval function. It:
          1. Vectorizes the query using the same TF-IDF vectorizer
          2. Computes cosine similarity against ALL indexed chunks
          3. Returns the top_k results above min_score threshold

        Cosine Similarity:
          Measures the angle between two vectors (0 = unrelated, 1 = identical).
          It's great for text because it's length-independent — a short query
          can still match a longer chunk if the words overlap meaningfully.

        Args:
            query (str): The search query text.
            top_k (int): Maximum number of results to return (default 5).
            min_score (float): Minimum similarity score threshold (default 0.2).

        Returns:
            list: List of dictionaries, each containing:
                - content (str): The chunk text
                - score (float): Cosine similarity score (0 to 1)
                - source (str): Where the chunk came from
                - topic (str): Category tag

        Example:
            >>> results = engine.search("what is machine learning")
            >>> results[0]
            {"content": "Machine Learning is...", "score": 0.85, "source": "static", "topic": "ai_ml"}
        """
        # Can't search if index is empty or not built
        if self.vectorizer is None or self.chunk_vectors is None:
            return []

        # Vectorize the query using the same TF-IDF vocabulary
        query_vec = self.vectorizer.transform([query.lower()])

        # Compute cosine similarity between the query and all chunks
        similarities = sklearn_cosine_sim(query_vec, self.chunk_vectors).flatten()

        # Get indices sorted by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]

        # Collect top_k results above the minimum score
        results = []
        for idx in sorted_indices[:top_k]:
            score = float(similarities[idx])

            # Skip results below the minimum score
            if score < min_score:
                break

            chunk_id = self.chunk_ids[idx]

            # Fetch full metadata from the database
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT content, source, topic FROM knowledge_chunks WHERE id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()

            if row:
                results.append({
                    "content": row["content"],
                    "score": round(score, 4),
                    "source": row["source"],
                    "topic": row["topic"],
                })

                # Update access statistics (this chunk was useful!)
                now = datetime.now().isoformat()
                cursor.execute(
                    "UPDATE knowledge_chunks SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (now, chunk_id)
                )
                self.conn.commit()

        return results

    def get_best_response(self, query):
        """
        Get the single best matching chunk for a query.

        This is a convenience method used by the hybrid engine. It returns
        the best match with a confidence threshold:
          - Score >= 0.5: Return the chunk content (confident match)
          - Score < 0.5: Return None (not confident enough)

        Args:
            query (str): The search query.

        Returns:
            tuple: (content_or_None, score)
                - If score >= 0.5: (chunk_content_string, score)
                - If score < 0.5: (None, score)

        Example:
            >>> content, score = engine.get_best_response("what is AI")
            >>> if content:
            ...     print(f"Found: {content} (confidence: {score})")
        """
        results = self.search(query, top_k=1, min_score=0.0)

        if not results:
            return (None, 0.0)

        best = results[0]
        score = best["score"]

        # Apply confidence threshold
        if score >= 0.5:
            return (best["content"], score)
        else:
            return (None, score)

    # =========================================================================
    # DYNAMIC LEARNING
    # =========================================================================

    def learn_from_message(self, user_message, intent, confidence):
        """
        Attempt to learn new knowledge from a user's message.

        This is what makes the system SELF-LEARNING. When a user states a fact
        or provides information, the system can absorb it for future retrieval.

        Quality Gate:
          Not everything a user says is worth storing. We filter out:
          - Short messages (< 20 chars) — too little information
          - Questions — the user is asking, not telling
          - Greetings/goodbyes — social niceties, not knowledge
          - Emotional expressions — not factual information
          - Low confidence messages — user might be uncertain

        Args:
            user_message (str): The user's message text.
            intent (str): The detected intent of the message.
            confidence (float): How confident the intent classifier is.

        Returns:
            bool: True if knowledge was extracted and stored, False otherwise.

        Example:
            >>> learned = engine.learn_from_message(
            ...     "Python was created by Guido van Rossum in 1991",
            ...     intent="statement",
            ...     confidence=0.85
            ... )
            >>> # learned == True (useful factual information)
        """
        # Quality gate: check if this message contains knowledge worth storing
        if not self._is_knowledge_worthy(user_message):
            # Log the rejection
            self._log_learning(
                user_id="anonymous",
                original_message=user_message,
                extracted_knowledge="",
                accepted=False
            )
            return False

        # Additional checks based on intent and confidence
        # We only learn from confident statements, not uncertain guesses
        if confidence < 0.5:
            return False

        # Don't learn from questions, greetings, or emotional intents
        non_learning_intents = [
            "greeting", "goodbye", "thanks", "question_ai",
            "question_general", "mood_positive", "mood_negative",
            "task_request", "about_bot"
        ]
        if intent in non_learning_intents:
            return False

        # Extract knowledge chunks from the message
        chunks = self.chunk_text(user_message)

        if not chunks:
            return False

        # Auto-detect topic from the message content
        topic = self._detect_topic(user_message)

        # Calculate quality score based on confidence and message characteristics
        # Longer, more confident messages get higher quality scores
        length_factor = min(len(user_message) / 200.0, 1.0)  # Caps at 1.0
        quality_score = (confidence * 0.7) + (length_factor * 0.3)
        quality_score = round(min(quality_score, 1.0), 3)

        # Store each chunk
        for chunk in chunks:
            # Only store chunks that pass the quality filter individually
            if len(chunk) >= 15:  # Minimum chunk length
                self._index_chunk(
                    content=chunk,
                    source="user_learned",
                    topic=topic,
                    quality_score=quality_score
                )

        # Log the successful learning
        self._log_learning(
            user_id="anonymous",
            original_message=user_message,
            extracted_knowledge="; ".join(chunks),
            accepted=True
        )

        return True

    def learn_from_text(self, text, source="system", topic=None):
        """
        Explicitly add new knowledge to the system.

        This is for administrative/programmatic use — when you want to bulk-add
        knowledge from a file, API response, tool output, etc.

        Unlike learn_from_message(), this bypasses the quality gate since it's
        assumed the caller has already validated the content.

        Args:
            text (str): The text to add as knowledge.
            source (str): The source tag ('system', 'tool_output', 'static').
            topic (str, optional): Category tag. Auto-detected if None.

        Example:
            >>> engine.learn_from_text(
            ...     "FastAPI is a modern Python web framework based on type hints. "
            ...     "It provides automatic API documentation and validation.",
            ...     source="system",
            ...     topic="programming"
            ... )
        """
        # Chunk the text into manageable pieces
        chunks = self.chunk_text(text)

        # Index each chunk
        for chunk in chunks:
            if len(chunk) >= 10:  # Minimum meaningful chunk size
                chunk_topic = topic if topic else self._detect_topic(chunk)
                self._index_chunk(
                    content=chunk,
                    source=source,
                    topic=chunk_topic,
                    quality_score=0.8  # System-added knowledge gets decent quality
                )

    # =========================================================================
    # QUALITY FILTERING
    # =========================================================================

    def _is_knowledge_worthy(self, text):
        """
        Determine if a text contains useful information worth storing.

        This is the quality gate that prevents garbage from polluting the
        knowledge base. It uses heuristics to identify informational content
        vs. social/emotional/question content.

        Checks:
          1. Length >= 20 characters (too short = not informative)
          2. Not a question (starts with wh-word or ends with ?)
          3. Not a greeting/goodbye (hello, bye, thanks, etc.)
          4. Not an emotional expression (I feel, I'm happy/sad)
          5. Contains at least one "noun-like" word (heuristic: word > 3 chars
             that's not a common stopword)

        Args:
            text (str): The text to evaluate.

        Returns:
            bool: True if the text is worth storing as knowledge.
        """
        if not text:
            return False

        # Check 1: Minimum length
        if len(text) < 20:
            return False

        text_lower = text.lower().strip()

        # Check 2: Not a question
        question_starters = ["what", "how", "why", "when", "where", "who", "which", "can", "do", "does", "is", "are"]
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in question_starters:
            return False
        if text_lower.endswith("?"):
            return False

        # Check 3: Not a greeting or goodbye
        greeting_words = [
            "hello", "hi", "hey", "bye", "goodbye", "good morning",
            "good night", "good evening", "good afternoon", "thanks",
            "thank you", "see you", "take care", "welcome"
        ]
        for greeting in greeting_words:
            if text_lower.startswith(greeting) or text_lower == greeting:
                return False

        # Check 4: Not an emotional expression
        emotional_patterns = [
            "i feel", "i'm happy", "i'm sad", "i'm feeling",
            "i am happy", "i am sad", "i am feeling",
            "i'm excited", "i'm bored", "i'm tired",
            "i'm stressed", "i'm angry", "i'm frustrated"
        ]
        for pattern in emotional_patterns:
            if text_lower.startswith(pattern):
                return False

        # Check 5: Contains at least one substantive word
        # (word > 3 chars that isn't a common stopword)
        stopwords = {
            "this", "that", "with", "from", "they", "them", "than",
            "then", "also", "just", "very", "really", "much", "more",
            "some", "have", "been", "were", "will", "would", "could",
            "should", "about", "into", "your", "their", "there", "here"
        }
        words = re.findall(r'\b[a-z]+\b', text_lower)
        substantive_words = [w for w in words if len(w) > 3 and w not in stopwords]

        if len(substantive_words) < 1:
            return False

        return True

    def _detect_topic(self, text):
        """
        Auto-detect the topic category of a text chunk.

        Uses simple keyword-based classification. Each category has a set of
        indicator words — if any appear in the text, that category is assigned.

        Categories:
          - programming: Code, software development topics
          - ai_ml: Artificial intelligence and machine learning
          - web_dev: Web development, APIs, servers
          - data: Databases, data analysis
          - general: Everything else (default)

        Args:
            text (str): The text to classify.

        Returns:
            str: The detected topic category.
        """
        text_lower = text.lower()

        # Topic keyword mappings (checked in order of specificity)
        topic_keywords = {
            "ai_ml": [
                "ai", "artificial intelligence", "machine learning", "neural",
                "model", "deep learning", "nlp", "natural language", "transformer",
                "gpt", "embedding", "training", "classification", "prediction",
                "supervised", "unsupervised", "reinforcement"
            ],
            "programming": [
                "python", "code", "function", "programming", "variable",
                "algorithm", "class", "object", "loop", "array", "list",
                "compile", "debug", "syntax", "library", "framework",
                "java", "rust", "golang", "typescript"
            ],
            "web_dev": [
                "api", "server", "web", "http", "html", "css", "javascript",
                "frontend", "backend", "rest", "endpoint", "route", "url",
                "react", "vue", "angular", "node", "express", "fastapi",
                "django", "flask"
            ],
            "data": [
                "data", "database", "sql", "table", "query", "schema",
                "postgresql", "mysql", "mongodb", "redis", "nosql",
                "dataframe", "pandas", "analytics", "csv", "json"
            ],
        }

        # Check each topic's keywords
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return topic

        return "general"

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self):
        """
        Get statistics about the knowledge base.

        Returns:
            dict: {
                "total_chunks": int,      — Total number of knowledge chunks
                "sources": dict,          — Breakdown by source type
                "topics": dict,           — Breakdown by topic category
                "avg_quality": float      — Average quality score
            }

        Example:
            >>> stats = engine.get_stats()
            >>> print(stats)
            {
                "total_chunks": 150,
                "sources": {"static": 97, "user_learned": 45, "system": 8},
                "topics": {"ai_ml": 40, "programming": 35, ...},
                "avg_quality": 0.75
            }
        """
        cursor = self.conn.cursor()

        # Total chunks
        cursor.execute("SELECT COUNT(*) as total FROM knowledge_chunks")
        total = cursor.fetchone()["total"]

        # Breakdown by source
        cursor.execute(
            "SELECT source, COUNT(*) as count FROM knowledge_chunks GROUP BY source"
        )
        sources = {row["source"]: row["count"] for row in cursor.fetchall()}

        # Breakdown by topic
        cursor.execute(
            "SELECT topic, COUNT(*) as count FROM knowledge_chunks GROUP BY topic"
        )
        topics = {row["topic"]: row["count"] for row in cursor.fetchall()}

        # Average quality score
        cursor.execute("SELECT AVG(quality_score) as avg_q FROM knowledge_chunks")
        avg_row = cursor.fetchone()
        avg_quality = round(avg_row["avg_q"], 3) if avg_row["avg_q"] else 0.0

        return {
            "total_chunks": total,
            "sources": sources,
            "topics": topics,
            "avg_quality": avg_quality,
        }

    def get_recent_learnings(self, limit=10):
        """
        Get the most recent things the system has learned from users.

        This is useful for monitoring what the system is absorbing and
        verifying that the quality filters are working correctly.

        Args:
            limit (int): Maximum number of entries to return (default 10).

        Returns:
            list: List of dictionaries, each containing:
                - user_id (str): Who taught us
                - original_message (str): What they said
                - extracted_knowledge (str): What we stored
                - accepted (bool): Whether it passed quality filters
                - timestamp (str): When it happened
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user_id, original_message, extracted_knowledge, accepted, timestamp
            FROM learning_log
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "user_id": row["user_id"],
                "original_message": row["original_message"],
                "extracted_knowledge": row["extracted_knowledge"],
                "accepted": bool(row["accepted"]),
                "timestamp": row["timestamp"],
            })

        return results

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _log_learning(self, user_id, original_message, extracted_knowledge, accepted):
        """
        Log a learning attempt (whether accepted or rejected).

        This creates an audit trail of everything the system considers learning,
        which is useful for debugging and improving the quality filters.

        Args:
            user_id (str): Identifier for the user.
            original_message (str): The user's original message.
            extracted_knowledge (str): What was extracted (empty if rejected).
            accepted (bool): Whether the knowledge was stored.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO learning_log (user_id, original_message, extracted_knowledge, accepted, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, original_message, extracted_knowledge, accepted, datetime.now().isoformat()))
        self.conn.commit()

    # =========================================================================
    # STATIC KNOWLEDGE SEEDING
    # =========================================================================

    def _seed_static_knowledge(self):
        """
        Seed the database with the initial static knowledge base.

        This loads 50+ knowledge entries covering AI, programming, web development,
        and general topics. These form the baseline knowledge that the system
        starts with before learning anything new from users.

        Only seeds if the database is empty — this prevents duplicate entries
        when the system restarts.

        The static knowledge comes from the same entries used in the hybrid_engine.py
        knowledge_base, ensuring consistency across the system.
        """
        # Check if database already has entries (avoid duplicate seeding)
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM knowledge_chunks")
        count = cursor.fetchone()["count"]

        if count > 0:
            # Already seeded — skip
            return

        # =====================================================================
        # STATIC KNOWLEDGE ENTRIES
        # These are the foundational facts the assistant knows from day one.
        # Each entry is stored as source="static" with quality_score=1.0 (trusted).
        # =====================================================================

        static_knowledge = [
            # --- Artificial Intelligence ---
            ("Artificial Intelligence (AI) is the field of computer science focused on creating "
             "machines that can perform tasks requiring human-like intelligence such as learning, "
             "reasoning, problem-solving, and understanding language", "ai_ml"),

            ("Machine Learning (ML) is a subset of AI where systems learn patterns from data "
             "without being explicitly programmed. Types include supervised learning, unsupervised "
             "learning, and reinforcement learning", "ai_ml"),

            ("Deep Learning uses neural networks with many layers to learn complex patterns. "
             "It powers image recognition, language translation, and speech synthesis", "ai_ml"),

            ("Natural Language Processing (NLP) is the branch of AI that helps machines understand "
             "human language. Applications include chatbots, translation, sentiment analysis, "
             "and text summarization", "ai_ml"),

            ("A neural network is a computing system inspired by biological neurons. Data flows "
             "through interconnected layers that detect patterns, enabling the network to learn "
             "from examples", "ai_ml"),

            ("Transformers are a neural network architecture that uses attention mechanisms to "
             "process sequences. They power models like GPT, BERT, and T5. Their key innovation "
             "is self-attention which captures long-range dependencies", "ai_ml"),

            ("GPT stands for Generative Pre-trained Transformer. It is a language model that "
             "generates human-like text by predicting the next word in a sequence. It is "
             "pre-trained on vast amounts of text data", "ai_ml"),

            ("Overfitting occurs when a model memorizes training data instead of learning "
             "generalizable patterns. It performs great on training data but poorly on new data. "
             "Solutions include regularization, dropout, more data, and early stopping", "ai_ml"),

            ("Supervised learning uses labeled data to train models. The model learns to map "
             "inputs to outputs from examples. Common tasks include classification and regression", "ai_ml"),

            ("Unsupervised learning finds hidden patterns in unlabeled data. Common techniques "
             "include clustering, dimensionality reduction, and anomaly detection", "ai_ml"),

            ("Reinforcement learning trains agents through rewards and penalties. The agent "
             "learns to take actions that maximize cumulative reward in an environment", "ai_ml"),

            ("Transfer learning reuses a pre-trained model on a new task. Instead of training "
             "from scratch, you fine-tune an existing model, saving time and requiring less data", "ai_ml"),

            ("Computer vision is the field of AI that enables machines to interpret and understand "
             "visual information from images and videos. Applications include object detection, "
             "facial recognition, and autonomous driving", "ai_ml"),

            ("Embeddings are dense vector representations of data like words or sentences in "
             "continuous vector space. Similar items have similar vectors, enabling semantic "
             "similarity search and analogical reasoning", "ai_ml"),

            ("TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures how "
             "important a word is to a document in a collection. Words frequent in one document "
             "but rare overall get high scores", "ai_ml"),

            # --- Programming ---
            ("Python is a high-level programming language used for web development with Django "
             "and Flask, data science with pandas and NumPy, AI and ML with PyTorch and "
             "scikit-learn, and automation scripting", "programming"),

            ("An API (Application Programming Interface) lets software applications communicate "
             "with each other. REST APIs use HTTP methods like GET, POST, PUT, and DELETE to "
             "perform operations on resources", "programming"),

            ("JavaScript is the language of the web. It runs in browsers for interactive "
             "websites and on servers via Node.js. Popular frameworks include React, Vue, "
             "Angular, and Express", "programming"),

            ("Git is a version control system that tracks code changes over time. It allows "
             "you to save snapshots (commits), revert mistakes, collaborate via branches, "
             "and share code on platforms like GitHub", "programming"),

            ("Object-oriented programming (OOP) organizes code into objects that contain data "
             "and behavior. Key principles are encapsulation, inheritance, polymorphism, and "
             "abstraction", "programming"),

            ("Data structures organize data for efficient access. Common ones include arrays, "
             "linked lists, stacks, queues, hash tables, trees, and graphs. Each has different "
             "time complexity tradeoffs", "programming"),

            ("Algorithms are step-by-step procedures for solving problems. Key categories include "
             "sorting (quicksort, mergesort), searching (binary search), graph traversal (BFS, DFS), "
             "and dynamic programming", "programming"),

            ("Design patterns are reusable solutions to common software problems. Examples include "
             "Singleton, Factory, Observer, Strategy, and MVC (Model-View-Controller)", "programming"),

            ("Testing ensures code works correctly. Unit tests check individual functions, "
             "integration tests verify components work together, and end-to-end tests simulate "
             "real user scenarios", "programming"),

            ("Docker packages applications into containers which are lightweight portable "
             "environments that run the same everywhere. It ensures consistency from development "
             "to production deployment", "programming"),

            # --- Web Development ---
            ("HTML (HyperText Markup Language) defines the structure of web pages using elements "
             "like headings, paragraphs, links, and images. It is the backbone of every website", "web_dev"),

            ("CSS (Cascading Style Sheets) controls the visual presentation of HTML elements "
             "including colors, layouts, fonts, and responsive design. Modern CSS uses Flexbox "
             "and Grid for layouts", "web_dev"),

            ("React is a JavaScript library created by Meta for building user interfaces with "
             "reusable components. It uses a virtual DOM for performance and hooks for state "
             "management", "web_dev"),

            ("REST (Representational State Transfer) is an architectural style for designing "
             "web APIs. It uses standard HTTP methods and status codes with stateless "
             "client-server communication", "web_dev"),

            ("FastAPI is a modern Python web framework that provides automatic API documentation, "
             "request validation using type hints, async support, and high performance comparable "
             "to Node.js", "web_dev"),

            ("Authentication verifies who a user is. Common methods include passwords, OAuth "
             "tokens, JWT (JSON Web Tokens), API keys, and multi-factor authentication", "web_dev"),

            ("WebSockets enable real-time bidirectional communication between client and server. "
             "Unlike HTTP which is request-response, WebSockets maintain a persistent connection "
             "for instant data transfer", "web_dev"),

            ("CORS (Cross-Origin Resource Sharing) is a security mechanism that controls which "
             "domains can access your API. Browsers block cross-origin requests unless the server "
             "explicitly allows them via headers", "web_dev"),

            # --- Databases ---
            ("SQL databases like PostgreSQL and MySQL use structured tables with predefined "
             "schemas and relationships. They excel at complex queries, transactions, and "
             "maintaining data integrity with ACID properties", "data"),

            ("NoSQL databases like MongoDB and Redis offer flexible schemas for different use "
             "cases. Document stores, key-value stores, column-family, and graph databases each "
             "optimize for specific access patterns", "data"),

            ("Database indexing speeds up data retrieval by creating lookup structures similar "
             "to a book index. Without indexes, the database must scan every row to find matches", "data"),

            ("SQL JOIN operations combine rows from multiple tables based on related columns. "
             "Types include INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN", "data"),

            ("Database normalization organizes data to reduce redundancy. Normal forms (1NF, 2NF, "
             "3NF) ensure each piece of data is stored once, preventing update anomalies", "data"),

            # --- General Knowledge ---
            ("To learn programming effectively: start with Python, learn fundamentals like "
             "variables loops and functions, build small projects, study data structures and "
             "algorithms, then pick a specialization", "general"),

            ("For coding interviews: practice LeetCode and HackerRank daily, study data "
             "structures and algorithms, do mock interviews, learn system design basics, "
             "and be ready to explain your project decisions", "general"),

            ("Building a developer portfolio: create 3-5 solid projects showing different skills, "
             "host code on GitHub with clean READMEs, deploy live demos, and write about your "
             "development process", "general"),

            ("Clean code principles include meaningful variable names, small focused functions, "
             "DRY (Don't Repeat Yourself), single responsibility, clear comments for why not "
             "what, and consistent formatting", "programming"),

            ("Agile development is an iterative approach to software delivery. Teams work in "
             "sprints (1-4 weeks), delivering small increments, gathering feedback, and "
             "continuously improving", "general"),

            ("Version control best practices include committing frequently with meaningful "
             "messages, using branches for features, reviewing code before merging, and never "
             "committing secrets or credentials", "programming"),

            ("The software development lifecycle includes requirements gathering, design, "
             "implementation, testing, deployment, and maintenance. Each phase has specific "
             "deliverables and quality gates", "general"),

            ("Debugging strategies include reading error messages carefully, using print "
             "statements or debuggers, isolating the problem with binary search, checking "
             "recent changes, and rubber duck debugging", "programming"),

            ("CI/CD (Continuous Integration/Continuous Deployment) automates testing and "
             "deployment. Code changes trigger automated tests, and passing code is automatically "
             "deployed to production", "programming"),

            ("Cloud computing provides on-demand computing resources over the internet. Major "
             "providers are AWS, Google Cloud, and Azure. Service models include IaaS, PaaS, "
             "and SaaS", "general"),

            ("Microservices architecture splits applications into small independent services "
             "that communicate via APIs. Each service handles one business capability and can "
             "be developed, deployed, and scaled independently", "web_dev"),

            ("Security best practices include input validation, parameterized queries to prevent "
             "SQL injection, HTTPS encryption, authentication and authorization, regular updates, "
             "and the principle of least privilege", "web_dev"),

            ("Performance optimization techniques include caching, database indexing, lazy "
             "loading, code profiling, CDN usage, image optimization, and minimizing HTTP "
             "requests", "web_dev"),

            ("The first computer programmer was Ada Lovelace who wrote algorithms for Charles "
             "Babbage's Analytical Engine in the 1840s, over 100 years before modern computers "
             "were built", "general"),

            ("Open source software is code that anyone can view, modify, and distribute. "
             "Popular licenses include MIT, Apache 2.0, and GPL. Contributing to open source "
             "builds skills and community", "general"),
        ]

        # Insert all static knowledge entries
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()

        for content, topic in static_knowledge:
            # Create embedding placeholder
            embedding_placeholder = np.zeros(10, dtype=np.float32).tobytes()

            cursor.execute("""
                INSERT INTO knowledge_chunks (content, embedding, source, topic, quality_score, access_count, created_at, last_accessed)
                VALUES (?, ?, 'static', ?, 1.0, 0, ?, ?)
            """, (content, embedding_placeholder, topic, now, now))

        self.conn.commit()
        print(f"[RAGEngine] Seeded {len(static_knowledge)} static knowledge entries")
