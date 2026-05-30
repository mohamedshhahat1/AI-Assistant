"""
Hybrid Response Engine for AI Assistant
========================================

This module implements a HYBRID decision system that combines three methods
to generate the most accurate response:

  1. EMBEDDING RETRIEVAL — Understands the MEANING of the sentence
     Uses cosine similarity on embeddings (or TF-IDF fallback) to find
     the closest knowledge base entry.

  2. INTENT CLASSIFICATION — Understands the user's INTENT
     Uses ML classification (TF-IDF + LogReg) to categorize messages
     into intents: greeting, question, command, etc.

  3. FALLBACK GENERATION — Safety net when confidence is low
     Uses keyword matching or asks clarification questions.

Decision Matrix:
  ┌────────────────────┬──────────────────┬────────────────────────────┐
  │ Embedding Score    │ Intent Confidence│ Action                     │
  ├────────────────────┼──────────────────┼────────────────────────────┤
  │ >= 0.6             │ any              │ USE EMBEDDING (strong)     │
  │ 0.35 - 0.6        │ intent confirms  │ USE EMBEDDING (confirmed)  │
  │ < 0.35            │ >= 0.7           │ USE INTENT TEMPLATE        │
  │ low                │ low              │ USE FALLBACK               │
  └────────────────────┴──────────────────┴────────────────────────────┘

Why Hybrid?
  - Not reliant on a single method
  - More accurate (systems confirm each other)
  - Fewer errors (multiple safety nets)
"""

import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

try:
    from .embeddings import EmbeddingEngine
except ImportError:
    EmbeddingEngine = None


class HybridEngine:
    """
    Hybrid decision engine combining embeddings + intent + fallback.

    Processes every message through all three systems simultaneously,
    then applies a decision matrix to select the best response.
    """

    # Mapping from KB pattern categories to expected intents
    CATEGORY_INTENT_MAP = {
        "greeting": ["greeting"],
        "goodbye": ["goodbye"],
        "thanks": ["thanks"],
        "ai_tech": ["question_ai"],
        "programming": ["question_general", "question_ai"],
        "mood_positive": ["mood_positive"],
        "mood_negative": ["mood_negative"],
        "about_bot": ["about_bot"],
        "task": ["task_request"],
        "general": ["question_general"],
    }

    def __init__(self):
        """Initialize all three subsystems."""
        # Build knowledge base with category tags
        self._build_knowledge_base()

        # Initialize embedding/TF-IDF retrieval
        self.use_embeddings = False
        self.embedding_engine = None
        self.kb_embeddings = None
        self.vectorizer = None
        self.kb_vectors = None

        if EmbeddingEngine is not None:
            self.embedding_engine = EmbeddingEngine()
            if self.embedding_engine.is_available():
                self.use_embeddings = True
                self._build_embedding_index()
                print("[HybridEngine] Retrieval: Sentence Embeddings (all-MiniLM-L6-v2)")
            else:
                self._build_tfidf_index()
                print("[HybridEngine] Retrieval: TF-IDF (fallback)")
        else:
            self._build_tfidf_index()
            print("[HybridEngine] Retrieval: TF-IDF (fallback)")

        # Build intent templates and fallback responses
        self._build_intent_templates()
        self._build_fallback_responses()

        print(f"[HybridEngine] Knowledge base: {len(self.knowledge_base)} entries")
        print("[HybridEngine] Mode: HYBRID (embedding + intent + fallback)")

    # =========================================================================
    # MAIN PROCESSING PIPELINE
    # =========================================================================

    def process(self, message, intent_result=None, memory=None):
        """
        Main hybrid processing pipeline.

        Runs all three systems and picks the best response using the decision matrix.

        Args:
            message (str): User's message.
            intent_result (dict, optional): Pre-computed intent {"intent": "...", "confidence": ...}.
            memory (dict, optional): User memory/context.

        Returns:
            dict: {
                "response": "The chosen response text",
                "method": "embedding" | "intent" | "fallback",
                "confidence": 0.85,
                "intent": "question_ai",
                "embedding_score": 0.72,
                "intent_confidence": 0.85,
                "reasoning": "Strong semantic match (0.72)"
            }
        """
        # Extract user context
        user_name = None
        if memory:
            user_name = memory.get("name") or memory.get("user_name")

        # --- System 1: Embedding Retrieval ---
        embed_response, embed_score, embed_category = self._retrieval_match(message)

        # --- System 2: Intent Classification ---
        intent = "unknown"
        intent_confidence = 0.0
        if intent_result:
            intent = intent_result.get("intent", "unknown")
            intent_confidence = float(intent_result.get("confidence", 0.0))

        # --- System 3: Fallback ---
        fallback_response = self._get_fallback_response(message)

        # --- HYBRID DECISION LOGIC ---
        response = ""
        method = ""
        reasoning = ""
        final_confidence = 0.0

        # Decision 1: Strong embedding match (>= 0.6)
        if embed_score >= 0.6:
            response = embed_response
            method = "embedding"
            final_confidence = embed_score
            reasoning = f"Strong semantic match ({embed_score:.2f})"

        # Decision 2: Medium embedding + intent confirms
        elif embed_score >= 0.35 and self._intent_confirms_embedding(intent, embed_category):
            response = embed_response
            method = "embedding"
            final_confidence = (embed_score + intent_confidence) / 2
            reasoning = (f"Medium semantic match ({embed_score:.2f}) "
                        f"confirmed by intent ({intent}, {intent_confidence:.2f})")

        # Decision 3: Low embedding but strong intent
        elif intent_confidence >= 0.7 and intent in self.intent_templates:
            response = self._get_intent_response(intent)
            method = "intent"
            final_confidence = intent_confidence
            reasoning = f"Intent classification ({intent}, {intent_confidence:.2f})"

        # Decision 4: Medium intent (0.5-0.7) — still use intent if we have templates
        elif intent_confidence >= 0.5 and intent in self.intent_templates:
            response = self._get_intent_response(intent)
            method = "intent"
            final_confidence = intent_confidence
            reasoning = f"Moderate intent match ({intent}, {intent_confidence:.2f})"

        # Decision 5: Fallback
        else:
            response = fallback_response
            method = "fallback"
            final_confidence = 0.2
            reasoning = "Low confidence across all systems, using fallback"

        # --- Personalization ---
        if user_name and method != "fallback" and random.random() < 0.25:
            response = f"{user_name}, " + response[0].lower() + response[1:]

        # --- Handle name introduction specially ---
        if intent == "name_introduction":
            name = self._extract_name(message)
            if name:
                response = random.choice([
                    f"Nice to meet you, {name}! 😊 How can I help you?",
                    f"Hello, {name}! Great to know your name. What can I do for you?",
                    f"Hi {name}! 👋 Lovely to meet you!",
                ])
                method = "intent"
                final_confidence = 0.95
                reasoning = f"Name introduction detected: {name}"

        return {
            "response": response,
            "method": method,
            "confidence": round(final_confidence, 3),
            "intent": intent,
            "embedding_score": round(embed_score, 3),
            "intent_confidence": round(intent_confidence, 3),
            "reasoning": reasoning,
        }

    # =========================================================================
    # SYSTEM 1: EMBEDDING/TF-IDF RETRIEVAL
    # =========================================================================

    def _retrieval_match(self, message):
        """
        Find the most semantically similar KB entry.

        Returns:
            tuple: (response, score, category)
        """
        if self.use_embeddings:
            query_emb = self.embedding_engine.encode([message])
            similarities = self.embedding_engine.similarity(query_emb, self.kb_embeddings)
        else:
            query_vec = self.vectorizer.transform([message.lower()])
            similarities = sklearn_cosine_sim(query_vec, self.kb_vectors).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        _, response, category = self.knowledge_base[best_idx]
        return response, best_score, category

    def _build_embedding_index(self):
        """Pre-compute embeddings for all KB patterns."""
        patterns = [p for p, _, _ in self.knowledge_base]
        self.kb_embeddings = self.embedding_engine.encode(patterns)

    def _build_tfidf_index(self):
        """Build TF-IDF index for KB patterns."""
        patterns = [p for p, _, _ in self.knowledge_base]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=5000,
            sublinear_tf=True,
        )
        self.kb_vectors = self.vectorizer.fit_transform(patterns)

    # =========================================================================
    # SYSTEM 2: INTENT CONFIRMATION
    # =========================================================================

    def _intent_confirms_embedding(self, intent, kb_category):
        """
        Check if the detected intent aligns with the KB entry's category.

        For example, if the embedding matched an AI question pattern AND
        the intent classifier says "question_ai", they AGREE — boosting confidence.
        """
        expected_intents = self.CATEGORY_INTENT_MAP.get(kb_category, [])
        return intent in expected_intents

    def _get_intent_response(self, intent):
        """Get a random response from intent templates."""
        if intent in self.intent_templates:
            return random.choice(self.intent_templates[intent])
        return random.choice(self.fallback_responses)

    # =========================================================================
    # SYSTEM 3: FALLBACK
    # =========================================================================

    def _get_fallback_response(self, message):
        """Keyword matching + clarification questions."""
        message_lower = message.lower()

        for keyword, response in self.keyword_responses.items():
            if keyword in message_lower:
                return response

        return random.choice(self.fallback_responses)

    # =========================================================================
    # DEBUG / EXPLANATION
    # =========================================================================

    def get_decision_explanation(self, message, intent_result=None):
        """
        Debug method: shows detailed scores from all three systems.

        Useful for understanding why a particular response was chosen.
        """
        result = self.process(message, intent_result)

        # Get top 3 embedding matches
        if self.use_embeddings:
            query_emb = self.embedding_engine.encode([message])
            similarities = self.embedding_engine.similarity(query_emb, self.kb_embeddings)
        else:
            query_vec = self.vectorizer.transform([message.lower()])
            similarities = sklearn_cosine_sim(query_vec, self.kb_vectors).flatten()

        top_indices = np.argsort(similarities)[-3:][::-1]
        top_matches = []
        for idx in top_indices:
            pattern, _, cat = self.knowledge_base[idx]
            top_matches.append({
                "pattern": pattern,
                "category": cat,
                "score": round(float(similarities[idx]), 4)
            })

        return {
            "message": message,
            "decision": result,
            "top_kb_matches": top_matches,
            "fallback_response": self._get_fallback_response(message),
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _extract_name(self, message):
        """Extract name from introduction messages."""
        patterns = [
            r"(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)",
            r"(?:name's|they call me|people call me)\s+([a-zA-Z]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).capitalize()
        return None

    # =========================================================================
    # KNOWLEDGE BASE (pattern, response, category)
    # =========================================================================

    def _build_knowledge_base(self):
        """Build KB with category tags for intent confirmation."""
        self.knowledge_base = [
            # --- Greetings ---
            ("hello", "Hello! 👋 How can I help you today?", "greeting"),
            ("hi there", "Hi there! What's on your mind?", "greeting"),
            ("hey how are you", "Hey! I'm doing great, thanks! How about you?", "greeting"),
            ("good morning", "Good morning! ☀️ Hope you're having a wonderful day!", "greeting"),
            ("good afternoon", "Good afternoon! How's your day going?", "greeting"),
            ("good evening", "Good evening! 🌙 How can I help you tonight?", "greeting"),
            ("what's up", "Not much! Just here ready to help. What's up with you?", "greeting"),
            ("howdy", "Howdy! 🤠 What brings you here today?", "greeting"),

            # --- Goodbyes ---
            ("goodbye", "Goodbye! Have a wonderful day! 👋", "goodbye"),
            ("see you later", "See you later! Take care!", "goodbye"),
            ("bye bye", "Bye bye! Come back anytime!", "goodbye"),
            ("good night", "Good night! 🌙 Sweet dreams!", "goodbye"),
            ("take care", "You too! Take care! 👋", "goodbye"),

            # --- Thanks ---
            ("thank you so much", "You're very welcome! 😊 Glad I could help!", "thanks"),
            ("thanks a lot", "Anytime! That's what I'm here for!", "thanks"),
            ("I appreciate your help", "Happy to help! Let me know if you need more.", "thanks"),
            ("that was helpful", "I'm glad! Don't hesitate to ask more.", "thanks"),

            # --- AI & Technology ---
            ("what is artificial intelligence",
             "Artificial Intelligence (AI) is the field of computer science focused on creating "
             "machines that can perform tasks requiring human-like intelligence — learning, "
             "reasoning, problem-solving, and understanding language.", "ai_tech"),
            ("what is machine learning",
             "Machine Learning (ML) is a subset of AI where systems learn patterns from data "
             "without being explicitly programmed. Types include supervised, unsupervised, "
             "and reinforcement learning.", "ai_tech"),
            ("explain machine learning",
             "ML lets computers learn from examples instead of rules. You show it data, "
             "it finds patterns, then makes predictions on new data. Think of it like "
             "teaching by example rather than by instruction.", "ai_tech"),
            ("what is deep learning",
             "Deep Learning uses neural networks with many layers to learn complex patterns. "
             "It powers image recognition, language translation, and speech synthesis.", "ai_tech"),
            ("what is natural language processing",
             "NLP is the branch of AI that helps machines understand human language. "
             "Applications include chatbots, translation, sentiment analysis, and summarization.", "ai_tech"),
            ("what is a neural network",
             "A neural network is a computing system inspired by biological neurons. "
             "Data flows through interconnected layers that detect patterns, "
             "enabling the network to learn from examples.", "ai_tech"),
            ("how does AI work",
             "AI works by: 1) Collecting data, 2) Training a model to find patterns, "
             "3) Evaluating performance, 4) Deploying for predictions. Different techniques "
             "(neural nets, decision trees, etc.) suit different problems.", "ai_tech"),
            ("what are transformers in AI",
             "Transformers are a neural network architecture that uses attention mechanisms "
             "to process sequences. They power models like GPT, BERT, and T5. Their key "
             "innovation is self-attention, which captures long-range dependencies.", "ai_tech"),
            ("what is GPT",
             "GPT (Generative Pre-trained Transformer) is a language model that generates "
             "human-like text. It's pre-trained on vast text data, then fine-tuned for tasks. "
             "ChatGPT is built on GPT architecture.", "ai_tech"),
            ("what is overfitting",
             "Overfitting is when a model memorizes training data instead of learning patterns. "
             "It performs great on training data but poorly on new data. Solutions include "
             "regularization, dropout, more data, and early stopping.", "ai_tech"),

            # --- Programming ---
            ("what is python used for",
             "Python is used for web dev (Django/Flask), data science (pandas/NumPy), "
             "AI/ML (PyTorch/sklearn), automation, and more. Its readability makes it "
             "great for beginners and experts.", "programming"),
            ("what is an API",
             "An API lets software applications communicate. Like a waiter — you order "
             "(request), the kitchen prepares (server processes), waiter delivers (response). "
             "REST APIs use HTTP methods: GET, POST, PUT, DELETE.", "programming"),
            ("what is javascript",
             "JavaScript is the language of the web! It runs in browsers for interactive "
             "websites, and on servers via Node.js. It's used for React, Vue, Express, "
             "and full-stack development.", "programming"),
            ("how to learn programming",
             "Great path: 1️⃣ Start with Python, 2️⃣ Learn fundamentals (variables, loops, functions), "
             "3️⃣ Build small projects, 4️⃣ Learn data structures, 5️⃣ Pick a specialization. "
             "Practice daily!", "programming"),
            ("what is git",
             "Git is a version control system that tracks code changes. It lets you save "
             "snapshots, revert mistakes, collaborate via branches, and share on GitHub/GitLab.", "programming"),
            ("what is a database",
             "A database stores and organizes data for efficient retrieval. SQL databases "
             "(PostgreSQL, MySQL) use tables with relations. NoSQL (MongoDB, Redis) offer "
             "flexible schemas for different use cases.", "programming"),
            ("what is react",
             "React is a JavaScript library for building user interfaces. It uses components, "
             "virtual DOM for performance, and hooks for state management. Created by Meta, "
             "it's the most popular frontend framework.", "programming"),
            ("what is docker",
             "Docker packages applications into containers — lightweight, portable environments "
             "that run the same everywhere. Think of it as a shipping container for software: "
             "consistent from development to production.", "programming"),

            # --- Career & Learning ---
            ("how to prepare for coding interview",
             "Tips: 1) Practice LeetCode/HackerRank daily, 2) Study data structures & algorithms, "
             "3) Do mock interviews, 4) Learn system design basics, 5) Review your projects "
             "and be ready to explain decisions.", "general"),
            ("how to build a portfolio",
             "Build 3-5 solid projects showing different skills. Host code on GitHub with "
             "clean READMEs. Deploy live demos. Write about your process. Include: a full-stack "
             "app, an API project, and something with data/ML.", "general"),

            # --- About Bot ---
            ("who are you",
             "I'm an AI Assistant built with Python! 🤖 I use a hybrid system combining "
             "semantic embeddings, intent classification, and smart fallbacks to understand "
             "your messages and help you out!", "about_bot"),
            ("what can you do",
             "I can: 💬 chat naturally, 🧮 calculate math, 📝 save notes, ⏰ set reminders, "
             "📖 define words, 🕐 tell the time, answer questions about AI/tech/programming, "
             "and remember our conversations!", "about_bot"),
            ("how were you made",
             "I was built with Python + FastAPI for the backend, scikit-learn for NLP, "
             "sentence-transformers for embeddings, SQLite for memory, and HTML/JS for "
             "the chat interface. 100% self-contained — no external AI APIs!", "about_bot"),

            # --- Mood ---
            ("I'm feeling happy", "That's wonderful! 🎉 What's making your day great?", "mood_positive"),
            ("I feel great today", "Awesome! Positive energy is contagious! 😄", "mood_positive"),
            ("I'm feeling sad", "I'm sorry to hear that. 💙 Want to talk about it, or shall I try to cheer you up?", "mood_negative"),
            ("I'm stressed out", "Take a deep breath. 🌿 Break big tasks into small ones, take a walk, "
             "and remember — it's okay to ask for help. You've got this!", "mood_negative"),
            ("I'm bored", "Let's fix that! 🎯 Learn something new, try a coding challenge, "
             "or ask me an interesting question!", "mood_negative"),

            # --- Task/Help ---
            ("can you help me", "Absolutely! 🙌 Tell me what you need and I'll do my best!", "task"),
            ("I need assistance", "I'm here to help! 💪 What's on your mind?", "task"),
            ("help me with something", "Sure thing! What do you need help with?", "task"),

            # --- Fun/Misc ---
            ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs! 😄🐛", "general"),
            ("tell me a fun fact", "Fun fact: The first programmer was Ada Lovelace in the 1840s — "
             "over 100 years before modern computers! 🖥️", "general"),
            ("recommend a book", "📖 Tech: 'Clean Code' by Robert Martin\n"
             "AI: 'Hands-On ML' by Aurélien Géron\nFiction: 'Project Hail Mary' by Andy Weir", "general"),
        ]

    # =========================================================================
    # INTENT TEMPLATES
    # =========================================================================

    def _build_intent_templates(self):
        """Build intent-based response templates."""
        self.intent_templates = {
            "greeting": [
                "Hello! 👋 How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to see you. How can I assist?",
                "Greetings! What can I do for you?",
            ],
            "goodbye": [
                "Goodbye! Have a great day! 👋",
                "See you later! Take care!",
                "Bye! Come back anytime!",
            ],
            "thanks": [
                "You're welcome! 😊",
                "Happy to help!",
                "Anytime! Let me know if you need more.",
            ],
            "question_ai": [
                "That's a great AI question! Let me explain...",
                "AI is fascinating! Here's what I know about that:",
            ],
            "question_general": [
                "Interesting question! Let me think about that...",
                "Good question! Here's what I can share:",
            ],
            "task_request": [
                "I'd love to help! What do you need?",
                "Sure thing! Tell me more about what you need.",
                "Absolutely! What are the details?",
            ],
            "mood_positive": [
                "That's wonderful to hear! 🎉",
                "Great to know you're doing well! 😊",
                "Awesome! Keep up the good vibes!",
            ],
            "mood_negative": [
                "I'm sorry to hear that. Is there anything I can do to help?",
                "I hope things get better soon. I'm here for you.",
                "That sounds tough. Remember, one step at a time. 💙",
            ],
            "about_bot": [
                "I'm an AI Assistant! I use a hybrid system combining embeddings, "
                "intent classification, and smart fallbacks to help you. 🤖",
            ],
            "name_introduction": [
                "Nice to meet you! 😊 How can I help?",
            ],
        }

    def _build_fallback_responses(self):
        """Build fallback and keyword responses."""
        self.fallback_responses = [
            "I'm not quite sure I understand. Could you rephrase that?",
            "Interesting! Could you tell me more about what you mean?",
            "I'd like to help — could you give me a bit more context?",
            "Hmm, I'm not sure how to respond to that. Try asking differently?",
            "That's outside my current knowledge. Could you elaborate?",
        ]

        self.keyword_responses = {
            "python": "Python is a fantastic language! Are you learning it or working on a project?",
            "javascript": "JavaScript powers the web! What are you building with it?",
            "programming": "Programming is valuable! What aspect interests you?",
            "weather": "I can't check real-time weather, but try a weather app! ☁️",
            "joke": "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
            "music": "Music is amazing! What genre are you into? 🎵",
            "game": "Gaming is fun! PC, console, or mobile?",
            "food": "Great topic! Looking for recipe ideas or just chatting? 🍕",
        }
