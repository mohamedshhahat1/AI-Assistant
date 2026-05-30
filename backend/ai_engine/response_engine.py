"""
Response Engine Module - Semantic Similarity Upgrade
=====================================================

An advanced response generation system that uses semantic similarity
(TF-IDF cosine similarity) to find the best response from a knowledge base,
combined with intent-aware response selection and contextual personalization.

Upgrade from static rule-based engine:
  OLD: intent → random.choice(static_responses)
  NEW: message → TF-IDF cosine similarity → best matching response from knowledge base
       + intent-aware fallback + contextual personalization

How it works:
  1. Build a knowledge base of (question, response) pairs
  2. When a user sends a message, vectorize it using TF-IDF
  3. Compute cosine similarity between user message and all knowledge base entries
  4. Return the response with the highest semantic match
  5. Fall back to intent-based templates if similarity is too low
  6. Personalize with user context (name, history)
"""

import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResponseEngine:
    """
    Semantic similarity-based response engine with intent-aware fallback.

    Uses TF-IDF vectorization + cosine similarity to find the most relevant
    response from a knowledge base. Falls back to intent templates when
    similarity is too low.

    Attributes:
        knowledge_base (list): List of (pattern, response) tuples for semantic matching.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer for the knowledge base.
        kb_vectors: TF-IDF matrix of all knowledge base patterns.
        similarity_threshold (float): Minimum cosine similarity to use KB response (default 0.25).
    """

    def __init__(self, similarity_threshold=0.25):
        """
        Initialize the ResponseEngine with knowledge base and TF-IDF model.

        Args:
            similarity_threshold (float): Minimum cosine similarity score (0-1)
                required to use a knowledge base response. Below this threshold,
                the engine falls back to intent-based templates.
                Default: 0.25 (fairly permissive — increase for stricter matching).
        """
        self.similarity_threshold = similarity_threshold

        # Build the knowledge base
        self._build_knowledge_base()

        # Build TF-IDF model from knowledge base patterns
        self._build_tfidf_model()

        # Intent-based fallback templates (used when semantic match is weak)
        self._build_intent_templates()

        # Keyword responses for additional fallback
        self._build_keyword_responses()

    # =========================================================================
    # KNOWLEDGE BASE - (pattern, response) pairs for semantic matching
    # =========================================================================

    def _build_knowledge_base(self):
        """
        Build the knowledge base of (pattern, response) pairs.

        The knowledge base contains diverse question/statement patterns paired
        with appropriate responses. The TF-IDF model learns from the patterns
        and matches new user messages to the most similar pattern.
        """
        self.knowledge_base = [
            # --- Greetings ---
            ("hello", "Hello! 👋 How can I help you today?"),
            ("hi there", "Hi there! What's on your mind?"),
            ("hey how are you", "Hey! I'm doing great, thanks for asking! How about you?"),
            ("good morning", "Good morning! ☀️ Hope you're having a wonderful start to your day!"),
            ("good afternoon", "Good afternoon! How's your day going so far?"),
            ("good evening", "Good evening! 🌙 How can I help you tonight?"),
            ("what's up", "Not much! Just here ready to help. What's up with you?"),
            ("howdy", "Howdy! 🤠 What brings you here today?"),
            ("greetings", "Greetings! I'm at your service. What can I do for you?"),

            # --- Goodbyes ---
            ("goodbye", "Goodbye! Have a wonderful day! 👋"),
            ("see you later", "See you later! Take care of yourself!"),
            ("bye bye", "Bye bye! Come back anytime you need help!"),
            ("good night", "Good night! 🌙 Sweet dreams and see you next time!"),
            ("take care", "You too! Take care and don't hesitate to come back!"),
            ("I'm leaving now", "Alright! It was great chatting with you. Until next time! 👋"),

            # --- Thanks ---
            ("thank you so much", "You're very welcome! 😊 Glad I could help!"),
            ("thanks a lot", "Anytime! That's what I'm here for!"),
            ("I appreciate your help", "I appreciate you reaching out! Let me know if you need anything else."),
            ("thanks for the info", "Happy to share! Knowledge is better when shared. 📚"),
            ("that was helpful", "I'm glad it was helpful! Don't hesitate to ask more."),

            # --- AI & Technology Questions ---
            ("what is artificial intelligence",
             "Artificial Intelligence (AI) is the field of computer science focused on creating "
             "machines that can perform tasks requiring human-like intelligence — such as learning, "
             "reasoning, problem-solving, and understanding language. It encompasses machine learning, "
             "deep learning, natural language processing, computer vision, and more."),
            ("what is machine learning",
             "Machine Learning (ML) is a subset of AI where systems learn patterns from data "
             "without being explicitly programmed. Instead of writing rules, you feed the algorithm "
             "examples and it discovers the rules itself. Types include supervised learning, "
             "unsupervised learning, and reinforcement learning."),
            ("what is deep learning",
             "Deep Learning is a subset of machine learning that uses artificial neural networks "
             "with many layers (hence 'deep'). It excels at complex tasks like image recognition, "
             "natural language processing, and speech recognition. Popular frameworks include "
             "PyTorch and TensorFlow."),
            ("what is natural language processing",
             "Natural Language Processing (NLP) is the branch of AI that helps machines understand, "
             "interpret, and generate human language. Applications include chatbots (like me!), "
             "translation, sentiment analysis, and text summarization."),
            ("what is a neural network",
             "A neural network is a computing system inspired by biological brain neurons. "
             "It consists of layers of interconnected nodes that process information. Input data "
             "flows through hidden layers where patterns are detected, leading to an output. "
             "They're the backbone of deep learning!"),
            ("how does AI work",
             "AI works by combining large datasets with intelligent algorithms that learn "
             "patterns from the data. The basic process is: 1) Collect data, 2) Train a model "
             "to find patterns, 3) Evaluate performance, 4) Deploy for predictions. Different "
             "techniques (neural networks, decision trees, etc.) suit different problems."),
            ("what is python used for",
             "Python is incredibly versatile! It's used for: 🐍 Web development (Django, Flask), "
             "📊 Data science & analytics (pandas, NumPy), 🤖 AI & Machine Learning (PyTorch, sklearn), "
             "🔧 Automation & scripting, 🎮 Game development, and much more. Its readability makes "
             "it perfect for beginners and experts alike."),
            ("what is an API",
             "An API (Application Programming Interface) is a set of rules that allows different "
             "software applications to communicate with each other. Think of it as a waiter in a "
             "restaurant — you (the client) tell the waiter (API) what you want, and they bring "
             "it from the kitchen (server). REST APIs use HTTP methods like GET, POST, PUT, DELETE."),
            ("what is tensorflow",
             "TensorFlow is an open-source deep learning framework developed by Google. It's used "
             "to build and train neural networks for tasks like image classification, NLP, and "
             "reinforcement learning. It supports both research prototyping and production deployment."),
            ("what is pytorch",
             "PyTorch is an open-source deep learning framework developed by Meta (Facebook). "
             "It's known for its dynamic computation graph, intuitive Python interface, and "
             "strong research community. Many cutting-edge AI papers use PyTorch."),

            # --- Programming & CS Questions ---
            ("how to learn programming",
             "Here's a great path to learn programming: 1️⃣ Start with Python (beginner-friendly), "
             "2️⃣ Learn core concepts (variables, loops, functions), 3️⃣ Build small projects, "
             "4️⃣ Learn data structures & algorithms, 5️⃣ Pick a specialization (web, AI, mobile). "
             "Practice daily and don't fear making mistakes — that's how you learn!"),
            ("what is an algorithm",
             "An algorithm is a step-by-step set of instructions for solving a problem. "
             "Think of it like a recipe: it takes inputs, follows specific steps, and produces "
             "an output. Algorithms power everything from search engines to social media feeds."),
            ("what is a database",
             "A database is an organized collection of data stored electronically. It allows you "
             "to efficiently store, retrieve, update, and manage large amounts of information. "
             "Types include relational databases (SQL) and NoSQL databases (MongoDB, Redis)."),
            ("what is version control",
             "Version control (like Git) tracks changes to files over time. It lets you: "
             "save snapshots of your code, revert to earlier versions, collaborate with others, "
             "and maintain parallel versions (branches). GitHub/GitLab host remote repositories."),

            # --- General Knowledge ---
            ("tell me a joke",
             "Why do programmers prefer dark mode? Because light attracts bugs! 😄🐛"),
            ("tell me another joke",
             "Why was the JavaScript developer sad? Because he didn't Node how to Express himself! 😂"),
            ("tell me a fun fact",
             "Fun fact: The first computer programmer was Ada Lovelace in the 1840s — "
             "she wrote algorithms for Charles Babbage's Analytical Engine, over 100 years "
             "before modern computers existed! 🖥️"),
            ("what is the meaning of life",
             "That's the big question! 🤔 Philosophically, many say it's about finding purpose, "
             "building connections, and continuous growth. From a practical standpoint — pursue "
             "what brings you fulfillment. Also, 42 (if you're a Hitchhiker's Guide fan 😄)."),

            # --- Bot Self-Awareness ---
            ("who are you",
             "I'm an AI Assistant built with Python! 🤖 I use natural language processing "
             "to understand your messages and generate helpful responses. I can chat, answer "
             "questions, and even remember our conversations!"),
            ("what can you do",
             "I can do quite a lot! 💪 Chat naturally, answer questions about AI/tech/programming, "
             "remember your name and preferences, detect your mood, and have fun conversations. "
             "I'm always learning to be more helpful!"),
            ("how were you made",
             "I was built with: 🐍 Python for the backend, ⚡ FastAPI for the web server, "
             "🧠 scikit-learn for NLP/intent detection (TF-IDF + cosine similarity), "
             "💾 SQLite for memory, and 🌐 HTML/CSS/JS for the chat interface. No external "
             "AI APIs — I'm 100% self-contained!"),
            ("are you a real person",
             "Nope! I'm an AI — a software program designed to understand and respond to "
             "natural language. But I try my best to be helpful, friendly, and sometimes "
             "even funny! 🤖😊"),

            # --- Mood & Emotion ---
            ("I'm feeling happy today",
             "That's wonderful to hear! 🎉 Happiness is contagious — thanks for sharing "
             "the good vibes! What's making your day so great?"),
            ("I'm feeling sad",
             "I'm sorry to hear that. 💙 It's okay to feel down sometimes. Would you like "
             "to talk about what's bothering you, or shall I try to cheer you up?"),
            ("I'm stressed out",
             "I'm sorry you're feeling stressed. 🌿 Here are some quick tips: take deep breaths, "
             "step away for a short walk, break big tasks into smaller ones, and remember — "
             "it's okay to ask for help. You've got this!"),
            ("I'm bored",
             "Let's fix that! 🎯 Here are some ideas: learn something new (like a programming "
             "concept), try a creative project, read an interesting article, or challenge yourself "
             "with a coding puzzle. Want me to suggest something specific?"),
            ("I feel tired",
             "Rest is important! 😴 Make sure you're getting enough sleep, staying hydrated, "
             "and taking breaks. Sometimes a short 10-minute power nap or a walk outside "
             "can recharge you. Take care of yourself!"),

            # --- Task/Help Requests ---
            ("can you help me with something",
             "Absolutely! I'd love to help. 🙌 Tell me what you need, and I'll do my best "
             "to assist you!"),
            ("I need assistance",
             "I'm here to help! 💪 What's on your mind? Whether it's a question, a task, "
             "or just someone to brainstorm with — fire away!"),
            ("how do I get started",
             "Great question! Getting started depends on what you're working on. Could you "
             "tell me a bit more about your goal? I can suggest a step-by-step approach."),

            # --- Miscellaneous ---
            ("what time is it",
             "I don't have access to a real-time clock, but you can check the time on your "
             "device! ⏰ Is there something time-related I can help with?"),
            ("what's the weather like",
             "I wish I could check the weather for you! ☁️ Unfortunately, I don't have access "
             "to real-time data. Try a weather app or website for the latest forecast."),
            ("recommend a book",
             "Here are some great books depending on your interest: 📖\n"
             "• Tech: 'Clean Code' by Robert Martin\n"
             "• AI: 'Hands-On Machine Learning' by Aurélien Géron\n"
             "• Fiction: 'Project Hail Mary' by Andy Weir\n"
             "• Self-improvement: 'Atomic Habits' by James Clear"),
            ("recommend a movie",
             "Here are some great films! 🎬\n"
             "• Sci-fi: Interstellar, The Matrix, Ex Machina\n"
             "• Drama: The Shawshank Redemption, Forrest Gump\n"
             "• Animation: Spider-Verse, Your Name\n"
             "Want a specific genre recommendation?"),
        ]

    # =========================================================================
    # TF-IDF MODEL - Vectorize knowledge base for semantic matching
    # =========================================================================

    def _build_tfidf_model(self):
        """
        Build and fit the TF-IDF vectorizer on knowledge base patterns.

        This creates a vector representation of each pattern in the knowledge base,
        enabling cosine similarity comparison with new user messages.
        """
        # Extract just the patterns (first element of each tuple)
        self.kb_patterns = [pattern for pattern, _ in self.knowledge_base]
        self.kb_responses = [response for _, response in self.knowledge_base]

        # Create and fit the TF-IDF vectorizer
        # ngram_range=(1,2) captures both single words and two-word phrases
        # This improves matching for phrases like "machine learning" or "deep learning"
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # Unigrams and bigrams
            stop_words='english',     # Remove common words (the, is, at, etc.)
            max_features=5000,        # Limit vocabulary size
            sublinear_tf=True,        # Apply log normalization to term frequency
        )

        # Fit and transform knowledge base patterns into TF-IDF vectors
        self.kb_vectors = self.vectorizer.fit_transform(self.kb_patterns)

        print(f"[ResponseEngine] Knowledge base loaded: {len(self.knowledge_base)} entries")
        print(f"[ResponseEngine] TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"[ResponseEngine] Similarity threshold: {self.similarity_threshold}")

    # =========================================================================
    # INTENT FALLBACK TEMPLATES
    # =========================================================================

    def _build_intent_templates(self):
        """Build intent-based response templates for fallback."""
        self.intent_templates = {
            "greeting": [
                "Hello! 👋 How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to see you. How can I assist?",
            ],
            "goodbye": [
                "Goodbye! Have a great day! 👋",
                "See you later! Take care!",
                "Bye! Come back anytime!",
            ],
            "thanks": [
                "You're welcome! 😊",
                "Happy to help!",
                "Anytime! Let me know if you need anything else.",
            ],
            "task_request": [
                "I'd love to help! What do you need?",
                "Sure thing! Tell me more.",
                "Absolutely! What are the details?",
            ],
            "mood_positive": [
                "That's wonderful to hear! 🎉",
                "Great to know you're doing well!",
                "Awesome! That positive energy is contagious! 😄",
            ],
            "mood_negative": [
                "I'm sorry to hear that. Is there anything I can do to help?",
                "I hope things get better soon. I'm here if you need to talk.",
                "That sounds tough. Remember, it's okay to take things one step at a time.",
            ],
            "name_introduction": [
                "Nice to meet you, {name}! 😊 How can I help you?",
                "Hello, {name}! Great to know your name. What can I do for you?",
                "Hi {name}! 👋 Lovely to meet you. How can I assist today?",
            ],
            "about_bot": [
                "I'm an AI Assistant built with Python! I use NLP to understand "
                "your messages and semantic similarity to find the best responses. 🤖",
            ],
        }

        self.fallback_responses = [
            "I'm not quite sure I understand. Could you rephrase that?",
            "Interesting! Tell me more about what you mean.",
            "I'm still learning! Could you explain that differently?",
            "Hmm, that's outside my current knowledge. Can you try asking another way?",
            "I appreciate your patience — could you give me a bit more context?",
        ]

    def _build_keyword_responses(self):
        """Build keyword-to-response mapping for additional fallback."""
        self.keyword_responses = {
            "python": "Python is a fantastic programming language! Known for readability "
                      "and versatility. Are you learning Python or working on a project?",
            "javascript": "JavaScript powers the web! It runs in browsers and on servers "
                          "(Node.js). What are you building with it?",
            "programming": "Programming is such a valuable skill! What aspect interests "
                           "you — web dev, AI, mobile apps, or something else?",
            "weather": "I don't have access to real-time weather data. Try a weather app!",
            "time": "I can't check the time, but your device can! Is there something "
                    "time-related I can help with?",
            "joke": "Why do programmers prefer dark mode? Because light attracts bugs! 😄",
            "music": "Music is amazing! What genre are you into?",
            "food": "Food is a great topic! Looking for recipe ideas or just chatting?",
            "game": "Gaming is fun! Are you into PC, console, or mobile games?",
            "sport": "Sports are great for health and fun! What's your favorite sport?",
        }

    # =========================================================================
    # MAIN RESPONSE GENERATION
    # =========================================================================

    def generate(self, message, intent, memory=None):
        """
        Generate a response using semantic similarity + intent-aware fallback.

        Strategy (in order of priority):
        1. Handle special cases (name introduction)
        2. Try semantic similarity match from knowledge base
        3. If similarity too low, use intent-based template
        4. If intent unknown, try keyword matching
        5. Last resort: generic fallback

        Args:
            message (str): The user's message.
            intent (str): Detected intent from the IntentDetector.
            memory (dict, optional): User memory containing name, history, etc.

        Returns:
            str: The generated response.
        """
        # Extract user context
        user_name = None
        if memory and memory.get("name"):
            user_name = memory["name"]

        # --- Special Case: Name Introduction ---
        if intent == "name_introduction":
            name = self._extract_name(message)
            if name:
                template = random.choice(self.intent_templates["name_introduction"])
                return template.format(name=name)
            return "Nice to meet you! What should I call you?"

        # --- Primary: Semantic Similarity Matching ---
        response, similarity_score = self._semantic_match(message)

        if similarity_score >= self.similarity_threshold:
            # High similarity — use the matched response
            # Optionally personalize with user name
            if user_name and random.random() < 0.3:
                response = f"{user_name}, " + response[0].lower() + response[1:]
            return response

        # --- Fallback 1: Intent-Based Templates ---
        if intent in self.intent_templates:
            template_response = random.choice(self.intent_templates[intent])
            if user_name and "{name}" not in template_response and random.random() < 0.3:
                template_response = f"{user_name}, " + template_response[0].lower() + template_response[1:]
            return template_response

        # --- Fallback 2: Keyword Matching ---
        keyword_response = self._keyword_fallback(message)
        if keyword_response:
            return keyword_response

        # --- Fallback 3: Generic Response ---
        return random.choice(self.fallback_responses)

    # Keep backward compatibility with old interface
    def generate_response(self, intent, confidence, user_message, context=None):
        """
        Backward-compatible interface for the old ResponseEngine.

        Args:
            intent (str): Detected intent.
            confidence (float): Confidence score.
            user_message (str): User's message.
            context (dict, optional): User context.

        Returns:
            str: Generated response.
        """
        # Convert old context format to new memory format
        memory = None
        if context:
            memory = {"name": context.get("user_name")}
        return self.generate(message=user_message, intent=intent, memory=memory)

    # =========================================================================
    # SEMANTIC MATCHING
    # =========================================================================

    def _semantic_match(self, message):
        """
        Find the most semantically similar knowledge base entry to the user message.

        Uses TF-IDF vectorization and cosine similarity to compare the user's
        message against all patterns in the knowledge base.

        Args:
            message (str): The user's message to match.

        Returns:
            tuple: (best_response, similarity_score)
                - best_response (str): The response paired with the best-matching pattern.
                - similarity_score (float): Cosine similarity (0-1) of the best match.
        """
        # Vectorize the user message using the same TF-IDF model
        message_vector = self.vectorizer.transform([message.lower()])

        # Compute cosine similarity between user message and all KB patterns
        similarities = cosine_similarity(message_vector, self.kb_vectors).flatten()

        # Find the index of the highest similarity score
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Return the corresponding response and its similarity score
        return self.kb_responses[best_idx], float(best_score)

    def get_top_matches(self, message, top_n=3):
        """
        Get the top N most similar knowledge base entries for a message.

        Useful for debugging or showing alternative responses.

        Args:
            message (str): The user's message.
            top_n (int): Number of top matches to return.

        Returns:
            list: List of (pattern, response, score) tuples, sorted by score descending.
        """
        message_vector = self.vectorizer.transform([message.lower()])
        similarities = cosine_similarity(message_vector, self.kb_vectors).flatten()

        # Get indices of top N scores
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "pattern": self.kb_patterns[idx],
                "response": self.kb_responses[idx],
                "similarity": round(float(similarities[idx]), 4),
            })

        return results

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _keyword_fallback(self, message):
        """
        Try keyword matching when semantic similarity fails.

        Args:
            message (str): User message to scan for keywords.

        Returns:
            str or None: Matched response, or None if no keywords found.
        """
        message_lower = message.lower()
        for keyword, response in self.keyword_responses.items():
            if keyword in message_lower:
                return response
        return None

    def _extract_name(self, message):
        """
        Extract a name from introduction messages.

        Args:
            message (str): Message like "my name is Ahmed" or "call me Sara".

        Returns:
            str or None: Extracted name (capitalized) or None.
        """
        patterns = [
            r"(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)",
            r"(?:name's|they call me|people call me)\s+([a-zA-Z]+)",
        ]

        message_lower = message.lower()
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1).capitalize()
        return None
