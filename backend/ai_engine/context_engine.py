"""
Context Memory Integration Engine
==================================

This module solves the CORE problem of conversational AI: context continuity.

Without this module, each message is processed independently:
  User: "I like AI"
  User: "explain this more"  → System has NO idea what "this" refers to

With ContextEngine, the system:
  1. Tracks conversation topic across messages
  2. Resolves pronouns/references using recent history
  3. Injects user profile into response generation
  4. Makes responses personalized based on interests/style

The ContextEngine builds a RICH context object before every response generation,
giving the response engine everything it needs to produce personalized, contextual replies.

Usage:
    context_engine = ContextEngine()
    context = context_engine.build_context(user_id, message, memory_system)
    # context["resolved_message"] → use for retrieval
    # context["personalization_hints"] → use for tone/style
    # context["is_follow_up"] → True if referencing previous message
"""

import re
from collections import Counter


class ContextEngine:
    """
    Context Memory Integration Engine that makes the AI truly remember users
    and understand conversational references.

    Bridges the gap between the AdvancedMemorySystem (storage) and the
    HybridEngine (response generation) by building rich context objects
    that enable personalized, contextual responses.
    """

    # Topic keywords mapped to canonical topic names
    TOPIC_KEYWORDS = {
        "artificial_intelligence": [
            "ai", "artificial intelligence", "machine learning", "ml",
            "deep learning", "neural network", "nlp", "natural language",
            "computer vision", "reinforcement learning", "gpt", "llm",
            "transformer", "model training", "chatbot", "intelligent"
        ],
        "programming": [
            "code", "coding", "programming", "developer", "software",
            "algorithm", "function", "variable", "debug", "compile",
            "syntax", "library", "framework", "git", "api", "backend",
            "frontend", "fullstack", "ide", "terminal"
        ],
        "python": [
            "python", "pip", "django", "flask", "fastapi", "pandas",
            "numpy", "pytorch", "tensorflow", "jupyter", "virtualenv"
        ],
        "javascript": [
            "javascript", "js", "node", "react", "vue", "angular",
            "typescript", "npm", "webpack", "express", "dom"
        ],
        "web_development": [
            "html", "css", "web", "website", "browser", "http",
            "rest", "graphql", "responsive", "bootstrap", "tailwind"
        ],
        "data_science": [
            "data science", "data analysis", "statistics", "visualization",
            "dataset", "csv", "sql", "database", "analytics", "dashboard"
        ],
        "career": [
            "job", "career", "interview", "resume", "cv", "salary",
            "hire", "hiring", "work", "internship", "company"
        ],
        "education": [
            "learn", "study", "course", "university", "school",
            "exam", "homework", "tutorial", "lesson", "student",
            "teacher", "professor", "degree"
        ],
        "greeting": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "howdy", "what's up", "sup"
        ],
        "personal": [
            "my name", "i am", "i like", "i love", "i hate",
            "i feel", "my hobby", "my favorite", "about me"
        ],
    }

    # Words that indicate the message references something previous
    REFERENCE_WORDS = [
        "this", "that", "it", "those", "them", "these",
        "more", "else", "another", "other", "same",
        "above", "previous", "last", "earlier"
    ]

    # Follow-up starter words
    FOLLOW_UP_STARTERS = [
        "yes", "no", "ok", "okay", "sure", "also", "and",
        "but", "so", "then", "right", "exactly", "indeed",
        "absolutely", "definitely", "of course", "please",
        "yep", "yeah", "nah", "nope", "true", "correct"
    ]

    # Patterns that indicate follow-up without introducing a new topic
    FOLLOW_UP_PATTERNS = [
        r"\bmore\b", r"\belse\b", r"\banother\b",
        r"\babout that\b", r"\babout it\b", r"\babout this\b",
        r"\bgo on\b", r"\bcontinue\b", r"\bkeep going\b",
        r"\bwhat about\b", r"\bhow about\b",
        r"\btell me more\b", r"\bexplain more\b",
        r"\bcan you elaborate\b", r"\belaborate\b",
        r"\bexpand on\b", r"\bwhat else\b",
    ]

    def __init__(self):
        """
        Initialize the ContextEngine.

        Sets up the topic tracker and reference resolver with their
        required keyword maps and pattern matchers.
        """
        # Topic tracker state (per-user, in-memory for the session)
        self._topic_history = {}  # user_id -> list of topics
        self._last_topic = {}    # user_id -> last detected topic

    def build_context(self, user_id, current_message, memory_system):
        """
        Build a RICH context object that gives the response engine everything
        it needs to generate a personalized, contextual response.

        This is the main entry point called before every response generation.

        Pipeline:
          1. Load user profile from memory system
          2. Get recent STM messages (last 10)
          3. Detect if current message is a follow-up (references previous topic)
          4. Resolve references ("this", "that", "it", "more") using context
          5. Track current topic
          6. Generate personalization hints

        Args:
            user_id (str): The user's unique identifier.
            current_message (str): What the user just said.
            memory_system: The AdvancedMemorySystem instance.

        Returns:
            dict: Rich context object with keys:
                - user_name (str or None): User's known name
                - resolved_message (str): Message with references resolved
                - current_topic (str): Detected current topic
                - topic_history (list): Previous topics in this session
                - conversation_summary (str): Brief summary of recent conversation
                - user_interests (list): User's known interests
                - communication_style (str): User's preferred style
                - personality_tags (list): User's personality traits
                - relevant_history (list): Recent relevant messages
                - session_mood (str): Current session mood
                - is_follow_up (bool): Whether this references previous message
                - personalization_hints (list): Hints for response personalization
        """
        # Step 1: Load user profile from memory system
        profile = memory_system.get_user_profile(user_id)

        # Step 2: Get recent STM messages (last 10)
        session_id = memory_system.get_active_session(user_id)
        recent_messages = []
        if session_id:
            recent_messages = memory_system.get_stm(session_id, limit=10)

        # Step 3: Detect if current message is a follow-up
        follow_up = self.is_follow_up(current_message)

        # Step 4: Resolve references using conversation history
        resolved_message = self.resolve_references(current_message, recent_messages)

        # Step 5: Track current topic
        previous_topic = self._last_topic.get(user_id)
        current_topic = self.detect_topic(current_message, previous_topic=previous_topic)

        # Update topic tracking state
        if user_id not in self._topic_history:
            self._topic_history[user_id] = []
        if current_topic != "unknown":
            self._topic_history[user_id].append(current_topic)
            self._last_topic[user_id] = current_topic
        elif previous_topic and follow_up:
            # If follow-up with no clear topic, keep the previous topic
            current_topic = previous_topic

        # Step 6: Generate personalization hints
        personalization_hints = self.generate_personalization_hints(profile, current_topic)

        # Build conversation summary
        conversation_summary = self.get_conversation_summary(recent_messages)

        # Get relevant history (user messages only, last 5)
        relevant_history = [
            msg["message"] for msg in recent_messages
            if msg["role"] == "user"
        ][-5:]

        # Determine session mood from memory context
        session_mood = "neutral"
        if session_id:
            context = memory_system.get_context(user_id, session_id)
            session_mood = context.get("session_mood", "neutral")

        # Get topic history (last 10 topics)
        topic_history = self._topic_history.get(user_id, [])[-10:]

        return {
            "user_name": profile.get("name"),
            "resolved_message": resolved_message,
            "current_topic": current_topic,
            "topic_history": topic_history,
            "conversation_summary": conversation_summary,
            "user_interests": profile.get("interests", []),
            "communication_style": profile.get("communication_style", "casual"),
            "personality_tags": profile.get("personality_tags", []),
            "relevant_history": relevant_history,
            "session_mood": session_mood,
            "is_follow_up": follow_up,
            "personalization_hints": personalization_hints,
        }

    def resolve_references(self, message, recent_messages):
        """
        Resolve pronouns and vague references using conversation history.

        When a user says "explain this more", the system needs to figure out
        what "this" refers to by looking at recent conversation history.

        Strategy:
          - Detect reference words: this, that, it, those, them, more, else, another
          - If found, look at last 3-5 messages for the topic
          - Extract the topic noun/phrase from recent history
          - Replace or append the resolved topic to the message
          - If no reference found, return original message unchanged

        Examples:
          "explain this more" + history has "AI" topic -> "explain AI more"
          "tell me more about it" + last topic was "Python" -> "tell me more about Python"
          "what else can you tell me" + topic was "neural networks"
              -> "what else can you tell me about neural networks"
          "yes please" + last bot asked "want to know about ML?" -> "yes tell me about ML"

        Args:
            message (str): The current user message.
            recent_messages (list): Recent messages from STM, each with
                                    keys: role, message, intent, timestamp.

        Returns:
            str: The message with references resolved, or original if no
                 references detected.
        """
        if not recent_messages:
            return message

        msg_lower = message.lower().strip()

        # Check if the message contains any reference words
        has_reference = False
        for ref_word in self.REFERENCE_WORDS:
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(ref_word) + r'\b', msg_lower):
                has_reference = True
                break

        # Also check if it's a very short affirming message
        is_short_affirm = msg_lower in [
            "yes", "yes please", "sure", "ok", "okay", "yep", "yeah",
            "tell me", "go on", "continue", "go ahead"
        ]

        if not has_reference and not is_short_affirm:
            return message

        # Extract the topic from recent messages (look at last 3-5 user messages)
        topic_phrase = self._extract_topic_from_history(recent_messages)

        if not topic_phrase:
            return message

        # Now resolve the reference by injecting the topic
        resolved = self._inject_topic_into_message(message, msg_lower, topic_phrase)
        return resolved

    def detect_topic(self, message, previous_topic=None):
        """
        Detect the current conversation topic from the message.

        Uses keyword detection against a curated topic map, considering
        topic continuity (if message is vague, likely same topic as before).

        Args:
            message (str): The user's message.
            previous_topic (str, optional): The previous topic for continuity.

        Returns:
            str: A topic string like "artificial_intelligence", "programming",
                 "greeting", "personal", "unknown".
        """
        msg_lower = message.lower()

        # Score each topic by keyword matches
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in msg_lower:
                    # Longer keywords get more weight (more specific)
                    score += len(keyword)
            if score > 0:
                topic_scores[topic] = score

        # If we found matching topics, return the highest scoring one
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic

        # If no topic detected and message is short/vague, maintain previous topic
        if previous_topic and len(message.split()) < 6:
            return previous_topic

        return "unknown"

    def is_follow_up(self, message):
        """
        Detect if the message is a follow-up to previous conversation.

        A follow-up message references or continues the previous topic
        without introducing a new clear topic of its own.

        Follow-up indicators:
          - Short message (<5 words) that contains reference words
          - Starts with "yes", "no", "ok", "sure", "also", "and"
          - Contains "more", "else", "another", "about that"
          - Doesn't introduce a new clear topic

        Args:
            message (str): The user's message to check.

        Returns:
            bool: True if the message is a follow-up to a previous message.
        """
        msg_lower = message.lower().strip()
        words = msg_lower.split()

        # Very short messages with reference words are almost always follow-ups
        if len(words) < 5:
            for ref_word in self.REFERENCE_WORDS:
                if re.search(r'\b' + re.escape(ref_word) + r'\b', msg_lower):
                    return True

        # Messages starting with follow-up starters
        if words:
            first_word = words[0].rstrip(",.:;!?")
            if first_word in self.FOLLOW_UP_STARTERS:
                # Only if the message doesn't clearly introduce a new topic
                if not self._introduces_new_topic(message):
                    return True

        # Messages matching follow-up patterns
        for pattern in self.FOLLOW_UP_PATTERNS:
            if re.search(pattern, msg_lower):
                return True

        # Very short messages (1-3 words) without a clear topic are likely follow-ups
        if len(words) <= 3 and not self._introduces_new_topic(message):
            # Unless it's a greeting or clearly standalone
            if msg_lower not in ["hello", "hi", "hey", "bye", "goodbye", "thanks"]:
                return True

        return False

    def generate_personalization_hints(self, user_profile, current_topic):
        """
        Generate hints that the response engine can use to personalize responses.

        These hints guide the tone, depth, and style of the response based on
        what we know about the user and the current conversation context.

        Args:
            user_profile (dict): The user's profile from memory system containing
                                 personality_tags, interests, communication_style, etc.
            current_topic (str): The detected current conversation topic.

        Returns:
            list: Personalization hint strings, e.g.:
                - "User is interested in AI - provide deeper explanations"
                - "User prefers casual tone"
                - "User is new, be welcoming"
        """
        hints = []

        interests = user_profile.get("interests", [])
        style = user_profile.get("communication_style", "casual")
        personality = user_profile.get("personality_tags", [])
        total_messages = user_profile.get("total_messages", 0)
        total_sessions = user_profile.get("total_sessions", 0)
        name = user_profile.get("name")

        # Personalization based on communication style
        if style == "formal":
            hints.append("User prefers formal communication - maintain professional tone")
        elif style == "casual":
            hints.append("User prefers casual tone - keep it friendly and informal")
        elif style == "technical":
            hints.append("User is technical - use precise terminology and code examples")

        # Personalization based on interests matching current topic
        if current_topic != "unknown":
            topic_readable = current_topic.replace("_", " ")
            matching_interests = [
                i for i in interests
                if i.lower() in topic_readable or topic_readable in i.lower()
            ]
            if matching_interests:
                hints.append(
                    f"User is interested in {', '.join(matching_interests)} - "
                    f"provide deeper explanations and advanced details"
                )
            else:
                # Topic is not in their known interests - might be new to them
                if interests:
                    hints.append(
                        f"Topic '{topic_readable}' is not in user's known interests "
                        f"({', '.join(interests[:3])}) - provide accessible explanation"
                    )

        # Personalization based on personality tags
        if "curious" in personality:
            hints.append("User is curious - offer additional related information")
        if "technical" in personality:
            hints.append("User is technical - include implementation details")
        if "concise" in personality:
            hints.append("User prefers concise responses - be brief and direct")
        if "detail-oriented" in personality:
            hints.append("User likes detail - provide comprehensive explanations")
        if "creative" in personality:
            hints.append("User is creative - suggest innovative approaches")
        if "friendly" in personality:
            hints.append("User is friendly - match their warm tone")

        # Personalization based on experience with the system
        if total_messages < 5:
            hints.append("User is new (< 5 messages) - be welcoming and helpful")
        elif total_sessions > 5:
            hints.append("User is a regular - reference familiarity, skip basic intros")

        # Returning user hint
        if total_sessions > 1 and name:
            hints.append(f"Welcome back {name} - reference previous conversations if relevant")

        return hints

    def get_conversation_summary(self, recent_messages, max_length=100):
        """
        Generate a brief summary of recent conversation.

        Creates a human-readable summary of what has been discussed,
        useful for providing context to the response engine.

        Args:
            recent_messages (list): Recent messages from STM.
            max_length (int): Maximum character length for the summary.

        Returns:
            str: Brief summary, e.g.:
                 "User asked about AI, then machine learning. Currently exploring NLP."
        """
        if not recent_messages:
            return "New conversation - no prior context."

        # Extract user messages only
        user_messages = [
            msg["message"] for msg in recent_messages
            if msg["role"] == "user"
        ]

        if not user_messages:
            return "Conversation started but no user messages yet."

        # Build summary from user messages
        parts = []
        for msg in user_messages[-5:]:  # Last 5 user messages
            # Truncate long messages
            short = msg[:50] + "..." if len(msg) > 50 else msg
            parts.append(short)

        # Detect topics discussed
        topics_mentioned = set()
        for msg in user_messages:
            topic = self.detect_topic(msg)
            if topic != "unknown":
                topics_mentioned.add(topic.replace("_", " "))

        if topics_mentioned:
            summary = f"Topics discussed: {', '.join(list(topics_mentioned)[:3])}. "
            summary += f"Recent: \"{parts[-1]}\""
        else:
            summary = f"User said: \"{parts[-1]}\""

        # Truncate to max_length
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        return summary

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _extract_topic_from_history(self, recent_messages):
        """
        Extract the most likely topic phrase from recent conversation history.

        Looks at the last 3-5 messages (prioritizing user messages) and
        identifies the main subject being discussed.

        Args:
            recent_messages (list): Recent STM messages.

        Returns:
            str or None: The extracted topic phrase, or None if no clear topic.
        """
        # Look at last 5 messages, prioritize user messages
        last_messages = recent_messages[-5:]

        # First, try to find topic from the most recent user message
        for msg in reversed(last_messages):
            if msg["role"] == "user":
                text = msg["message"]
                # Try to extract a meaningful topic phrase
                topic_phrase = self._extract_noun_phrase(text)
                if topic_phrase:
                    return topic_phrase

        # If no user message has a clear topic, check assistant messages
        # (e.g., "Would you like to know about machine learning?")
        for msg in reversed(last_messages):
            if msg["role"] == "assistant":
                text = msg["message"]
                topic_phrase = self._extract_topic_from_assistant(text)
                if topic_phrase:
                    return topic_phrase

        return None

    def _extract_noun_phrase(self, text):
        """
        Extract the main noun phrase / topic from a user message.

        Uses keyword matching against known topics and simple heuristics
        to identify what the user is talking about.

        Args:
            text (str): The message text.

        Returns:
            str or None: The extracted topic phrase.
        """
        text_lower = text.lower()

        # Skip messages that are themselves just reference words
        reference_only = {"this", "that", "it", "those", "them", "these",
                          "more", "else", "another", "other", "same"}
        stripped_words = set(text_lower.split())
        filler_words = {"tell", "me", "about", "explain", "more", "what",
                        "can", "you", "the", "a", "an", "is", "please",
                        "i", "want", "to", "know"}
        meaningful_words = stripped_words - filler_words - reference_only
        if not meaningful_words:
            return None

        # First check against known topic keywords for a canonical match
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower and len(keyword) > 2:
                    # Return the keyword as-is (it's already a meaningful phrase)
                    return keyword

        # Try to extract "about X" pattern
        about_match = re.search(r'\babout\s+(.+?)(?:\?|!|$|,|\.)', text, re.IGNORECASE)
        if about_match:
            phrase = about_match.group(1).strip()
            # Filter out if phrase is just a reference word
            if phrase.lower() not in reference_only and 2 <= len(phrase.split()) <= 5:
                return phrase

        # Try to extract from "what is X" / "explain X" / "tell me about X" patterns
        patterns = [
            r'\bwhat (?:is|are)\s+(.+?)(?:\?|!|$)',
            r'\bexplain\s+(.+?)(?:\?|!|$|,)',
            r'\btell me (?:about|more about)\s+(.+?)(?:\?|!|$)',
            r'\bhow (?:does|do|to)\s+(.+?)(?:\?|!|$|,)',
            r'\bwhat (?:can|does)\s+(.+?)(?:\?|!|$)',
            r'\bI (?:like|love|enjoy|am interested in)\s+(.+?)(?:\?|!|$|,|\.)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phrase = match.group(1).strip()
                # Clean up and validate
                phrase = re.sub(r'\b(the|a|an)\b', '', phrase).strip()
                if phrase and phrase.lower() not in reference_only and len(phrase.split()) <= 5:
                    return phrase

        # Fallback: if the message is short enough (1-4 words), use it as the topic
        words = text.strip().split()
        if 1 <= len(words) <= 4:
            # Filter out common non-topic words and reference words
            non_topic = {"i", "you", "me", "the", "a", "an", "is", "are", "it", "do",
                         "this", "that", "those", "them", "these", "more", "else"}
            topic_words = [w for w in words if w.lower() not in non_topic]
            if topic_words:
                return " ".join(topic_words)

        return None

    def _extract_topic_from_assistant(self, text):
        """
        Extract topic from an assistant message (e.g., questions the bot asked).

        Looks for patterns like "about X" or "regarding X" in bot messages.

        Args:
            text (str): The assistant's message.

        Returns:
            str or None: Extracted topic phrase.
        """
        patterns = [
            r'\babout\s+(.+?)(?:\?|!|$|,|\.)',
            r'\bregarding\s+(.+?)(?:\?|!|$|,|\.)',
            r'\bknow (?:about|more about)\s+(.+?)(?:\?|!|$)',
            r'\binterested in\s+(.+?)(?:\?|!|$|,|\.)',
            r'\blearn (?:about|more about)\s+(.+?)(?:\?|!|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phrase = match.group(1).strip()
                if phrase and len(phrase.split()) <= 5:
                    return phrase
        return None

    def _inject_topic_into_message(self, original_message, msg_lower, topic_phrase):
        """
        Inject the resolved topic into the message, replacing references.

        Handles different cases:
          - "explain this more" -> "explain {topic} more"
          - "tell me more about it" -> "tell me more about {topic}"
          - "yes please" -> "yes, tell me about {topic}"
          - "what else" -> "what else about {topic}"

        Args:
            original_message (str): The original message (preserving case).
            msg_lower (str): Lowercased message for pattern matching.
            topic_phrase (str): The resolved topic to inject.

        Returns:
            str: The message with the topic injected.
        """
        # Pattern: "explain/tell X more" - replace reference word with topic
        for ref_word in ["this", "that", "it", "those", "them", "these"]:
            pattern = r'\b' + re.escape(ref_word) + r'\b'
            if re.search(pattern, msg_lower):
                # Replace the reference word with the topic
                resolved = re.sub(pattern, topic_phrase, original_message, count=1,
                                  flags=re.IGNORECASE)
                return resolved

        # Pattern: "more" / "else" without a clear object - append topic
        if re.search(r'\bmore\b', msg_lower) and "about" not in msg_lower:
            return original_message + f" about {topic_phrase}"

        if re.search(r'\belse\b', msg_lower) and "about" not in msg_lower:
            return original_message + f" about {topic_phrase}"

        # Pattern: short affirming messages - expand with topic
        short_affirms = [
            "yes", "yes please", "sure", "ok", "okay", "yep", "yeah",
            "tell me", "go on", "continue", "go ahead"
        ]
        if msg_lower in short_affirms:
            return f"{original_message}, tell me about {topic_phrase}"

        # Default: append "about {topic}" if nothing else matched
        if "about" not in msg_lower:
            return original_message + f" about {topic_phrase}"

        return original_message

    def _introduces_new_topic(self, message):
        """
        Check if a message introduces a clearly new topic.

        Used to distinguish between follow-ups and topic changes.

        Args:
            message (str): The message to check.

        Returns:
            bool: True if the message introduces a new topic.
        """
        msg_lower = message.lower()

        # Count how many topic keywords match
        topic_matches = 0
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in msg_lower and len(keyword) > 3:
                    topic_matches += 1

        # If multiple strong keyword matches, it's likely a new topic
        return topic_matches >= 2
