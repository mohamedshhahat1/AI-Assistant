"""
Hybrid Response Engine for AI Assistant
========================================

This module implements a HYBRID decision system that combines three methods
to generate the most accurate response:

  1. RAG RETRIEVAL — Uses the RAG Engine for semantic search over a dynamic,
     self-learning knowledge base (replaces the old static TF-IDF index).

  2. INTENT CLASSIFICATION — Understands the user's INTENT
     Uses ML classification (TF-IDF + LogReg) to categorize messages
     into intents: greeting, question, command, etc.

  3. FALLBACK GENERATION — Safety net when confidence is low
     Uses keyword matching or asks clarification questions.

Decision Matrix:
  ┌────────────────────┬──────────────────┬────────────────────────────┐
  │ RAG Score          │ Intent Confidence│ Action                     │
  ├────────────────────┼──────────────────┼────────────────────────────┤
  │ >= 0.6             │ any              │ USE RAG (strong match)     │
  │ 0.35 - 0.6        │ intent confirms  │ USE RAG (confirmed)        │
  │ < 0.35            │ >= 0.7           │ USE INTENT TEMPLATE        │
  │ low                │ low              │ USE FALLBACK               │
  └────────────────────┴──────────────────┴────────────────────────────┘

Why Hybrid?
  - Not reliant on a single method
  - More accurate (systems confirm each other)
  - Fewer errors (multiple safety nets)
  - Dynamic learning via RAG engine
"""

import random
import re
import numpy as np

from .rag_engine import RAGEngine


class HybridEngine:
    """
    Hybrid decision engine combining RAG retrieval + intent + fallback.

    Processes every message through all three systems simultaneously,
    then applies a decision matrix to select the best response.
    """

    # Mapping from RAG topic categories to expected intents
    CATEGORY_INTENT_MAP = {
        "greeting": ["greeting"],
        "goodbye": ["goodbye"],
        "thanks": ["thanks"],
        "ai_ml": ["question_ai"],
        "programming": ["question_general", "question_ai"],
        "web_dev": ["question_general", "question_ai"],
        "data": ["question_general", "question_ai"],
        "mood_positive": ["mood_positive"],
        "mood_negative": ["mood_negative"],
        "about_bot": ["about_bot"],
        "task": ["task_request"],
        "general": ["question_general"],
    }

    def __init__(self):
        """Initialize all subsystems including RAG engine."""
        # Initialize RAG Engine as the PRIMARY retrieval system
        self.rag_engine = RAGEngine()

        # Build intent templates and fallback responses
        self._build_intent_templates()
        self._build_fallback_responses()

        print("[HybridEngine] Retrieval: RAG Engine (TF-IDF + dynamic learning)")
        print(f"[HybridEngine] Knowledge base: {self.rag_engine.get_stats()['total_chunks']} chunks")
        print("[HybridEngine] Mode: HYBRID (RAG + intent + fallback)")

    # =========================================================================
    # MAIN PROCESSING PIPELINE
    # =========================================================================

    def process(self, message, intent_result=None, memory=None, context_engine=None, memory_system=None, user_id=None):
        """
        Main hybrid processing pipeline.

        Runs all three systems and picks the best response using the decision matrix.
        When context_engine and memory_system are provided, builds rich context to
        enable follow-up resolution and personalized responses.

        Args:
            message (str): User's message.
            intent_result (dict, optional): Pre-computed intent {"intent": "...", "confidence": ...}.
            memory (dict, optional): User memory/context.
            context_engine (ContextEngine, optional): Context engine for reference resolution.
            memory_system (AdvancedMemorySystem, optional): Memory system for context building.
            user_id (str, optional): User ID for context building.

        Returns:
            dict: {
                "response": "The chosen response text",
                "method": "embedding" | "intent" | "fallback",
                "confidence": 0.85,
                "intent": "question_ai",
                "embedding_score": 0.72,
                "intent_confidence": 0.85,
                "reasoning": "Strong semantic match (0.72)",
                "context": {...}  # Rich context if context_engine was used
            }
        """
        # --- Context Integration ---
        # If context_engine and memory_system are provided, build rich context
        rich_context = None
        retrieval_message = message  # Message used for RAG retrieval

        if context_engine and memory_system and user_id:
            rich_context = context_engine.build_context(user_id, message, memory_system)

            # If this is a follow-up, use the resolved message for retrieval
            if rich_context.get("is_follow_up") and rich_context.get("resolved_message") != message:
                retrieval_message = rich_context["resolved_message"]

        # Extract user context
        user_name = None
        if memory:
            user_name = memory.get("name") or memory.get("user_name")
        # Also check rich_context for user_name
        if not user_name and rich_context:
            user_name = rich_context.get("user_name")

        # --- System 1: RAG Retrieval ---
        embed_response, embed_score, embed_category = self._retrieval_match(retrieval_message)

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

        # Decision 1: Strong RAG match (>= 0.6)
        if embed_score >= 0.6:
            response = embed_response
            method = "embedding"
            final_confidence = embed_score
            reasoning = f"Strong semantic match ({embed_score:.2f})"

        # Decision 2: Medium RAG + intent confirms
        elif embed_score >= 0.35 and self._intent_confirms_embedding(intent, embed_category):
            response = embed_response
            method = "embedding"
            final_confidence = (embed_score + intent_confidence) / 2
            reasoning = (f"Medium semantic match ({embed_score:.2f}) "
                        f"confirmed by intent ({intent}, {intent_confidence:.2f})")

        # Decision 3: Low RAG but strong intent
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

        # --- Context-based Personalization ---
        if rich_context and rich_context.get("personalization_hints"):
            response = self._apply_personalization(response, rich_context)

        # --- Handle name introduction specially ---
        if intent == "name_introduction":
            name = self._extract_name(message)
            if name:
                response = random.choice([
                    f"Nice to meet you, {name}! How can I help you?",
                    f"Hello, {name}! Great to know your name. What can I do for you?",
                    f"Hi {name}! Lovely to meet you!",
                ])
                method = "intent"
                final_confidence = 0.95
                reasoning = f"Name introduction detected: {name}"

        # --- Dynamic Learning ---
        # After generating a response, attempt to learn from the user's message
        # if it looks like factual/informational content
        if intent_result:
            self.rag_engine.learn_from_message(message, intent, intent_confidence)

        result = {
            "response": response,
            "method": method,
            "confidence": round(final_confidence, 3),
            "intent": intent,
            "embedding_score": round(embed_score, 3),
            "intent_confidence": round(intent_confidence, 3),
            "reasoning": reasoning,
        }

        # Include rich context in result if available
        if rich_context:
            result["context"] = rich_context

        return result

    # =========================================================================
    # SYSTEM 1: RAG RETRIEVAL
    # =========================================================================

    def _retrieval_match(self, message):
        """
        Find the most semantically similar knowledge chunk using the RAG engine.

        Returns:
            tuple: (response, score, category)
        """
        # Use RAG engine's get_best_response for retrieval
        content, score = self.rag_engine.get_best_response(message)

        if content is not None:
            # Detect category from the retrieved content
            category = self.rag_engine._detect_topic(content)
            return (content, score, category)
        else:
            # RAG returned no confident match
            # Still return the score so the decision matrix can use it
            return ("", score, "unknown")

    # =========================================================================
    # SYSTEM 2: INTENT CONFIRMATION
    # =========================================================================

    def _intent_confirms_embedding(self, intent, kb_category):
        """
        Check if the detected intent aligns with the RAG entry's category.

        For example, if the RAG matched an AI question pattern AND
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

        # Get top 3 RAG matches
        top_results = self.rag_engine.search(message, top_k=3, min_score=0.0)
        top_matches = []
        for r in top_results:
            top_matches.append({
                "pattern": r["content"][:100],
                "category": r["topic"],
                "score": r["score"]
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
    # INTENT TEMPLATES
    # =========================================================================

    def _build_intent_templates(self):
        """Build intent-based response templates."""
        self.intent_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to see you. How can I assist?",
                "Greetings! What can I do for you?",
            ],
            "goodbye": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Come back anytime!",
            ],
            "thanks": [
                "You're welcome!",
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
                "That's wonderful to hear!",
                "Great to know you're doing well!",
                "Awesome! Keep up the good vibes!",
            ],
            "mood_negative": [
                "I'm sorry to hear that. Is there anything I can do to help?",
                "I hope things get better soon. I'm here for you.",
                "That sounds tough. Remember, one step at a time.",
            ],
            "about_bot": [
                "I'm an AI Assistant! I use a hybrid system combining RAG retrieval, "
                "intent classification, and smart fallbacks to help you.",
            ],
            "name_introduction": [
                "Nice to meet you! How can I help?",
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
            "weather": "I can't check real-time weather, but try a weather app!",
            "joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
            "music": "Music is amazing! What genre are you into?",
            "game": "Gaming is fun! PC, console, or mobile?",
            "food": "Great topic! Looking for recipe ideas or just chatting?",
        }

    # =========================================================================
    # CONTEXT-BASED PERSONALIZATION
    # =========================================================================

    def _apply_personalization(self, response, rich_context):
        """
        Apply personalization to the response based on rich context hints.

        Uses the personalization_hints and relevant_history from the context
        to adjust the response style and add contextual references.

        Args:
            response (str): The original response text.
            rich_context (dict): The rich context from ContextEngine.

        Returns:
            str: The personalized response.
        """
        hints = rich_context.get("personalization_hints", [])
        relevant_history = rich_context.get("relevant_history", [])
        is_follow_up = rich_context.get("is_follow_up", False)

        # If this is a follow-up and we have history, acknowledge continuity
        if is_follow_up and relevant_history and len(response) > 50:
            # For longer responses on follow-ups, add a continuity prefix
            continuity_prefixes = [
                "Continuing on that topic — ",
                "Building on what we discussed — ",
                "To expand on that — ",
                "Going deeper on this — ",
            ]
            # Only add prefix if response doesn't already start with a name or greeting
            if not response[0].isupper() or response.split()[0].endswith(","):
                pass  # Skip if already personalized with name
            elif not any(response.startswith(p) for p in ["Hello", "Hi", "Hey", "Sure"]):
                response = random.choice(continuity_prefixes) + response[0].lower() + response[1:]

        # For concise users, trim long responses
        if any("concise" in h for h in hints) and len(response) > 200:
            # Try to find a natural break point
            sentences = response.split(". ")
            if len(sentences) > 2:
                response = ". ".join(sentences[:2]) + "."

        return response
