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
        "ai_ml": ["question_ai", "learning_request"],
        "programming": ["question_general", "question_ai", "question_programming", "learning_request"],
        "web_dev": ["question_general", "question_ai", "question_web", "learning_request"],
        "data": ["question_general", "question_ai", "question_programming"],
        "mood_positive": ["mood_positive"],
        "mood_negative": ["mood_negative"],
        "about_bot": ["about_bot"],
        "task": ["task_request"],
        "general": ["question_general", "learning_request"],
    }

    def __init__(self):
        """Initialize all subsystems including RAG engine."""
        # Initialize RAG Engine as the PRIMARY retrieval system
        self.rag_engine = RAGEngine()

        # Build intent templates and fallback responses
        self._build_intent_templates()
        self._build_fallback_responses()
        self._build_arabic_synonym_bridge()

        print("[HybridEngine] Retrieval: RAG Engine (TF-IDF + dynamic learning)")
        print(f"[HybridEngine] Knowledge base: {self.rag_engine.get_stats()['total_chunks']} chunks")
        print("[HybridEngine] Mode: HYBRID (RAG + intent + fallback + Arabic bridge)")

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
        # First try direct retrieval on the message (works for both Arabic & English)
        embed_response, embed_score, embed_category = self._retrieval_match(retrieval_message)

        # --- System 1b: Arabic Semantic Bridge ---
        # If RAG score is low, try translating Arabic keywords to English equivalents
        # This bridges "ايه هو AI" → "what is artificial intelligence" for better matching
        bridge_response, bridge_score, bridge_category = None, 0.0, "unknown"
        bridged_query = self._arabic_to_english_bridge(retrieval_message)

        if bridged_query and bridged_query != retrieval_message:
            bridge_response, bridge_score, bridge_category = self._retrieval_match(bridged_query)

            # Use the bridged result if it's significantly better
            if bridge_score > embed_score + 0.05:
                embed_response = bridge_response
                embed_score = bridge_score
                embed_category = bridge_category

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

        # Decision 2: Medium RAG + intent confirms (lowered to 0.20 for Arabic TF-IDF)
        elif embed_score >= 0.20 and self._intent_confirms_embedding(intent, embed_category):
            response = embed_response
            method = "embedding"
            final_confidence = (embed_score + intent_confidence) / 2
            reasoning = (f"Semantic match ({embed_score:.2f}) "
                        f"confirmed by intent ({intent}, {intent_confidence:.2f})")

        # Decision 3: RAG score >= 0.15 AND strong intent — use RAG content as the answer
        # This ensures we give real knowledge answers, not just generic templates
        elif embed_score >= 0.15 and embed_response and intent_confidence >= 0.6:
            response = embed_response
            method = "embedding"
            final_confidence = (embed_score + intent_confidence) / 2
            reasoning = (f"Weak RAG ({embed_score:.2f}) boosted by strong intent "
                        f"({intent}, {intent_confidence:.2f})")

        # Decision 4: Strong intent with no RAG match — use intent template
        elif intent_confidence >= 0.7 and intent in self.intent_templates:
            response = self._get_intent_response(intent)
            method = "intent"
            final_confidence = intent_confidence
            reasoning = f"Intent classification ({intent}, {intent_confidence:.2f})"

        # Decision 5: Medium intent (0.5-0.7) — still use intent if we have templates
        elif intent_confidence >= 0.5 and intent in self.intent_templates:
            response = self._get_intent_response(intent)
            method = "intent"
            final_confidence = intent_confidence
            reasoning = f"Moderate intent match ({intent}, {intent_confidence:.2f})"

        # Decision 5: Arabic keyword fallback (before generic fallback)
        else:
            arabic_fallback_resp = self._get_arabic_keyword_response(message)
            if arabic_fallback_resp:
                response = arabic_fallback_resp
                method = "arabic_fallback"
                final_confidence = 0.4
                reasoning = "Arabic keyword match in fallback"

            # Decision 6: Generic Fallback
            else:
                response = fallback_response
                method = "fallback"
                final_confidence = 0.2
                reasoning = "Low confidence across all systems, using fallback"

        # --- Personalization ---
        if user_name and method != "fallback" and response and random.random() < 0.25:
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
                "اهلا! ازاي اقدر اساعدك النهارده؟",
                "مرحبا! محتاج مساعدة في ايه؟",
            ],
            "goodbye": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Come back anytime!",
                "مع السلامة! يوم سعيد!",
                "باي! ارجع في اي وقت!",
            ],
            "thanks": [
                "You're welcome!",
                "Happy to help!",
                "Anytime! Let me know if you need more.",
                "العفو! اي وقت تحتاج مساعدة انا موجود.",
                "متشكر انك سألت! عايز حاجة تاني؟",
            ],
            "question_ai": [
                "That's a great AI question! Let me explain...",
                "AI is fascinating! Here's what I know about that:",
                "سؤال ممتاز عن الذكاء الاصطناعي! خليني اشرحلك...",
                "الذكاء الاصطناعي موضوع مهم جدا! اليك الاجابة:",
            ],
            "question_general": [
                "Interesting question! Let me think about that...",
                "Good question! Here's what I can share:",
                "سؤال كويس! خليني افكر واقولك...",
                "سؤال جميل! اليك اللي اعرفه:",
            ],
            "question_web": [
                "Great web development question! Here's what I know:",
                "Web development is exciting! Let me help with that:",
                "سؤال كويس عن تطوير الويب! خليني اشرحلك...",
                "تطوير الويب مجال ممتع! اليك الاجابة:",
            ],
            "question_programming": [
                "That's a solid programming question! Here's my take:",
                "Programming is a great skill to develop! Let me explain:",
                "سؤال برمجة ممتاز! خليني اقولك...",
                "البرمجة مجال رائع! اليك الاجابة:",
            ],
            "learning_request": [
                "I'd love to help you learn! Here are some resources and tips:",
                "Learning is a journey - let me guide you on where to start:",
                "Great initiative! Here's how you can get started:",
                "عظيم انك عايز تتعلم! خليني ادلك على الطريق...",
                "ممتاز! التعلم رحلة حلوة. اليك خطوات البداية:",
                "اكيد اساعدك تتعلم! خليني اقولك تبدأ منين...",
            ],
            "task_request": [
                "I'd love to help! What do you need?",
                "Sure thing! Tell me more about what you need.",
                "Absolutely! What are the details?",
                "اكيد اساعدك! محتاج ايه بالظبط؟",
                "طبعا! قولي تفاصيل اكتر.",
            ],
            "mood_positive": [
                "That's wonderful to hear!",
                "Great to know you're doing well!",
                "Awesome! Keep up the good vibes!",
                "حلو اوي! الحمد لله!",
                "جميل! ربنا يديم عليك!",
            ],
            "mood_negative": [
                "I'm sorry to hear that. Is there anything I can do to help?",
                "I hope things get better soon. I'm here for you.",
                "That sounds tough. Remember, one step at a time.",
                "متقلقش، ان شاء الله هتبقى احسن. انا هنا لو محتاج حاجة.",
                "ربنا يفرجها. خطوة خطوة وهتعدي.",
            ],
            "about_bot": [
                "I'm an AI Assistant! I use a hybrid system combining RAG retrieval, "
                "intent classification, and smart fallbacks to help you.",
                "انا مساعد ذكي! بستخدم نظام هجين بيجمع بين البحث الدلالي "
                "وتصنيف النوايا والردود الذكية عشان اساعدك.",
            ],
            "name_introduction": [
                "Nice to meet you! How can I help?",
                "اهلا بيك! تشرفنا! ازاي اقدر اساعدك؟",
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
            "مش متأكد اني فهمت. ممكن توضح اكتر؟",
            "سؤال مثير! ممكن تديني تفاصيل اكتر؟",
            "عايز اساعدك بس محتاج افهم اكتر. ممكن تشرح تاني؟",
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

        # Arabic keyword responses — used when all other systems fail
        self.arabic_keyword_responses = {
            # Programming languages
            "بايثون": "Python لغة ممتازة! بتستخدم في الذكاء الاصطناعي والويب والداتا. عايز تتعلمها؟",
            "جافاسكريبت": "JavaScript هي لغة الويب! بتشتغل في المتصفح وعلى السيرفر بـ Node.js.",
            "جافا": "Java لغة قوية جدا! بتستخدم في تطبيقات الاندرويد والانظمة الكبيرة.",
            # Topics
            "برمجة": "البرمجة مهارة قيمة جدا! اي جانب يهمك؟ لغة معينة؟ مجال معين؟",
            "كود": "الكود هو التعليمات اللي بنكتبها للكمبيوتر. عايز تتعلم تكتب كود؟",
            "ويب": "تطوير الويب مجال واسع! فرونت اند ولا باك اند ولا الاتنين؟",
            "موبايل": "تطبيقات الموبايل ممكن تعملها بـ Flutter او React Native او Kotlin. عايز تفاصيل؟",
            "داتا": "علم البيانات مجال مطلوب جدا! بيجمع بين البرمجة والاحصاء والتحليل.",
            "قاعدة بيانات": "قواعد البيانات بتنظم وتخزن البيانات. SQL ولا NoSQL — الاتنين مهمين!",
            # AI terms
            "ذكاء اصطناعي": "الذكاء الاصطناعي بيخلي الكمبيوتر يفكر ويتعلم. عايز اشرحلك اكتر؟",
            "شبكات عصبية": "الشبكات العصبية مستوحاة من المخ البشري وهي اساس التعلم العميق!",
            "تعلم الة": "تعلم الالة هو ان الكمبيوتر يتعلم من البيانات بدون برمجة مباشرة.",
            # General
            "نكتة": "ليه المبرمج بيحب الدارك مود؟ عشان النور بيجذب البق (الأخطاء)! 😄",
            "اخبار": "للأسف مش بقدر اجيبلك اخبار حية، بس ممكن اساعدك في اي سؤال تقني!",
            "شغل": "لو عايز تلاقي شغل في البرمجة، ابني بورتفوليو قوي واتعلم اللي السوق محتاجه!",
            "فريلانس": "الفريلانس في البرمجة ممكن على Upwork او Mostaql. ابدأ بأسعار معقولة وابني سمعة.",
        }

    # =========================================================================
    # ARABIC SEMANTIC BRIDGE
    # =========================================================================

    def _build_arabic_synonym_bridge(self):
        """
        Build Arabic-to-English semantic bridge mapping.

        This bridges the gap between Arabic queries and English knowledge entries.
        When a user asks "ايه هو AI؟", the bridge translates key terms to English
        equivalents so the RAG search can find matches in both Arabic and English KB.

        The bridge doesn't do full translation — it maps KEY TERMS that the TF-IDF
        vectorizer would use for matching. This dramatically improves retrieval
        for mixed-language queries.
        """
        # Arabic term → English equivalent(s) for RAG search boosting
        self.arabic_english_bridge = {
            # AI & ML
            "ذكاء اصطناعي": "artificial intelligence AI",
            "تعلم الة": "machine learning ML",
            "تعلم الالة": "machine learning ML",
            "تعلم عميق": "deep learning neural networks",
            "شبكات عصبية": "neural networks deep learning",
            "معالجة لغات": "natural language processing NLP",
            # Programming
            "برمجة": "programming coding software development",
            "لغة برمجة": "programming language best language",
            "بايثون": "Python programming language",
            "جافاسكريبت": "JavaScript web development",
            "خوارزمية": "algorithm data structures",
            "دالة": "function method programming",
            "متغير": "variable programming",
            "كائنية": "object oriented programming OOP",
            # Web
            "موقع": "website web development HTML CSS",
            "ويب": "web development frontend backend",
            "فرونت اند": "frontend HTML CSS JavaScript React",
            "باك اند": "backend server API database",
            "سيرفر": "server backend deployment",
            # Data
            "قاعدة بيانات": "database SQL NoSQL",
            "داتا": "data science analysis pandas",
            "بيانات": "data database information",
            # General tech
            "تطبيق": "application app development",
            "اتعلم": "learn study resources tutorial",
            "كورس": "course tutorial learning resources",
            "مشروع": "project build create",
            "شغل": "job career work programming",
            "مقابلة": "interview coding job",
            # Question patterns
            "ايه هو": "what is explain define",
            "ازاي": "how to create build make",
            "ليه": "why reason explanation",
            "فين": "where find resources",
            "افضل": "best top recommended",
            "الفرق": "difference between compare",
        }

    def _arabic_to_english_bridge(self, message):
        """
        Translate Arabic key terms in a message to English for better RAG retrieval.

        This doesn't do full translation — it appends English equivalents of
        detected Arabic terms to create a hybrid query that matches both Arabic
        and English entries in the knowledge base.

        Example:
            "ايه هو الذكاء الاصطناعي" →
            "ايه هو الذكاء الاصطناعي artificial intelligence AI what is explain define"

        Args:
            message (str): The user's message (potentially Arabic).

        Returns:
            str or None: Enhanced query with English terms appended,
                         or None if no Arabic terms were detected.
        """
        message_lower = message.lower() if message else ""

        # Check if message contains Arabic characters
        has_arabic = any('؀' <= c <= 'ۿ' for c in message_lower)
        if not has_arabic:
            return None

        # Find matching Arabic terms and collect English equivalents
        english_terms = []
        for arabic_term, english_equiv in self.arabic_english_bridge.items():
            if arabic_term in message_lower:
                english_terms.append(english_equiv)

        if not english_terms:
            return None

        # Create a bridged query: original Arabic + English equivalents
        # This lets TF-IDF match against both Arabic and English KB entries
        bridged = message + " " + " ".join(english_terms)
        return bridged

    def _get_arabic_keyword_response(self, message):
        """
        Check if the message matches any Arabic keyword and return a response.

        This is the Arabic equivalent of the English keyword fallback.
        Used as a last resort before the generic fallback responses.

        Args:
            message (str): The user's message.

        Returns:
            str or None: A response if an Arabic keyword matched, None otherwise.
        """
        message_lower = message.lower() if message else ""

        for keyword, response in self.arabic_keyword_responses.items():
            if keyword in message_lower:
                return response

        return None

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
        if not response:
            return response

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
