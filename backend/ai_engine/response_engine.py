"""
Response Engine Module
=====================
A rule-based response generation system that creates intelligent replies
based on detected intent, confidence score, and user context.

This module selects appropriate responses from predefined templates,
personalizes them with user information, and falls back to keyword
matching when intent detection confidence is low.
"""

import random
import re


class ResponseEngine:
    """
    Generates intelligent responses based on detected intent and user context.

    The engine uses a template-based approach with multiple response variations
    per intent category to keep conversations feeling natural and varied.
    """

    def __init__(self):
        """
        Initialize the ResponseEngine with response templates and fallback responses.

        Templates are organized by intent category, with multiple variations
        for each category to provide conversational variety.
        """

        # Response templates organized by intent category
        # Each category has multiple variations to avoid repetitive replies
        self.response_templates = {
            "greeting": [
                "Hello! 👋 How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to see you. How can I assist?",
                "Greetings! I'm here to help. What can I do for you?",
                "Hi! 😊 Ready to chat whenever you are!",
            ],
            "goodbye": [
                "Goodbye! Have a great day! 👋",
                "See you later! Take care!",
                "Bye! Come back anytime!",
                "It was nice chatting with you. Until next time!",
                "Take care! I'll be here whenever you need me. 👋",
            ],
            "thanks": [
                "You're welcome! 😊",
                "Happy to help!",
                "Anytime! Let me know if you need anything else.",
                "Glad I could assist! Don't hesitate to ask again.",
                "No problem at all! That's what I'm here for. 😊",
            ],
            "question_ai": [
                "AI (Artificial Intelligence) is the simulation of human intelligence "
                "by machines. It includes areas like machine learning, where computers "
                "learn from data, and deep learning, which uses neural networks inspired "
                "by the human brain.",
                "Artificial Intelligence encompasses many fields: Machine Learning (ML) "
                "allows systems to learn from experience, Deep Learning (DL) uses layered "
                "neural networks, and Natural Language Processing (NLP) helps machines "
                "understand human language.",
                "AI is a broad field of computer science focused on creating smart machines. "
                "Key branches include: ML (learning from data), Computer Vision (understanding "
                "images), NLP (processing language), and Robotics (physical AI agents).",
                "Great question! AI refers to systems that can perform tasks typically "
                "requiring human intelligence. This includes understanding language, "
                "recognizing patterns, making decisions, and even generating creative content.",
                "AI is transforming our world! From virtual assistants to self-driving cars, "
                "AI systems use algorithms and large datasets to make intelligent decisions "
                "without being explicitly programmed for every scenario.",
            ],
            "question_general": [
                "That's an interesting question! Let me think...",
                "I'd be happy to help with that! Here's what I know:",
                "Good question! Let me share what I can.",
                "Hmm, let me consider that for a moment...",
                "That's a great thing to wonder about! Here's my take:",
            ],
            "task_request": [
                "I'd love to help! What do you need?",
                "Sure thing! Tell me more about what you need.",
                "I'm on it! Could you give me a few more details?",
                "Absolutely! Let me know the specifics and I'll do my best.",
                "Of course! I'm ready to assist. What are the details?",
            ],
            "name_introduction": [
                "Nice to meet you, {name}! 😊 How can I help you?",
                "Hello, {name}! Great to know your name. What can I do for you?",
                "Hi {name}! 👋 Lovely to meet you. How can I assist today?",
                "Welcome, {name}! I'll remember that. What's on your mind?",
                "It's a pleasure, {name}! Feel free to ask me anything. 😊",
            ],
            "mood_positive": [
                "That's wonderful to hear! 🎉",
                "Great to know you're doing well!",
                "Awesome! That positive energy is contagious! 😄",
                "I'm glad to hear that! Keep up the good vibes! 🌟",
                "Fantastic! It makes me happy to hear you're doing great!",
            ],
            "mood_negative": [
                "I'm sorry to hear that. Is there anything I can do to help?",
                "I hope things get better soon. I'm here if you need to talk.",
                "That sounds tough. Remember, it's okay to take things one step at a time.",
                "I'm here for you. Would you like to talk about it or change the subject?",
                "I understand that can be difficult. Let me know how I can support you. 💙",
            ],
            "about_bot": [
                "I'm an AI Assistant built with Python! I can chat, answer questions, "
                "and remember our conversations.",
                "I'm a conversational AI designed to be helpful and friendly. "
                "I use natural language processing to understand what you need!",
                "I'm your AI assistant! I was built to have natural conversations, "
                "help answer questions, and make your day a little easier. 🤖",
                "I'm a Python-powered AI chatbot. I can understand your messages, "
                "detect your intent, and try to give helpful responses!",
                "Think of me as your digital companion! I use AI techniques to "
                "understand and respond to your messages in a helpful way.",
            ],
        }

        # Fallback responses used when no intent or keyword matches
        self.fallback_responses = [
            "I'm not quite sure I understand. Could you rephrase that?",
            "Interesting! Tell me more.",
            "I'm still learning! Could you explain what you mean?",
            "Hmm, I'm not sure how to respond to that. Can you try asking differently?",
            "That's beyond my current knowledge, but I'm always learning!",
        ]

        # Keyword-to-response mapping for fallback keyword matching
        self.keyword_responses = {
            "python": "Python is a fantastic programming language! It's known for its "
                      "readability and versatility. Are you learning Python or working on a project?",
            "programming": "Programming is such a valuable skill! Whether you're into web "
                           "development, data science, or AI — there's always something new to learn. "
                           "What aspect interests you?",
            "weather": "I wish I could check the weather for you! Unfortunately, I don't have "
                       "access to real-time weather data. Try checking a weather app or website.",
            "time": "I don't have access to a real-time clock, but you can check the time "
                    "on your device. Is there something time-related I can help with?",
            "joke": "Here's one: Why do programmers prefer dark mode? "
                    "Because light attracts bugs! 😄",
            "music": "Music is amazing! I can't play songs, but I'd love to chat about "
                     "your favorite genres or artists.",
            "food": "Food is a great topic! Whether you're looking for recipe ideas or "
                    "just chatting about favorites, I'm all ears!",
            "movie": "I love talking about movies! Do you have a favorite genre or "
                     "a recent film you enjoyed?",
            "book": "Books are wonderful! Are you looking for recommendations, or "
                    "would you like to discuss something you've read?",
            "help": "I'm here to help! You can ask me questions, have a conversation, "
                    "or just chat. What would you like to do?",
        }

    def generate_response(self, intent, confidence, user_message, context=None):
        """
        Generate an appropriate response based on intent, confidence, and context.

        Args:
            intent (str): The detected intent category (e.g., "greeting", "goodbye").
            confidence (float): Confidence score of the intent detection (0.0 to 1.0).
            user_message (str): The original message from the user.
            context (dict, optional): User context containing info like user name.
                                      Example: {"user_name": "Ahmed"}

        Returns:
            str: The generated response string.
        """

        # If confidence is too low, fall back to keyword matching
        if confidence < 0.3:
            return self._keyword_fallback(user_message)

        # Handle special case: name introduction
        if intent == "name_introduction":
            # Try to extract the user's name from the message
            name = self._extract_name(user_message)
            if name:
                # Pick a random name_introduction template and fill in the name
                template = random.choice(self.response_templates["name_introduction"])
                return template.format(name=name)
            else:
                # Could not extract name, use a generic greeting
                return "Nice to meet you! What should I call you?"

        # Handle special case: AI-related questions
        if intent == "question_ai":
            return random.choice(self.response_templates["question_ai"])

        # Handle special case: general questions with keyword matching
        if intent == "question_general":
            # Try keyword matching first for a more specific answer
            keyword_response = self._keyword_fallback(user_message)
            # If keyword fallback returned a generic fallback, use intent template instead
            if keyword_response in self.fallback_responses:
                return self._get_intent_response(intent, context)
            return keyword_response

        # For all other intents, use template-based response
        return self._get_intent_response(intent, context)

    def _get_intent_response(self, intent, context=None):
        """
        Select a random response from templates for the given intent.

        If the intent is not found in templates, returns a fallback response.
        Personalizes the response with the user's name if available in context.

        Args:
            intent (str): The intent category to get a response for.
            context (dict, optional): User context that may contain "user_name".

        Returns:
            str: A randomly selected and optionally personalized response.
        """

        # Check if we have templates for this intent
        if intent in self.response_templates:
            response = random.choice(self.response_templates[intent])
        else:
            # Intent not recognized, use fallback
            response = random.choice(self.fallback_responses)

        # Personalize response with user name if available
        if context and context.get("user_name"):
            user_name = context["user_name"]
            # Add personalization to the beginning of the response
            # Only if the response doesn't already contain a name placeholder
            if "{name}" not in response:
                response = f"{user_name}, " + response[0].lower() + response[1:]

        return response

    def _keyword_fallback(self, message):
        """
        Attempt to match keywords in the user message for a relevant response.

        This is used when intent detection confidence is low, providing a
        secondary method of generating useful responses.

        Args:
            message (str): The user's message to scan for keywords.

        Returns:
            str: A keyword-matched response or a generic fallback.
        """

        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()

        # Check each keyword against the message
        for keyword, response in self.keyword_responses.items():
            if keyword in message_lower:
                return response

        # No keywords matched, return a generic fallback response
        return random.choice(self.fallback_responses)

    def _extract_name(self, message):
        """
        Try to extract a name from messages like 'my name is Ahmed' or 'call me Sara'.

        Uses simple regex pattern matching to find names after common introduction
        phrases. The extracted name is capitalized for proper formatting.

        Args:
            message (str): The user's message that may contain a name introduction.

        Returns:
            str or None: The extracted name (capitalized) or None if not found.
        """

        # Define patterns that commonly precede a name introduction
        # Patterns: "my name is X", "I am X", "call me X", "I'm X"
        patterns = [
            r"(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)",
            r"(?:name's|they call me|people call me)\s+([a-zA-Z]+)",
        ]

        # Convert to lowercase for pattern matching but preserve original for extraction
        message_lower = message.lower()

        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Extract the name and capitalize it
                name = match.group(1).capitalize()
                return name

        # No name pattern found
        return None
