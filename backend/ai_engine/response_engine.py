"""
Response Engine Module - Hybrid Mode
======================================

This module provides the ResponseEngine class which delegates to the
HybridEngine for intelligent response generation.

The HybridEngine combines THREE systems:
  1. Embedding/TF-IDF Retrieval (semantic meaning)
  2. Intent Classification (user intent category)
  3. Fallback Generation (keyword matching + clarification)

This file acts as the public interface, maintaining backward compatibility
with all existing code that imports ResponseEngine.
"""

from .hybrid_engine import HybridEngine


class ResponseEngine:
    """
    Public interface for response generation.

    Delegates to HybridEngine internally for the actual processing.
    Maintains backward compatibility with all existing callers.
    """

    def __init__(self, similarity_threshold=0.4):
        """
        Initialize the ResponseEngine with the Hybrid system.

        Args:
            similarity_threshold (float): Not used directly anymore (HybridEngine
                uses its own decision matrix), but kept for interface compatibility.
        """
        self.hybrid = HybridEngine()
        self.similarity_threshold = similarity_threshold

    def generate(self, message, intent, memory=None):
        """
        Generate a response using the hybrid system.

        Args:
            message (str): The user's message.
            intent (str): Detected intent from IntentDetector.
            memory (dict, optional): User memory/context.

        Returns:
            str: The generated response.
        """
        intent_result = {"intent": intent, "confidence": 0.5}
        result = self.hybrid.process(message, intent_result=intent_result, memory=memory)
        return result["response"]

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
        memory = None
        if context:
            memory = {"name": context.get("user_name")}
        intent_result = {"intent": intent, "confidence": confidence}
        result = self.hybrid.process(user_message, intent_result=intent_result, memory=memory)
        return result["response"]

    def get_top_matches(self, message, top_n=5):
        """
        Get the top N matches with reasoning (for debugging).

        Args:
            message (str): User message.
            top_n (int): Number of top matches.

        Returns:
            dict: Detailed explanation of the hybrid decision.
        """
        return self.hybrid.get_decision_explanation(message)
