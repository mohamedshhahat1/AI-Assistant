"""
Response Streamer Module
========================
Handles streaming/chunking of AI responses for Server-Sent Events (SSE).
Provides word-by-word streaming with realistic typing delays to create
a natural ChatGPT/Claude-like typing effect.
"""

import asyncio
import json
import random
from typing import AsyncGenerator


class ResponseStreamer:
    """
    Streams AI response text word-by-word with realistic delays.

    Simulates a natural typing effect by yielding chunks of text
    with variable delays between them. Punctuation triggers slightly
    longer pauses to mimic human reading/typing patterns.
    """

    def __init__(self, min_delay: float = 0.03, max_delay: float = 0.08):
        """
        Initialize the ResponseStreamer.

        Args:
            min_delay: Minimum seconds between chunks (default 0.03s).
            max_delay: Maximum seconds between chunks (default 0.08s).
        """
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def stream_response(self, text: str) -> AsyncGenerator[str, None]:
        """
        Stream a complete response text word by word.

        Takes a complete response text and yields it in chunks (word by word)
        with realistic delays between chunks. Adds slightly longer pauses
        after punctuation marks (., !, ?, :) to feel natural.

        Args:
            text: The complete response text to stream.

        Yields:
            JSON strings representing chunks and a final done message.
            Chunk format: {"type": "chunk", "content": "word "}
            Done format: {"type": "done", "full_response": "complete text here"}
        """
        words = text.split(" ")

        for i, word in enumerate(words):
            # Add space after word (except for the last word)
            chunk = word + " " if i < len(words) - 1 else word

            # Yield the chunk as a JSON object
            yield json.dumps({"type": "chunk", "content": chunk})

            # Determine delay
            delay = random.uniform(self.min_delay, self.max_delay)

            # Add longer pause after punctuation for natural feel
            if word and word[-1] in ".!?:":
                delay += random.uniform(0.05, 0.15)

            await asyncio.sleep(delay)

        # Yield the final done message with the full response
        yield json.dumps({"type": "done", "full_response": text})

    async def stream_with_thinking(self, text: str) -> AsyncGenerator[str, None]:
        """
        Stream a response with a simulated 'thinking' phase first.

        First yields a thinking indicator, waits briefly to simulate
        AI processing, then streams the response word by word.

        Args:
            text: The complete response text to stream.

        Yields:
            JSON strings representing thinking state, start signal,
            chunks, and a final done message.
            Thinking format: {"type": "thinking", "content": ""}
            Start format: {"type": "start", "content": ""}
            Chunk format: {"type": "chunk", "content": "word "}
            Done format: {"type": "done", "full_response": "complete text here"}
        """
        # Signal UI to show thinking indicator
        yield json.dumps({"type": "thinking", "content": ""})

        # Simulate thinking delay (0.5 to 1 second)
        await asyncio.sleep(random.uniform(0.5, 1.0))

        # Signal start of response
        yield json.dumps({"type": "start", "content": ""})

        # Stream the response word by word
        words = text.split(" ")

        for i, word in enumerate(words):
            # Add space after word (except for the last word)
            chunk = word + " " if i < len(words) - 1 else word

            # Yield the chunk as a JSON object
            yield json.dumps({"type": "chunk", "content": chunk})

            # Determine delay
            delay = random.uniform(self.min_delay, self.max_delay)

            # Add longer pause after punctuation for natural feel
            if word and word[-1] in ".!?:":
                delay += random.uniform(0.05, 0.15)

            await asyncio.sleep(delay)

        # Yield the final done message with the full response
        yield json.dumps({"type": "done", "full_response": text})
