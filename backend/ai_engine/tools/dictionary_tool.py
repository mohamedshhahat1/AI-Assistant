"""
Dictionary Tool - Define words and explain terminology.

Provides definitions for common and technical terms using a built-in
dictionary. Handles queries like "define X" or "what does X mean".
"""

import re
from typing import Dict, Optional

from .tool_registry import BaseTool


class DictionaryTool(BaseTool):
    """
    Dictionary tool for looking up word definitions.

    Contains a built-in dictionary of 30+ common and technical terms.
    Supports queries like:
        - "define algorithm" -> returns the definition
        - "what does API mean" -> returns the definition
        - "meaning of recursion" -> returns the definition
        - "define machine learning" -> returns the definition
    """

    name = "dictionary"
    description = "Look up definitions of words and technical terms"
    keywords = [
        "define", "definition", "meaning", "means",
        "what does", "what is a", "what is an", "what are",
        "explain", "describe"
    ]
    patterns = [
        r"define\s+\w+",
        r"what\s+(?:does|is|are)\s+(?:a\s+|an\s+)?[\w\s]+\s+mean",
        r"meaning\s+of\s+\w+",
        r"definition\s+of\s+\w+",
        r"what\s+is\s+(?:a|an)\s+\w+",
    ]

    # Built-in dictionary with 30+ tech and common terms
    DEFINITIONS: Dict[str, str] = {
        "algorithm": "A step-by-step procedure or set of rules for solving a problem or accomplishing a task, especially by a computer.",
        "api": "Application Programming Interface - a set of protocols and tools that allows different software applications to communicate with each other.",
        "array": "A data structure consisting of a collection of elements, each identified by an index or key, stored in contiguous memory locations.",
        "binary": "A number system that uses only two digits (0 and 1), which is the foundation of all computing and digital systems.",
        "boolean": "A data type that has only two possible values: true or false. Named after mathematician George Boole.",
        "bug": "An error, flaw, or fault in a computer program that causes it to produce incorrect or unexpected results.",
        "cache": "A hardware or software component that stores data so future requests for that data can be served faster.",
        "class": "A blueprint or template in object-oriented programming that defines the properties and behaviors of objects.",
        "cloud": "Cloud computing refers to delivering computing services (servers, storage, databases, networking) over the internet.",
        "compiler": "A program that translates source code written in a high-level programming language into machine code that a computer can execute.",
        "cpu": "Central Processing Unit - the primary component of a computer that performs most of the processing and executes instructions.",
        "database": "An organized collection of structured data stored electronically, designed for efficient access, management, and updating.",
        "debugging": "The process of finding and fixing errors (bugs) in computer software or hardware.",
        "encryption": "The process of converting data into a coded format to prevent unauthorized access, using algorithms and keys.",
        "framework": "A pre-built structure or platform that provides a foundation for developing software applications more efficiently.",
        "function": "A reusable block of code that performs a specific task, accepts inputs (parameters), and may return a value.",
        "git": "A distributed version control system that tracks changes in source code during software development.",
        "html": "HyperText Markup Language - the standard language for creating and structuring content on web pages.",
        "http": "HyperText Transfer Protocol - the foundation of data communication on the World Wide Web.",
        "ide": "Integrated Development Environment - a software application that provides comprehensive tools for software development in one place.",
        "inheritance": "A mechanism in object-oriented programming where a new class derives properties and behaviors from an existing class.",
        "json": "JavaScript Object Notation - a lightweight data interchange format that is easy for humans to read and machines to parse.",
        "kernel": "The core component of an operating system that manages system resources and communication between hardware and software.",
        "library": "A collection of pre-written code that developers can use to perform common tasks without writing code from scratch.",
        "loop": "A programming construct that repeats a block of code multiple times until a specified condition is met.",
        "machine learning": "A subset of artificial intelligence where systems learn and improve from experience without being explicitly programmed.",
        "neural network": "A computing system inspired by biological neural networks, consisting of interconnected nodes that process information in layers.",
        "object": "An instance of a class in object-oriented programming that contains data (attributes) and code (methods).",
        "python": "A high-level, interpreted programming language known for its simple syntax and versatility in web development, data science, and AI.",
        "recursion": "A programming technique where a function calls itself to solve a problem by breaking it into smaller sub-problems.",
        "repository": "A central location where data, code, or files are stored and managed, often used with version control systems like Git.",
        "rest": "Representational State Transfer - an architectural style for designing networked applications using standard HTTP methods.",
        "runtime": "The period during which a program is executing, or the environment that executes code (like the Java Runtime Environment).",
        "server": "A computer or program that provides services, data, or resources to other computers (clients) over a network.",
        "sql": "Structured Query Language - a standard language for managing and querying data in relational databases.",
        "stack": "A linear data structure that follows Last-In-First-Out (LIFO) principle, or the set of technologies used in a project.",
        "syntax": "The set of rules that defines how code must be written in a programming language to be valid and executable.",
        "terminal": "A text-based interface for interacting with a computer's operating system by typing commands.",
        "variable": "A named storage location in a program that holds a value which can be changed during execution.",
        "virtual machine": "A software emulation of a physical computer that runs an operating system and applications as if it were real hardware.",
    }

    def execute(self, message: str, user_id: Optional[str] = None, args: Optional[Dict] = None) -> Dict:
        """
        Look up a word definition based on the user's message.

        Extracts the target word from the message and searches the
        built-in dictionary for a matching definition.

        Args:
            message: The user's input message requesting a definition.
            user_id: Optional user identifier (not used).
            args: Optional additional arguments.

        Returns:
            Dictionary with the definition or an error message.
        """
        # Extract the word to define
        word = self._extract_word(message)

        if not word:
            return {
                "result": "Please specify a word to define (e.g., 'define algorithm').",
                "tool_used": "dictionary",
                "word": None
            }

        # Look up the word (case-insensitive)
        word_lower = word.lower().strip()
        definition = self.DEFINITIONS.get(word_lower)

        if definition:
            return {
                "result": f"{word.capitalize()}: {definition}",
                "tool_used": "dictionary",
                "word": word_lower,
                "definition": definition
            }
        else:
            # Try partial matching for multi-word terms
            for term, defn in self.DEFINITIONS.items():
                if word_lower in term or term in word_lower:
                    return {
                        "result": f"{term.upper()}: {defn}",
                        "tool_used": "dictionary",
                        "word": term,
                        "definition": defn
                    }

            return {
                "result": f"Sorry, I don't have a definition for \"{word}\". "
                          f"I can define terms like: algorithm, API, database, machine learning, and more.",
                "tool_used": "dictionary",
                "word": word_lower
            }

    def _extract_word(self, message: str) -> Optional[str]:
        """
        Extract the target word from the user's message.

        Handles various query formats:
            - "define algorithm"
            - "what does API mean"
            - "meaning of recursion"
            - "what is a database"

        Args:
            message: The user's input message.

        Returns:
            The extracted word/term, or None if not found.
        """
        message_clean = message.strip()

        # Pattern: "define <word>"
        match = re.search(r"define\s+(.+?)[\?\.\!]?$", message_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: "what does <word> mean"
        match = re.search(r"what\s+does\s+(.+?)\s+mean", message_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: "meaning/definition of <word>"
        match = re.search(r"(?:meaning|definition)\s+of\s+(.+?)[\?\.\!]?$", message_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: "what is a/an <word>"
        match = re.search(r"what\s+(?:is|are)\s+(?:a\s+|an\s+)?(.+?)[\?\.\!]?$", message_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: try to get the last meaningful word
        words = message_clean.split()
        if words:
            # Remove common query words
            skip_words = {"define", "what", "does", "mean", "meaning", "of", "is", "a", "an", "the", "explain"}
            meaningful = [w for w in words if w.lower() not in skip_words]
            if meaningful:
                return " ".join(meaningful)

        return None
