"""
Response Engine Module - Dual Mode: Sentence Embeddings + TF-IDF Fallback
==========================================================================

An advanced response generation system that uses semantic similarity to find
the best response from an expanded knowledge base (80+ entries).

DUAL MODE SYSTEM:
  - PRIMARY: Sentence-Transformers embeddings (all-MiniLM-L6-v2, 384-dim)
    Superior quality — captures semantic meaning, synonyms, paraphrases.
  - FALLBACK: TF-IDF + cosine similarity (sklearn)
    Lightweight — works without the heavy sentence-transformers dependency.

The system automatically detects which mode is available at startup and
prints the active mode. Both modes use the same interface.

How it works:
  1. Build a knowledge base of 80+ (question, response) pairs
  2. Pre-compute embeddings (or TF-IDF vectors) for all KB patterns at startup
  3. When a user sends a message, embed it and compute cosine similarity
  4. Return the response with the highest semantic match
  5. Fall back to intent-based templates if similarity is too low
  6. Personalize with user context (name, history)
"""

import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .embeddings import EmbeddingEngine


class ResponseEngine:
    """
    Dual-mode semantic similarity response engine with intent-aware fallback.

    Uses sentence-transformers embeddings (if available) or TF-IDF vectorization
    with cosine similarity to find the most relevant response from a knowledge base.
    Falls back to intent templates when similarity is too low.

    Attributes:
        knowledge_base (list): List of (pattern, response) tuples for semantic matching.
        embedding_engine (EmbeddingEngine): Sentence-transformers engine (may be unavailable).
        use_embeddings (bool): Whether embeddings mode is active.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer (fallback mode).
        kb_vectors: TF-IDF matrix or embedding matrix of all KB patterns.
        similarity_threshold (float): Minimum similarity to use KB response.
    """

    def __init__(self, similarity_threshold=0.4):
        """
        Initialize the ResponseEngine with knowledge base and matching model.

        Attempts to use sentence-transformers embeddings for superior semantic
        matching. Falls back to TF-IDF if sentence-transformers is not available.

        Args:
            similarity_threshold (float): Minimum similarity score (0-1)
                required to use a knowledge base response. Below this threshold,
                the engine falls back to intent-based templates.
                Default: 0.4 (embedding similarity is more meaningful than TF-IDF).
        """
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = False
        self.embedding_engine = None
        self.kb_embeddings = None

        # TF-IDF fallback attributes
        self.vectorizer = None
        self.kb_vectors = None

        # Build the expanded knowledge base (80+ entries)
        self._build_knowledge_base()

        # Try to initialize embedding engine
        self.embedding_engine = EmbeddingEngine()

        if self.embedding_engine.is_available():
            # PRIMARY MODE: Sentence Embeddings
            self.use_embeddings = True
            self._build_embedding_index()
            print(f"[ResponseEngine] Using: Sentence Embeddings (all-MiniLM-L6-v2)")
        else:
            # FALLBACK MODE: TF-IDF
            self.use_embeddings = False
            self._build_tfidf_model()
            print(f"[ResponseEngine] Using: TF-IDF (fallback mode)")

        print(f"[ResponseEngine] Knowledge base loaded: {len(self.knowledge_base)} entries")
        print(f"[ResponseEngine] Similarity threshold: {self.similarity_threshold}")

        # Intent-based fallback templates
        self._build_intent_templates()

        # Keyword responses for additional fallback
        self._build_keyword_responses()

    # =========================================================================
    # KNOWLEDGE BASE - 80+ (pattern, response) pairs for semantic matching
    # =========================================================================

    def _build_knowledge_base(self):
        """
        Build the expanded knowledge base of (pattern, response) pairs.

        Contains 80+ entries covering greetings, AI/ML, programming, career,
        science, daily life, and more. The matching model learns from patterns
        and matches new user messages to the most similar pattern.
        """
        self.knowledge_base = [
            # --- Greetings (9 entries) ---
            ("hello", "Hello! 👋 How can I help you today?"),
            ("hi there", "Hi there! What's on your mind?"),
            ("hey how are you", "Hey! I'm doing great, thanks for asking! How about you?"),
            ("good morning", "Good morning! ☀️ Hope you're having a wonderful start to your day!"),
            ("good afternoon", "Good afternoon! How's your day going so far?"),
            ("good evening", "Good evening! 🌙 How can I help you tonight?"),
            ("what's up", "Not much! Just here ready to help. What's up with you?"),
            ("howdy", "Howdy! 🤠 What brings you here today?"),
            ("greetings", "Greetings! I'm at your service. What can I do for you?"),

            # --- Goodbyes (6 entries) ---
            ("goodbye", "Goodbye! Have a wonderful day! 👋"),
            ("see you later", "See you later! Take care of yourself!"),
            ("bye bye", "Bye bye! Come back anytime you need help!"),
            ("good night", "Good night! 🌙 Sweet dreams and see you next time!"),
            ("take care", "You too! Take care and don't hesitate to come back!"),
            ("I'm leaving now", "Alright! It was great chatting with you. Until next time! 👋"),

            # --- Thanks (5 entries) ---
            ("thank you so much", "You're very welcome! 😊 Glad I could help!"),
            ("thanks a lot", "Anytime! That's what I'm here for!"),
            ("I appreciate your help", "I appreciate you reaching out! Let me know if you need anything else."),
            ("thanks for the info", "Happy to share! Knowledge is better when shared. 📚"),
            ("that was helpful", "I'm glad it was helpful! Don't hesitate to ask more."),

            # --- AI & Technology Questions (12 entries) ---
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
            ("what are transformers in AI",
             "Transformers are a neural network architecture that revolutionized NLP. Introduced "
             "in the 'Attention Is All You Need' paper (2017), they use self-attention mechanisms "
             "to process sequences in parallel. GPT, BERT, and T5 are all transformer-based models. "
             "They power modern chatbots, translation, and text generation."),
            ("what is GPT",
             "GPT (Generative Pre-trained Transformer) is a family of large language models by "
             "OpenAI. It's pre-trained on massive text data and fine-tuned for tasks like "
             "conversation, writing, coding, and reasoning. GPT uses the transformer architecture "
             "with billions of parameters."),

            # --- More AI/ML Detail (8 entries) ---
            ("what is a convolutional neural network",
             "A CNN (Convolutional Neural Network) is a type of neural network specialized for "
             "processing grid-like data such as images. It uses convolutional layers to detect "
             "features (edges, textures, shapes) and pooling layers to reduce dimensions. "
             "CNNs power image classification, object detection, and facial recognition."),
            ("what is a recurrent neural network",
             "An RNN (Recurrent Neural Network) is designed for sequential data like text or "
             "time series. It maintains a hidden state that captures information from previous "
             "steps. Variants like LSTM and GRU solve the vanishing gradient problem. "
             "RNNs are used in language modeling, speech recognition, and translation."),
            ("what is overfitting in machine learning",
             "Overfitting occurs when a model learns the training data too well — including "
             "noise and outliers — and performs poorly on new, unseen data. Solutions include: "
             "regularization (L1/L2), dropout, early stopping, data augmentation, and using "
             "more training data. It's the opposite of underfitting."),
            ("what is transfer learning",
             "Transfer learning is a technique where a model trained on one task is reused as "
             "the starting point for a different task. For example, a model trained on ImageNet "
             "can be fine-tuned for medical imaging. It saves time, data, and compute — "
             "especially powerful when labeled data is scarce."),
            ("what is reinforcement learning",
             "Reinforcement Learning (RL) is a type of ML where an agent learns by interacting "
             "with an environment. It receives rewards or penalties for actions and learns to "
             "maximize cumulative reward. Applications include game playing (AlphaGo), robotics, "
             "and autonomous driving."),
            ("what is supervised learning",
             "Supervised learning is a type of ML where the model is trained on labeled data — "
             "each input has a known correct output. The model learns to map inputs to outputs. "
             "Examples: classification (spam detection), regression (price prediction). "
             "Common algorithms: linear regression, decision trees, SVMs, neural networks."),
            ("what is unsupervised learning",
             "Unsupervised learning is a type of ML where the model finds patterns in unlabeled "
             "data without explicit guidance. The model discovers structure on its own. "
             "Examples: clustering (customer segmentation), dimensionality reduction (PCA), "
             "anomaly detection. Common algorithms: K-means, DBSCAN, autoencoders."),
            ("what is gradient descent",
             "Gradient descent is an optimization algorithm used to minimize a model's loss "
             "function. It iteratively adjusts parameters in the direction of steepest descent "
             "(negative gradient). Variants include SGD (stochastic), mini-batch, Adam, and "
             "RMSprop. Learning rate controls step size."),

            # --- Programming & CS Questions (14 entries) ---
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
            ("what is JavaScript",
             "JavaScript is the language of the web! It runs in browsers to make pages interactive "
             "(DOM manipulation, events, animations) and on servers via Node.js. It's dynamically "
             "typed, supports both OOP and functional styles, and has a massive ecosystem (npm). "
             "Essential for front-end development."),
            ("what is React",
             "React is a JavaScript library for building user interfaces, created by Meta. "
             "It uses a component-based architecture where UI is built from reusable pieces. "
             "Key concepts: JSX, virtual DOM, state management, hooks, and unidirectional data flow. "
             "It's the most popular front-end framework."),
            ("what is Django",
             "Django is a high-level Python web framework that follows the 'batteries included' "
             "philosophy. It provides: ORM for database access, admin panel, authentication, "
             "URL routing, template engine, and security features out of the box. "
             "Great for building web apps quickly with best practices."),
            ("what is SQL",
             "SQL (Structured Query Language) is the standard language for managing relational "
             "databases. Key operations: SELECT (read), INSERT (create), UPDATE (modify), "
             "DELETE (remove). It also handles table creation, joins, indexing, and transactions. "
             "Used with MySQL, PostgreSQL, SQLite, and more."),
            ("what is git and how to use it",
             "Git is a distributed version control system. Essential commands: git init (create repo), "
             "git add (stage changes), git commit (save snapshot), git push (upload to remote), "
             "git pull (download changes), git branch (create branches), git merge (combine branches). "
             "Use GitHub/GitLab for collaboration."),
            ("what is object oriented programming",
             "Object-Oriented Programming (OOP) is a paradigm where code is organized around "
             "objects — instances of classes that bundle data (attributes) and behavior (methods). "
             "Four pillars: Encapsulation, Inheritance, Polymorphism, Abstraction. "
             "Used in Python, Java, C++, and most modern languages."),
            ("what is a REST API",
             "A REST API follows representational state transfer principles: stateless communication, "
             "resource-based URLs, standard HTTP methods (GET, POST, PUT, DELETE), and typically "
             "JSON responses. It's the most common way to build web services that front-end apps "
             "and mobile apps consume."),
            ("what is Docker",
             "Docker is a platform for containerizing applications. Containers package your app "
             "with all its dependencies into a portable, isolated unit that runs consistently "
             "anywhere. Key concepts: Dockerfile (build instructions), images (templates), "
             "containers (running instances), Docker Compose (multi-container apps)."),
            ("what is cloud computing",
             "Cloud computing delivers computing resources (servers, storage, databases, networking) "
             "over the internet on-demand. Major providers: AWS, Google Cloud, Azure. "
             "Models: IaaS (infrastructure), PaaS (platform), SaaS (software). "
             "Benefits: scalability, cost efficiency, global reach."),
            ("what are data structures",
             "Data structures are ways of organizing and storing data for efficient access. "
             "Common ones: Arrays (indexed), Linked Lists (sequential), Stacks (LIFO), "
             "Queues (FIFO), Hash Tables (key-value), Trees (hierarchical), Graphs (networked). "
             "Choosing the right structure depends on your use case."),

            # --- Career Advice (6 entries) ---
            ("how to prepare for coding interviews",
             "To prepare for coding interviews: 1) Master data structures & algorithms, "
             "2) Practice on LeetCode/HackerRank (aim for 100+ problems), 3) Learn patterns "
             "(sliding window, two pointers, BFS/DFS), 4) Practice explaining your thought process, "
             "5) Do mock interviews, 6) Study system design for senior roles."),
            ("how to write a good resume",
             "Resume tips for tech roles: 1) Keep it to 1 page (2 max for senior), "
             "2) Lead with impact metrics ('Reduced latency by 40%'), 3) List relevant skills "
             "and tech stack, 4) Include projects with links (GitHub), 5) Tailor to each job, "
             "6) Use action verbs (built, designed, optimized, led)."),
            ("what programming language should I learn first",
             "For beginners, I recommend Python — it's readable, versatile, and has amazing "
             "community support. After Python: learn JavaScript for web development, or "
             "dive deeper into Python for AI/data science. The best language is the one that "
             "matches your goals: web (JS), mobile (Swift/Kotlin), systems (Rust/C++)."),
            ("how to build a portfolio as a developer",
             "Build a strong developer portfolio: 1) Create 3-5 quality projects (not 20 tutorials), "
             "2) Include diverse projects (full-stack app, API, CLI tool), 3) Write clean README "
             "files with screenshots and live demos, 4) Contribute to open source, "
             "5) Write blog posts explaining what you built, 6) Deploy projects (not just GitHub)."),
            ("tips for learning to code faster",
             "Speed up your coding journey: 1) Code every day (even 30 minutes), 2) Build projects "
             "instead of only watching tutorials, 3) Read others' code on GitHub, 4) Teach what "
             "you learn (blog, YouTube), 5) Join communities (Discord, Reddit), 6) Don't compare "
             "yourself to others — focus on your own progress."),
            ("how to transition into tech career",
             "Transitioning into tech: 1) Pick a clear path (web dev, data science, DevOps), "
             "2) Learn fundamentals through bootcamps or self-study, 3) Build a portfolio of "
             "projects, 4) Network on LinkedIn and at meetups, 5) Apply broadly (aim for 50+ apps), "
             "6) Leverage transferable skills from your current career."),

            # --- Science Topics (6 entries) ---
            ("how does gravity work",
             "Gravity is the fundamental force of attraction between objects with mass. "
             "Newton described it as F = G(m1*m2)/r² — force proportional to masses and "
             "inversely proportional to distance squared. Einstein's general relativity "
             "explains it as massive objects curving spacetime, and other objects follow that curvature."),
            ("what is DNA",
             "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions "
             "for all living organisms. It's a double helix made of nucleotide base pairs "
             "(A-T, G-C). Segments of DNA called genes encode proteins that determine traits. "
             "The human genome has about 3 billion base pairs and ~20,000 genes."),
            ("how does the internet work",
             "The internet is a global network of interconnected computers. When you visit a "
             "website: 1) Your browser sends a DNS request to find the server's IP address, "
             "2) Establishes a TCP connection, 3) Sends an HTTP request, 4) The server responds "
             "with HTML/CSS/JS, 5) Your browser renders the page. Data travels as packets via routers."),
            ("what is quantum computing",
             "Quantum computing uses quantum mechanical phenomena (superposition, entanglement) "
             "to process information. Unlike classical bits (0 or 1), quantum bits (qubits) can "
             "be in superposition of both states simultaneously. This enables exponential speedup "
             "for specific problems like cryptography, optimization, and molecular simulation."),
            ("what is photosynthesis",
             "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
             "The equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. It occurs in chloroplasts "
             "using chlorophyll (which absorbs light). It's essential for life — producing oxygen "
             "and forming the base of most food chains."),
            ("how do vaccines work",
             "Vaccines work by training your immune system to recognize and fight pathogens. "
             "They introduce a harmless piece of the pathogen (weakened, inactivated, or mRNA "
             "instructions for a protein). Your immune system creates antibodies and memory cells, "
             "so if you encounter the real pathogen later, you can fight it quickly."),

            # --- Daily Life & Productivity (8 entries) ---
            ("how to be more productive",
             "Boost productivity with these strategies: 1) Use time blocking (dedicate hours to "
             "specific tasks), 2) Try the Pomodoro technique (25 min work, 5 min break), "
             "3) Eliminate distractions (phone away, notifications off), 4) Prioritize with "
             "Eisenhower matrix (urgent vs important), 5) Get enough sleep — it's not optional!"),
            ("tips for better sleep",
             "Improve your sleep quality: 1) Consistent schedule (same bedtime/wake time), "
             "2) No screens 1 hour before bed (blue light disrupts melatonin), 3) Cool, dark room, "
             "4) Avoid caffeine after 2pm, 5) Exercise regularly (but not right before bed), "
             "6) Wind-down routine (reading, stretching, meditation)."),
            ("how to stay motivated",
             "Staying motivated: 1) Set clear, specific goals (not vague wishes), 2) Break big "
             "goals into small milestones, 3) Track progress visually, 4) Find accountability "
             "partners, 5) Celebrate small wins, 6) Remember your 'why' — connect tasks to "
             "your bigger purpose. Motivation follows action — start small!"),
            ("how to manage stress",
             "Manage stress effectively: 1) Identify your stressors and what you can control, "
             "2) Practice deep breathing (4-7-8 technique), 3) Exercise regularly (even a walk helps), "
             "4) Talk to someone you trust, 5) Limit news/social media consumption, "
             "6) Prioritize self-care — you can't pour from an empty cup."),
            ("benefits of exercise",
             "Regular exercise benefits: 🏃 Physical: strengthens heart, builds muscle, boosts "
             "immunity, improves sleep. 🧠 Mental: reduces stress/anxiety, improves mood (endorphins), "
             "enhances focus and memory. Start with 30 min/day — walking, cycling, swimming, "
             "or bodyweight exercises. Consistency beats intensity!"),
            ("how to learn a new skill quickly",
             "Learn faster with these techniques: 1) Deconstruct the skill into sub-skills, "
             "2) Focus on the 20% that gives 80% of results (Pareto principle), 3) Practice "
             "deliberately (push your limits, get feedback), 4) Use spaced repetition, "
             "5) Teach others what you learn, 6) Immerse yourself — make it part of daily life."),
            ("how to read more books",
             "Read more books: 1) Set a daily page goal (even 10 pages = 15+ books/year), "
             "2) Always carry a book (or e-reader), 3) Read at fixed times (morning, before bed), "
             "4) Quit books you don't enjoy (life's too short), 5) Join a book club for "
             "accountability, 6) Alternate between easy and challenging reads."),
            ("tips for healthy eating",
             "Eat healthier: 1) Cook more at home (you control ingredients), 2) Eat more "
             "whole foods (fruits, vegetables, whole grains, lean protein), 3) Meal prep on "
             "weekends, 4) Stay hydrated (aim for 8 glasses/day), 5) Practice portion control, "
             "6) Don't restrict too much — allow treats in moderation."),

            # --- General Knowledge & Fun (8 entries) ---
            ("tell me a joke",
             "Why do programmers prefer dark mode? Because light attracts bugs! 😄🐛"),
            ("tell me another joke",
             "Why was the JavaScript developer sad? Because he didn't Node how to Express himself! 😂"),
            ("tell me a fun fact",
             "Fun fact: The first computer programmer was Ada Lovelace in the 1840s — "
             "she wrote algorithms for Charles Babbage's Analytical Engine, over 100 years "
             "before modern computers existed! 🖥️"),
            ("tell me something interesting",
             "Here's something cool: Honey never spoils! Archaeologists found 3000-year-old "
             "honey in Egyptian tombs that was still perfectly edible. Its low moisture and "
             "acidic pH make it inhospitable to bacteria. 🍯"),
            ("what is the meaning of life",
             "That's the big question! 🤔 Philosophically, many say it's about finding purpose, "
             "building connections, and continuous growth. From a practical standpoint — pursue "
             "what brings you fulfillment. Also, 42 (if you're a Hitchhiker's Guide fan 😄)."),
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
            ("recommend a podcast",
             "Great podcasts to check out: 🎧\n"
             "• Tech: 'Lex Fridman Podcast', 'Syntax' (web dev)\n"
             "• Science: 'Radiolab', 'StarTalk'\n"
             "• Business: 'How I Built This', 'The Tim Ferriss Show'\n"
             "• Storytelling: 'Serial', 'This American Life'"),

            # --- Bot Self-Awareness (4 entries) ---
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
             "🧠 Sentence-Transformers + scikit-learn for semantic matching, "
             "💾 SQLite for memory, and 🌐 HTML/CSS/JS for the chat interface. No external "
             "AI APIs — I'm 100% self-contained!"),
            ("are you a real person",
             "Nope! I'm an AI — a software program designed to understand and respond to "
             "natural language. But I try my best to be helpful, friendly, and sometimes "
             "even funny! 🤖😊"),

            # --- Mood & Emotion (5 entries) ---
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

            # --- Task/Help Requests (3 entries) ---
            ("can you help me with something",
             "Absolutely! I'd love to help. 🙌 Tell me what you need, and I'll do my best "
             "to assist you!"),
            ("I need assistance",
             "I'm here to help! 💪 What's on your mind? Whether it's a question, a task, "
             "or just someone to brainstorm with — fire away!"),
            ("how do I get started",
             "Great question! Getting started depends on what you're working on. Could you "
             "tell me a bit more about your goal? I can suggest a step-by-step approach."),

            # --- Miscellaneous (3 entries) ---
            ("what time is it",
             "I don't have access to a real-time clock, but you can check the time on your "
             "device! ⏰ Is there something time-related I can help with?"),
            ("what's the weather like",
             "I wish I could check the weather for you! ☁️ Unfortunately, I don't have access "
             "to real-time data. Try a weather app or website for the latest forecast."),
            ("what is blockchain",
             "Blockchain is a decentralized, distributed ledger technology that records "
             "transactions across many computers. Each block contains a cryptographic hash "
             "of the previous block, creating an immutable chain. It powers cryptocurrencies "
             "like Bitcoin and has applications in supply chain, voting, and smart contracts."),
        ]

        # Extract patterns and responses for quick access
        self.kb_patterns = [pattern for pattern, _ in self.knowledge_base]
        self.kb_responses = [response for _, response in self.knowledge_base]

    # =========================================================================
    # EMBEDDING INDEX - Pre-compute embeddings for all KB patterns
    # =========================================================================

    def _build_embedding_index(self):
        """
        Pre-compute sentence embeddings for all knowledge base patterns.

        This runs at startup and stores the embeddings matrix for fast
        similarity computation at inference time.
        """
        print(f"[ResponseEngine] Pre-computing embeddings for {len(self.kb_patterns)} patterns...")
        self.kb_embeddings = self.embedding_engine.encode(self.kb_patterns)
        print(f"[ResponseEngine] Embedding index built: shape {self.kb_embeddings.shape}")

    # =========================================================================
    # TF-IDF MODEL (FALLBACK) - Vectorize knowledge base for semantic matching
    # =========================================================================

    def _build_tfidf_model(self):
        """
        Build and fit the TF-IDF vectorizer on knowledge base patterns.

        This is the fallback mode used when sentence-transformers is not available.
        Creates a vector representation of each pattern in the knowledge base,
        enabling cosine similarity comparison with new user messages.
        """
        # Create and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # Unigrams and bigrams
            stop_words='english',     # Remove common words
            max_features=5000,        # Limit vocabulary size
            sublinear_tf=True,        # Apply log normalization to term frequency
        )

        # Fit and transform knowledge base patterns into TF-IDF vectors
        self.kb_vectors = self.vectorizer.fit_transform(self.kb_patterns)

        print(f"[ResponseEngine] TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

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

        Automatically uses sentence embeddings (if available) or TF-IDF for matching.

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
    # SEMANTIC MATCHING (DUAL MODE)
    # =========================================================================

    def _semantic_match(self, message):
        """
        Find the most semantically similar knowledge base entry to the user message.

        Uses sentence-transformers embeddings (if available) or TF-IDF vectorization
        with cosine similarity as fallback.

        Args:
            message (str): The user's message to match.

        Returns:
            tuple: (best_response, similarity_score)
                - best_response (str): The response paired with the best-matching pattern.
                - similarity_score (float): Similarity score (0-1) of the best match.
        """
        if self.use_embeddings:
            # PRIMARY MODE: Sentence Embeddings
            # Encode the user message
            query_embedding = self.embedding_engine.encode([message.lower()])

            # Compute cosine similarity vs all pre-computed KB embeddings
            similarities = self.embedding_engine.similarity(
                query_embedding[0], self.kb_embeddings
            )

            # Find the best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            return self.kb_responses[best_idx], float(best_score)

        else:
            # FALLBACK MODE: TF-IDF
            # Vectorize the user message using the same TF-IDF model
            message_vector = self.vectorizer.transform([message.lower()])

            # Compute cosine similarity between user message and all KB patterns
            similarities = sklearn_cosine_similarity(
                message_vector, self.kb_vectors
            ).flatten()

            # Find the best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            return self.kb_responses[best_idx], float(best_score)

    def get_top_matches(self, message, top_n=5):
        """
        Get the top N most similar knowledge base entries for a message.

        Works with both embedding and TF-IDF mode. Useful for debugging
        or showing alternative responses.

        Args:
            message (str): The user's message.
            top_n (int): Number of top matches to return.

        Returns:
            list: List of dicts with keys: pattern, response, similarity.
                Sorted by similarity score descending.
        """
        if self.use_embeddings:
            # Embedding mode
            query_embedding = self.embedding_engine.encode([message.lower()])
            similarities = self.embedding_engine.similarity(
                query_embedding[0], self.kb_embeddings
            )
        else:
            # TF-IDF mode
            message_vector = self.vectorizer.transform([message.lower()])
            similarities = sklearn_cosine_similarity(
                message_vector, self.kb_vectors
            ).flatten()

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
