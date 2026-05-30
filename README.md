# 🤖 AI Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Arabic](https://img.shields.io/badge/Arabic-Egyptian_Dialect-red?style=for-the-badge)

## 📖 Description

An intelligent AI assistant featuring a modern web chat interface, built entirely from scratch with **no external AI APIs**. This project demonstrates how to create a functional conversational AI using classical machine learning techniques.

The assistant leverages **NLP-based intent detection**, a **persistent memory system** that remembers users across sessions, **RAG-powered knowledge retrieval**, and **contextual response generation** — all running locally without any third-party AI service dependencies.

### 🌍 Bilingual Support
Fully supports **English** and **Egyptian Arabic (العامية المصرية)** with dialect normalization, Arabic knowledge base, and cross-language semantic bridging.

---

## ✨ Features

- 💬 **Natural language understanding** with ML-based intent detection (13 intents, 296 patterns)
- 🧠 **Advanced memory system** — STM/LTM architecture with user profile evolution
- 🌐 **Modern web chat interface** with dark theme and responsive design
- 🚀 **FastAPI backend** with REST API and streaming endpoints
- 🎯 **Hybrid Decision Engine** — RAG + Intent + Fallback (3-system intelligence)
- 📚 **RAG Knowledge Base** — 104 chunks, self-learning, persistent storage
- 🔄 **Streaming responses** — ChatGPT-style word-by-word typing effect (SSE)
- 🛠️ **Built-in tools** — Calculator, notes, reminders, datetime, dictionary
- 📊 **Analytics dashboard** — Real-time usage stats and visualizations
- 🇪🇬 **Egyptian Arabic support** — Normalization, knowledge base, intent training, and semantic bridge
- 🔗 **Context memory** — Tracks topics, resolves references, personalizes responses
- ☁️ **Deployment ready** — Render, Railway, Docker support

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌─────────────────────────────────────────┐
│              │     │               │     │            AI Engine                      │
│   Web UI     │────▶│   FastAPI     │────▶│                                         │
│  (HTML/JS)   │◀────│   Backend     │◀────│  Normalizer → RAG → Intent → Hybrid     │
│              │     │               │     │         ↕           ↕          ↕         │
└──────────────┘     └───────────────┘     │    Memory      Context     Tools        │
                                           └──────────────────┬──────────────────────┘
                                                              │
                                                        ┌─────▼─────┐
                                                        │  SQLite   │
                                                        │ Databases │
                                                        └───────────┘
```

### Hybrid Decision Flow

```
User Message
  │
  ├─ 1️⃣ Arabic Normalization (dialect → standard)
  │
  ├─ 2️⃣ Tool Dispatch (calculator, notes, reminders)
  │     └─ If tool matches → return tool result directly
  │
  ├─ 3️⃣ Context Engine (topic tracking, reference resolution)
  │
  ├─ 4️⃣ RAG Retrieval + Arabic Semantic Bridge
  │     ├─ Score ≥ 0.6 → USE RAG (strong match)
  │     └─ Score 0.35-0.6 + intent confirms → USE RAG (confirmed)
  │
  ├─ 5️⃣ Intent Classification (TF-IDF + Logistic Regression)
  │     └─ Confidence ≥ 0.5 → USE INTENT TEMPLATE
  │
  ├─ 6️⃣ Arabic Keyword Fallback (17 Arabic keyword responses)
  │
  └─ 7️⃣ Generic Fallback (bilingual clarification questions)
```

---

## 📁 Project Structure

```
AI-Assistant/
├── backend/
│   ├── main.py                      # FastAPI app (all endpoints)
│   ├── ai_engine/
│   │   ├── __init__.py              # Package exports
│   │   ├── arabic_normalizer.py     # Egyptian dialect normalization
│   │   ├── egyptian_knowledge.py    # Arabic knowledge base (51 entries)
│   │   ├── hybrid_engine.py         # Core decision engine
│   │   ├── rag_engine.py            # Self-learning knowledge retrieval
│   │   ├── intent_model.py          # Intent classifier (TF-IDF + LogReg)
│   │   ├── response_engine.py       # Response generation wrapper
│   │   ├── memory.py                # Advanced STM/LTM memory system
│   │   ├── context_engine.py        # Context & reference resolution
│   │   ├── analytics.py             # Usage statistics
│   │   ├── streamer.py              # SSE streaming
│   │   ├── embeddings.py            # Embedding utilities
│   │   └── tools/                   # Built-in tools
│   └── data/
│       ├── intents.json             # 13 intents, 296 patterns (EN + AR)
│       ├── knowledge.db             # RAG knowledge (auto-generated)
│       └── memory.db                # User memory (auto-generated)
├── frontend/
│   ├── index.html                   # Chat UI (dark theme, streaming)
│   └── dashboard.html               # Analytics dashboard
├── models/
│   └── intent_model.pkl             # Trained intent classifier
├── requirements.txt                 # Python dependencies
├── run.py                           # Application entry point
├── Procfile                         # Heroku/Railway deployment
└── render.yaml                      # Render.com deployment
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/mohamedshhahat1/AI-Assistant.git
cd AI-Assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

The application will be available at `http://localhost:8000`

---

## 📡 API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message, get JSON response |
| POST | `/chat/stream` | Send a message, get streaming SSE response |

### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/memory/{user_id}` | Get user's memory/history |
| DELETE | `/memory/{user_id}` | Clear user's memory |
| GET | `/memory/{user_id}/profile` | Get evolved user profile |
| GET | `/memory/{user_id}/ltm` | Get long-term memories |
| POST | `/memory/{user_id}/ltm` | Add a long-term memory |
| GET | `/memory/{user_id}/session` | Get current session STM |
| POST | `/memory/{user_id}/session/end` | End current session |

### Knowledge (RAG)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/knowledge/stats` | Knowledge base statistics |
| GET | `/knowledge/search?q=...` | Search the knowledge base |
| POST | `/knowledge/add` | Add new knowledge |
| GET | `/knowledge/recent` | Recent learnings log |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analytics/overview` | General stats |
| GET | `/analytics/intents` | Intent frequency |
| GET | `/analytics/users` | Most active users |
| GET | `/analytics/activity/hourly` | Activity by hour |
| GET | `/analytics/activity/daily` | Activity by day |
| GET | `/analytics/conversations` | Recent conversations |
| GET | `/analytics/user/{user_id}` | User-specific stats |
| GET | `/dashboard` | Visual analytics dashboard |

### Tools & Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tools` | List available tools |
| GET | `/notes/{user_id}` | Get user's notes |
| GET | `/reminders/{user_id}` | Get user's reminders |
| GET | `/health` | Server health check |

---

## 🧠 How It Works

### 1. Arabic/Egyptian Normalization Layer

Before any AI processing, Egyptian dialect text is standardized:

```python
"إيه الأخبار" → "ايه الاخبار"
"انا عايز مساعده" → "انا اريد مساعدة"
"ازاى حالك" → "ازاي حالك"
```

**Pipeline:** Diacritics removal → Tatweel removal → Dialect standardization → Character normalization → Whitespace cleanup

### 2. Hybrid Decision Engine (Three-System Intelligence)

```
User: "ايه هو AI؟"
  │
  ├──→ [1. RAG + Arabic Bridge]  "AI artificial intelligence" → score: 0.45
  ├──→ [2. Intent]              question_ai, confidence: 0.85
  └──→ [3. Keyword Fallback]    "ذكاء اصطناعي" → Arabic response
  │
  ▼ HYBRID DECISION
  │
  Result: Intent classification (strong confidence) → AI explanation response
```

| RAG Score | Intent Confidence | Action |
|:-:|:-:|:--|
| ≥ 0.6 | any | ✅ Use RAG (strong semantic match) |
| 0.35-0.6 | intent confirms | ✅ Use RAG (confirmed by intent) |
| < 0.35 | ≥ 0.7 | 🎯 Use intent template |
| < 0.35 | 0.5-0.7 | 🎯 Use intent template (moderate) |
| low | low + Arabic keyword | 🔤 Arabic keyword fallback |
| low | low | 🔄 Generic fallback |

### 3. RAG Engine (Self-Learning Knowledge)

- **104 knowledge chunks** (53 English + 51 Egyptian Arabic)
- Uses TF-IDF + cosine similarity for semantic search
- **Self-learning**: Absorbs factual information from user conversations
- **Quality gate**: Only stores high-quality, informational content
- **Arabic Semantic Bridge**: Maps Arabic terms to English equivalents for cross-language matching

### 4. Advanced Memory System

```
┌─────────────────────────────────────────────────┐
│                 Memory Architecture               │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │   STM        │    │      LTM             │   │
│  │ (50 msgs)    │    │ Facts, Preferences,  │   │
│  │ Current      │    │ Topics, Events       │   │
│  │ Session      │    │ Importance-ranked     │   │
│  └──────────────┘    └──────────────────────┘   │
│                                                   │
│  ┌──────────────────────────────────────────┐   │
│  │          User Profile Evolution           │   │
│  │ Personality tags • Interests • Style      │   │
│  │ Evolves at end of each session            │   │
│  └──────────────────────────────────────────┘   │
│                                                   │
└─────────────────────────────────────────────────┘
```

### 5. Context Engine

Solves conversational continuity:
- **Topic tracking**: Knows what you're currently discussing
- **Reference resolution**: "this", "that", "it" → resolved from history
- **Follow-up detection**: Recognizes continuation messages
- **Personalization**: Adjusts responses based on user profile

### 6. Intent Detection

**13 intents** trained on **296 patterns** (English + Egyptian Arabic):

| Intent | Example (Arabic) | Example (English) |
|--------|-------------------|-------------------|
| `greeting` | "ازيك" | "hello" |
| `goodbye` | "مع السلامة" | "bye" |
| `thanks` | "شكرا" | "thanks" |
| `question_ai` | "ايه هو الذكاء الاصطناعي" | "what is AI" |
| `question_general` | "ايه ده" | "what is this" |
| `question_web` | "ازاي اعمل موقع" | "how to make a website" |
| `question_programming` | "ايه افضل لغة برمجة" | "best programming language" |
| `learning_request` | "عايز اتعلم AI" | "I want to learn" |
| `task_request` | "ساعدني" | "help me" |
| `name_introduction` | "اسمي محمد" | "my name is" |
| `mood_positive` | "الحمد لله" | "I'm great" |
| `mood_negative` | "انا زعلان" | "I'm sad" |
| `about_bot` | "انت مين" | "who are you" |

---

## 🇪🇬 Egyptian Arabic Support

### Full Pipeline

```
"إيه هو الذكاء الاصطناعي؟"
  │
  ├─ Normalize: "ايه هو الذكاء الاصطناعي"
  ├─ RAG Search: matches Arabic KB entry (score: 0.23)
  ├─ Arabic Bridge: adds "artificial intelligence AI" → better match
  ├─ Intent: question_ai (85%)
  └─ Response: "الذكاء الاصطناعي موضوع مهم جدا! اليك الاجابة:"
```

### Dialect Normalization Examples

| Input (Egyptian) | Normalized | Meaning |
|---|---|---|
| عايز | اريد | want |
| ازاى | ازاي | how |
| إيه | ايه | what |
| مش عارف | لا اعرف | don't know |
| دلوقتي | الان | now |
| عشان | لان | because |
| ده / دي | هذا / هذه | this |
| فين | اين | where |

### Arabic Knowledge Base Topics

| Topic | Entries | Coverage |
|---|---|---|
| AI & Machine Learning | 12 | الذكاء الاصطناعي، تعلم الالة، Deep Learning |
| Programming | 14 | بايثون، OOP، Git، خوارزميات |
| Web Development | 7 | API، React، Frontend/Backend |
| Data & Databases | 4 | SQL، pandas، قواعد بيانات |
| Python | 4 | pip، virtual env، تنصيب |
| Career & Learning | 5 | كورسات، مقابلات، فريلانس |
| Common Tech | 5 | Docker، Linux، Cloud |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core programming language |
| **FastAPI** | Web framework and REST API |
| **scikit-learn** | ML (TF-IDF, Logistic Regression, Cosine Similarity) |
| **SQLite** | Memory, knowledge, and analytics storage |
| **Uvicorn** | ASGI server |
| **NumPy** | Numerical operations |
| **joblib** | Model serialization |
| **HTML/CSS/JS** | Frontend web chat interface |
| **SSE** | Server-Sent Events for streaming |

---

## 🔮 Future Improvements

- 🧬 **Sentence Transformers** — Replace TF-IDF with dense embeddings for better semantic understanding
- 🎙️ **Voice input** — Speech-to-text integration
- 🌍 **More languages** — Gulf Arabic, Levantine Arabic, French
- ☁️ **Docker deployment** — Containerized for any cloud
- 🔗 **Plugin system** — Third-party tool extensions
- 📱 **Mobile app** — React Native or Flutter frontend
- 🤝 **Multi-turn reasoning** — Chain-of-thought for complex questions
- 📈 **A/B testing** — Compare response quality across methods

---

## 📊 System Statistics

| Metric | Value |
|--------|-------|
| Knowledge chunks | 104 (53 EN + 51 AR) |
| Intent patterns | 296 across 13 intents |
| Arabic normalizations | 50+ dialect mappings |
| Arabic keyword responses | 17 |
| Arabic-English bridge terms | 35+ |
| Response methods | 6 (RAG, intent, Arabic fallback, keyword, bridge, generic) |
| Memory tables | 4 (users, sessions, STM, LTM) |
| Built-in tools | 5 (calculator, notes, reminders, datetime, dictionary) |
| API endpoints | 25+ |

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shahat</a>
</p>
