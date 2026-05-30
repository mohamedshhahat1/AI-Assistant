# 🤖 AI Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## 📖 Description

An intelligent AI assistant featuring a modern web chat interface, built entirely from scratch with **no external AI APIs**. This project demonstrates how to create a functional conversational AI using classical machine learning techniques.

The assistant leverages **NLP-based intent detection**, a **persistent memory system** that remembers users across sessions, and **contextual response generation** to deliver meaningful interactions — all running locally without any third-party AI service dependencies.

---

## ✨ Features

- 💬 **Natural language understanding** with ML-based intent detection
- 🧠 **Persistent memory system** — remembers users across sessions
- 🌐 **Modern web chat interface** with responsive design
- 🚀 **FastAPI backend** with REST API endpoints
- 🎯 **TF-IDF + Logistic Regression** for intent classification
- 💾 **SQLite database** for user memory storage
- 🔄 **Contextual responses** based on conversation history
- 📊 **10 supported intents** (greeting, goodbye, questions, tasks, and more)

---

## 🏗️ Architecture

```
┌──────────┐       ┌──────────┐       ┌─────────────────────────────────────┐       ┌────────────┐
│          │       │          │       │           AI Engine                  │       │            │
│  Web UI  │──────▶│  FastAPI │──────▶│  Intent + Response + Memory         │──────▶│ SQLite DB  │
│          │◀──────│          │◀──────│                                     │◀──────│            │
└──────────┘       └──────────┘       └─────────────────────────────────────┘       └────────────┘
```

---

## 📁 Project Structure

```
AI-Assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── ai_engine.py         # Core AI logic (intent detection + response)
│   ├── memory.py            # Persistent memory system
│   ├── intent_classifier.py # TF-IDF + Logistic Regression model
│   ├── intents.py           # Intent definitions and training data
│   └── responses.py         # Response templates
├── static/
│   ├── index.html           # Web chat interface
│   ├── style.css            # UI styling
│   └── script.js            # Frontend logic
├── database/
│   └── memory.db            # SQLite database (auto-generated)
├── requirements.txt         # Python dependencies
├── README.md
└── LICENSE
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
uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

---

## 📡 API Documentation

### POST `/chat`

Send a message to the AI assistant.

**Request:**
```json
{
  "user_id": "user123",
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "Hello! I'm doing great, thanks for asking. How can I help you today?",
  "intent": "greeting",
  "confidence": 0.92
}
```

---

### GET `/memory/{user_id}`

Retrieve stored memory for a specific user.

**Request:**
```
GET /memory/user123
```

**Response:**
```json
{
  "user_id": "user123",
  "interactions": 15,
  "first_seen": "2026-05-20T10:30:00",
  "last_seen": "2026-05-30T14:22:00",
  "context": {
    "name": "Mohamed",
    "preferences": ["tech", "programming"]
  }
}
```

---

### DELETE `/memory/{user_id}`

Clear all stored memory for a specific user.

**Request:**
```
DELETE /memory/user123
```

**Response:**
```json
{
  "message": "Memory cleared for user: user123",
  "status": "success"
}
```

---

## 🧠 How It Works

### 1. Intent Detection

The system uses a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer combined with a **Logistic Regression** classifier to determine user intent.

- Text is preprocessed (lowercased, tokenized, stop words removed)
- TF-IDF transforms text into numerical feature vectors
- Logistic Regression classifies the vector into one of 10 intents
- A confidence score determines response certainty

### 2. Memory System

The memory module provides persistent context across conversations:

- User interactions are stored in an SQLite database
- Previous context is injected into response generation
- The system remembers user names, preferences, and conversation topics
- Memory enables personalized and contextually relevant responses

### 3. Response Engine

Responses are generated through a rule-based template system:

- Each intent maps to a set of response templates
- Templates are dynamically filled with context from memory
- Conversation history influences response selection
- Fallback responses handle low-confidence classifications

---

## 📊 Supported Intents

| Intent | Example | Response Type |
|--------|---------|---------------|
| `greeting` | "Hello!", "Hi there" | Friendly welcome message |
| `goodbye` | "Bye", "See you later" | Farewell response |
| `thanks` | "Thank you", "Thanks a lot" | Acknowledgment |
| `identity` | "Who are you?", "What's your name?" | Self-introduction |
| `help` | "Can you help me?", "What can you do?" | Capability overview |
| `mood` | "How are you?", "How's it going?" | Status response |
| `weather` | "What's the weather like?" | Weather-related response |
| `time` | "What time is it?", "What day is it?" | Time/date information |
| `joke` | "Tell me a joke", "Make me laugh" | Humor response |
| `knowledge` | "What is Python?", "Explain AI" | Informational response |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core programming language |
| **FastAPI** | Web framework and REST API |
| **scikit-learn** | Machine learning (TF-IDF + Logistic Regression) |
| **SQLite** | Lightweight database for memory persistence |
| **Uvicorn** | ASGI server |
| **HTML/CSS/JS** | Frontend web chat interface |
| **Jinja2** | Template rendering |

---

## 🔮 Future Improvements

- 🎯 **Add more intents** — Expand to 20+ intents for broader coverage
- 🧬 **Integrate vector embeddings** — Use word2vec or sentence transformers for semantic understanding
- 🎙️ **Add voice input** — Speech-to-text integration for hands-free interaction
- ☁️ **Deploy to cloud** — Containerize with Docker and deploy to AWS/GCP
- 🔗 **Plugin system** — Allow third-party extensions for custom capabilities
- 📈 **Analytics dashboard** — Track usage patterns and model performance
- 🌍 **Multi-language support** — Detect and respond in multiple languages

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🚀 Upgrade: Analytics Dashboard

**New Feature:** Real-time analytics dashboard with charts and stats

### New Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /analytics/overview` | Total users, messages, active today/week |
| `GET /analytics/intents` | Most common intents (sorted by frequency) |
| `GET /analytics/users` | Top 10 active users |
| `GET /analytics/activity/hourly` | Messages by hour (0-23) |
| `GET /analytics/activity/daily` | Messages per day (last 30 days) |
| `GET /analytics/conversations` | Recent 20 conversations |
| `GET /dashboard` | Serves the analytics dashboard UI |

### Dashboard Features
- 📈 Stats cards: Total Users, Messages, Active Today, Avg/User
- 🎯 Doughnut chart: Intent distribution
- 📅 Line chart: Daily activity (30 days)
- ⏰ Bar chart: Hourly activity patterns
- 👥 Active users table
- 💬 Recent conversations feed
- Auto-refreshes every 30 seconds
- Chart.js powered, dark theme, responsive

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shahat</a>
</p>
