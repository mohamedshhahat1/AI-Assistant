"""
AI Assistant Backend - FastAPI Application
==========================================
This is the main backend server for the AI Assistant project.
It handles chat requests, memory management, analytics, RAG knowledge management,
and serves the frontend.
"""

import sys
import os
import sqlite3

# Add backend directory to Python path so imports work from both
# root directory (deployment) and backend directory (local dev)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

# Import our AI engine components
from ai_engine import IntentDetector, ResponseEngine, MemorySystem, Analytics, ToolDispatcher, RAGEngine, ArabicNormalizer
from ai_engine.context_engine import ContextEngine
from ai_engine.streamer import ResponseStreamer


# ====================
# Pydantic Models
# ====================
# These models define the shape of request/response data.
# FastAPI uses them for automatic validation and documentation.

class ChatRequest(BaseModel):
    """Model for incoming chat messages."""
    user_id: str  # Unique identifier for the user
    message: str  # The user's message text


class ChatResponse(BaseModel):
    """Model for chat response data."""
    response: str        # The AI-generated response text
    intent: str          # Detected intent (e.g., "greeting", "question")
    confidence: float    # How confident the model is in the detected intent (0-1)
    memory_used: bool    # Whether user memory was used to generate the response
    topic: Optional[str] = None      # Current conversation topic detected by ContextEngine
    is_follow_up: bool = False       # Whether this message was a follow-up to previous


class MemoryResponse(BaseModel):
    """Model for user memory data."""
    user_id: str                    # The user's ID
    name: Optional[str]             # The user's name (if known)
    history: list                   # List of past interactions
    preferences: dict               # User preferences learned over time


class DeleteMemoryResponse(BaseModel):
    """Model for memory deletion confirmation."""
    message: str  # Confirmation message


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str    # Server status (e.g., "healthy")
    version: str   # API version number


# ====================
# Analytics Pydantic Models
# ====================

class OverviewResponse(BaseModel):
    """Model for analytics overview stats."""
    total_users: int
    total_messages: int
    total_conversations: int
    avg_messages_per_user: float
    active_today: int
    active_this_week: int


class IntentStat(BaseModel):
    """Model for a single intent statistic."""
    intent: str
    count: int
    percentage: float


class ActiveUser(BaseModel):
    """Model for an active user entry."""
    user_id: str
    name: Optional[str]
    message_count: int
    last_active: Optional[str]


class HourlyActivity(BaseModel):
    """Model for hourly activity data point."""
    hour: int
    count: int


class DailyActivity(BaseModel):
    """Model for daily activity data point."""
    date: str
    count: int


class ConversationEntry(BaseModel):
    """Model for a recent conversation entry."""
    user_id: str
    name: Optional[str]
    message: str
    intent: Optional[str]
    timestamp: str


class UserIntentStat(BaseModel):
    """Model for a user's intent stat."""
    intent: str
    count: int


class UserStatsResponse(BaseModel):
    """Model for detailed user statistics."""
    user_id: str
    name: Optional[str]
    total_messages: int
    first_seen: Optional[str]
    last_active: Optional[str]
    top_intents: List[UserIntentStat]
    preferences: dict


# ====================
# Knowledge/RAG Pydantic Models
# ====================

class KnowledgeAddRequest(BaseModel):
    """Model for adding new knowledge to the RAG system."""
    text: str                          # The text content to add
    source: Optional[str] = "admin"    # Source tag (admin, system, etc.)
    topic: Optional[str] = None        # Topic category (auto-detected if None)


class KnowledgeSearchResult(BaseModel):
    """Model for a single knowledge search result."""
    content: str
    score: float
    source: str
    topic: str


class KnowledgeStatsResponse(BaseModel):
    """Model for RAG knowledge base statistics."""
    total_chunks: int
    sources: dict
    topics: dict
    avg_quality: float


class LearningEntry(BaseModel):
    """Model for a learning log entry."""
    user_id: str
    original_message: str
    extracted_knowledge: str
    accepted: bool
    timestamp: str


# ====================
# Application Setup
# ====================

# Create the FastAPI application instance
app = FastAPI(
    title="AI Assistant API",
    description="A smart AI assistant with memory, intent detection, and RAG-powered knowledge",
    version="1.0.0"
)

# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows the frontend to make requests to the backend
# even if they're served from different origins/ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (use specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)

# ====================
# AI Engine Components
# ====================
# These will be initialized on application startup

intent_detector: Optional[IntentDetector] = None
response_engine: Optional[ResponseEngine] = None
memory_system: Optional[MemorySystem] = None
analytics: Optional[Analytics] = None
tool_dispatcher: Optional[ToolDispatcher] = None
response_streamer: Optional[ResponseStreamer] = None
rag_engine: Optional[RAGEngine] = None
context_engine: Optional[ContextEngine] = None
arabic_normalizer: Optional[ArabicNormalizer] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize AI engine components when the server starts.
    This runs once before the server begins accepting requests.
    """
    global intent_detector, response_engine, memory_system, analytics, tool_dispatcher, response_streamer, rag_engine, context_engine, arabic_normalizer

    print("Initializing AI engine components...")

    # Create instances of each AI component
    intent_detector = IntentDetector()
    response_engine = ResponseEngine()
    memory_system = MemorySystem()
    analytics = Analytics()
    tool_dispatcher = ToolDispatcher()
    response_streamer = ResponseStreamer()
    context_engine = ContextEngine()
    arabic_normalizer = ArabicNormalizer()

    # Initialize RAG engine (shared reference with the one inside HybridEngine)
    # Access the RAG engine from the response engine's hybrid engine
    rag_engine = response_engine.hybrid.rag_engine

    print("AI engine components initialized successfully!")
    print(f"RAG Engine: {rag_engine.get_stats()['total_chunks']} knowledge chunks loaded")
    print("[ContextEngine] Context memory integration active")
    print("[ArabicNormalizer] Egyptian dialect normalization active")


# ====================
# API Endpoints
# ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes a user message and returns an AI response.

    Uses the advanced memory system with session tracking and STM/LTM.
    Tool dispatch is attempted first - if a tool matches, its result is returned
    directly without going through the intent detection and response engine.

    After generating a response, the RAG engine attempts to learn from the user's
    message if it contains factual/informational content.
    """
    try:
        # Step 0a: Normalize Arabic/Egyptian dialect before any AI processing
        # Original message is preserved for memory storage (what the user actually typed)
        normalized_message = arabic_normalizer.normalize(request.message)

        # Step 0b: Try tool dispatch first
        tool_result = tool_dispatcher.dispatch(normalized_message, request.user_id)
        if tool_result is not None:
            response_text = tool_result["result"]
            intent = "tool_use"
            confidence = 1.0

            # Still save interaction to memory (use original message for history)
            session_id = memory_system.get_active_session(request.user_id)
            if session_id is None:
                session_id = memory_system.start_session(request.user_id)

            context = memory_system.get_context(request.user_id, session_id)
            memory_used = context.get("is_returning_user", False)

            memory_system.save_to_stm(session_id, request.user_id, "user", request.message, intent=intent)
            memory_system.save_to_stm(session_id, request.user_id, "assistant", response_text, intent=intent)
            memory_system.save_interaction(
                user_id=request.user_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=intent
            )

            # Attempt to learn from user message even for tool interactions
            rag_engine.learn_from_message(normalized_message, intent, confidence)

            # Build context for topic/follow-up info even for tool results
            rich_context = context_engine.build_context(
                request.user_id, normalized_message, memory_system
            )

            return ChatResponse(
                response=response_text,
                intent=intent,
                confidence=confidence,
                memory_used=memory_used,
                topic=rich_context.get("current_topic"),
                is_follow_up=rich_context.get("is_follow_up", False)
            )

        # Step 1: Ensure active session exists
        session_id = memory_system.get_active_session(request.user_id)
        if session_id is None:
            session_id = memory_system.start_session(request.user_id)

        # Step 2: Build RICH context using ContextEngine (with normalized text)
        rich_context = context_engine.build_context(
            request.user_id, normalized_message, memory_system
        )

        # Step 3: Get memory context (STM + LTM + profile)
        context = memory_system.get_context(request.user_id, session_id)
        memory_used = context.get("is_returning_user", False)

        # Step 4: Detect the intent of the message (using normalized text)
        # Use resolved message for better intent detection on follow-ups
        detection_message = normalized_message
        if rich_context.get("is_follow_up") and rich_context.get("resolved_message") != normalized_message:
            detection_message = rich_context["resolved_message"]

        intent_result = intent_detector.detect(detection_message)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Step 5: Generate a response using hybrid engine with context integration
        result = response_engine.hybrid.process(
            message=normalized_message,
            intent_result=intent_result,
            memory=context,
            context_engine=context_engine,
            memory_system=memory_system,
            user_id=request.user_id
        )
        response_text = result["response"]

        # Step 6: Save to STM and backward-compatible history (original message for history)
        memory_system.save_to_stm(session_id, request.user_id, "user", request.message, intent=intent)
        memory_system.save_to_stm(session_id, request.user_id, "assistant", response_text, intent=intent)

        # Also save via backward-compatible method (populates old history table)
        memory_system.save_interaction(
            user_id=request.user_id,
            user_message=request.message,
            assistant_response=response_text,
            intent=intent
        )

        # Step 7: Return the response with context info
        return ChatResponse(
            response=response_text,
            intent=intent,
            confidence=confidence,
            memory_used=memory_used,
            topic=rich_context.get("current_topic"),
            is_follow_up=rich_context.get("is_follow_up", False)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - processes a user message and streams the AI response
    word-by-word using Server-Sent Events (SSE).

    Same processing logic as /chat but returns a streaming response instead of
    a single JSON payload. This creates a ChatGPT/Claude-like typing effect.

    The stream sends JSON events in SSE format:
      - {"type": "thinking", "content": ""} - AI is processing
      - {"type": "start", "content": ""} - Response is starting
      - {"type": "chunk", "content": "word "} - Each word/chunk of the response
      - {"type": "done", "full_response": "..."} - Complete response delivered
    """
    try:
        # Step 0a: Normalize Arabic/Egyptian dialect before any AI processing
        normalized_message = arabic_normalizer.normalize(request.message)

        # Step 0b: Try tool dispatch first
        tool_result = tool_dispatcher.dispatch(normalized_message, request.user_id)
        if tool_result is not None:
            response_text = tool_result["result"]
            intent = "tool_use"
            confidence = 1.0

            # Still save interaction to memory (original message for history)
            session_id = memory_system.get_active_session(request.user_id)
            if session_id is None:
                session_id = memory_system.start_session(request.user_id)

            context = memory_system.get_context(request.user_id, session_id)
            memory_used = context.get("is_returning_user", False)

            memory_system.save_to_stm(session_id, request.user_id, "user", request.message, intent=intent)
            memory_system.save_to_stm(session_id, request.user_id, "assistant", response_text, intent=intent)
            memory_system.save_interaction(
                user_id=request.user_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=intent
            )

            # Attempt to learn from user message (normalized)
            rag_engine.learn_from_message(normalized_message, intent, confidence)
        else:
            # Step 1: Ensure active session exists
            session_id = memory_system.get_active_session(request.user_id)
            if session_id is None:
                session_id = memory_system.start_session(request.user_id)

            # Step 2: Get rich context (STM + LTM + profile)
            context = memory_system.get_context(request.user_id, session_id)
            memory_used = context.get("is_returning_user", False)

            # Step 3: Detect the intent of the message (normalized)
            intent_result = intent_detector.detect(normalized_message)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]

            # Step 4: Generate a response using intent + rich context (normalized)
            response_text = response_engine.generate(
                message=normalized_message,
                intent=intent,
                memory=context
            )

            # Step 5: Save to STM and backward-compatible history (original for history)
            memory_system.save_to_stm(session_id, request.user_id, "user", request.message, intent=intent)
            memory_system.save_to_stm(session_id, request.user_id, "assistant", response_text, intent=intent)

            # Also save via backward-compatible method (populates old history table)
            memory_system.save_interaction(
                user_id=request.user_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=intent
            )

        # Step 6: Stream the response using SSE
        async def event_generator():
            async for chunk in response_streamer.stream_with_thinking(response_text):
                yield f"data: {chunk}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming chat message: {str(e)}"
        )


# ====================
# Knowledge/RAG Endpoints
# ====================

@app.get("/knowledge/stats", response_model=KnowledgeStatsResponse)
async def knowledge_stats():
    """
    Get RAG knowledge base statistics.
    Returns total chunks, source breakdown, topic breakdown, and average quality.
    """
    try:
        stats = rag_engine.get_stats()
        return KnowledgeStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting knowledge stats: {str(e)}"
        )


@app.get("/knowledge/search", response_model=List[KnowledgeSearchResult])
async def knowledge_search(q: str = Query(..., description="Search query")):
    """
    Search the knowledge base.
    Returns top 5 matches with scores, sources, and topics.
    """
    try:
        results = rag_engine.search(q, top_k=5, min_score=0.1)
        return [KnowledgeSearchResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching knowledge base: {str(e)}"
        )


@app.post("/knowledge/add")
async def knowledge_add(request: KnowledgeAddRequest):
    """
    Manually add new knowledge to the RAG system.
    Body: {"text": "...", "source": "admin", "topic": "web_dev"}
    The text will be chunked and indexed automatically.
    """
    try:
        rag_engine.learn_from_text(
            text=request.text,
            source=request.source,
            topic=request.topic
        )
        stats = rag_engine.get_stats()
        return {
            "message": "Knowledge added successfully",
            "total_chunks": stats["total_chunks"],
            "source": request.source,
            "topic": request.topic or "auto-detected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding knowledge: {str(e)}"
        )


@app.get("/knowledge/recent", response_model=List[LearningEntry])
async def knowledge_recent():
    """
    Get recent learnings - what the system has learned from user interactions.
    Returns the 10 most recent learning log entries.
    """
    try:
        learnings = rag_engine.get_recent_learnings(limit=10)
        return [LearningEntry(**entry) for entry in learnings]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting recent learnings: {str(e)}"
        )


# ====================
# Tool Endpoints
# ====================

@app.get("/tools")
async def get_tools():
    """
    Get a list of all available tools and their descriptions.
    Returns tool names and what they can do.
    """
    return tool_dispatcher.get_available_tools()


@app.get("/notes/{user_id}")
async def get_notes(user_id: str):
    """
    Get all notes for a specific user.
    Queries the notes table directly for the given user_id.
    """
    try:
        conn = sqlite3.connect(tool_dispatcher.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, title, content, created_at FROM notes WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        notes = [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "content": row["content"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
        return {"user_id": user_id, "notes": notes}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving notes for user {user_id}: {str(e)}"
        )


@app.get("/reminders/{user_id}")
async def get_reminders(user_id: str):
    """
    Get all reminders for a specific user.
    Queries the reminders table directly for the given user_id.
    """
    try:
        conn = sqlite3.connect(tool_dispatcher.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, content, remind_at, created_at, completed FROM reminders WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        reminders = [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "remind_at": row["remind_at"],
                "created_at": row["created_at"],
                "completed": bool(row["completed"])
            }
            for row in rows
        ]
        return {"user_id": user_id, "reminders": reminders}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving reminders for user {user_id}: {str(e)}"
        )


@app.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memory(user_id: str):
    """
    Retrieve a user's memory/history.
    This shows what the AI remembers about the user.
    """
    try:
        # Load the user's memory from the memory system
        user_memory = memory_system.load_memory(user_id)

        # If no memory exists, return empty defaults
        if user_memory is None:
            return MemoryResponse(
                user_id=user_id,
                name=None,
                history=[],
                preferences={}
            )

        # Return the user's memory data
        return MemoryResponse(
            user_id=user_id,
            name=user_memory.get("name"),
            history=user_memory.get("history", []),
            preferences=user_memory.get("preferences", {})
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving memory for user {user_id}: {str(e)}"
        )


@app.delete("/memory/{user_id}", response_model=DeleteMemoryResponse)
async def delete_memory(user_id: str):
    """
    Clear a user's memory/history.
    This makes the AI forget everything about the user.
    """
    try:
        # Clear the user's memory from the memory system
        memory_system.clear_memory(user_id)

        return DeleteMemoryResponse(
            message=f"Memory cleared for user {user_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing memory for user {user_id}: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Used to verify the server is running and responsive.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


# ====================
# Analytics Endpoints
# ====================

@app.get("/analytics/overview", response_model=OverviewResponse)
async def analytics_overview():
    """
    Get general overview statistics about the system.
    Returns total users, messages, conversations, and activity counts.
    """
    data = analytics.get_overview()
    return OverviewResponse(**data)


@app.get("/analytics/intents", response_model=List[IntentStat])
async def analytics_intents():
    """
    Get intent frequency statistics.
    Returns intents sorted by how often they occur.
    """
    return analytics.get_intent_stats()


@app.get("/analytics/users", response_model=List[ActiveUser])
async def analytics_users():
    """
    Get the most active users (top 10).
    Returns users sorted by message count.
    """
    return analytics.get_active_users(limit=10)


@app.get("/analytics/activity/hourly", response_model=List[HourlyActivity])
async def analytics_hourly():
    """
    Get message count by hour of day (0-23).
    Useful for seeing when users are most active.
    """
    return analytics.get_hourly_activity()


@app.get("/analytics/activity/daily", response_model=List[DailyActivity])
async def analytics_daily():
    """
    Get message count by date for the last 30 days.
    Shows daily usage trends.
    """
    return analytics.get_daily_activity(days=30)


@app.get("/analytics/conversations", response_model=List[ConversationEntry])
async def analytics_conversations():
    """
    Get the 20 most recent user messages across all users.
    Shows what users are currently asking about.
    """
    return analytics.get_recent_conversations(limit=20)


@app.get("/analytics/user/{user_id}", response_model=UserStatsResponse)
async def analytics_user(user_id: str):
    """
    Get detailed statistics for a specific user.
    Includes message counts, activity timeline, top intents, and preferences.
    """
    data = analytics.get_user_stats(user_id)
    return UserStatsResponse(**data)


@app.get("/dashboard")
async def serve_dashboard():
    """
    Serve the analytics dashboard HTML page.
    The dashboard provides a visual overview of system analytics.
    """
    dashboard_path = os.path.join(frontend_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        return {"message": "Dashboard not found. Place dashboard.html in the frontend/ directory."}


# ====================
# Advanced Memory Endpoints
# ====================

@app.get("/memory/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get the evolved user profile with personality, interests, and communication style."""
    try:
        profile = memory_system.get_user_profile(user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting profile: {str(e)}")


@app.get("/memory/{user_id}/ltm")
async def get_long_term_memory(user_id: str, type: str = None):
    """
    Get user's long-term memories.
    Optional query param: ?type=fact|preference|topic|event
    """
    try:
        memories = memory_system.get_ltm(user_id, memory_type=type, limit=20)
        return {"user_id": user_id, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting LTM: {str(e)}")


@app.post("/memory/{user_id}/ltm")
async def add_long_term_memory(user_id: str, memory: dict):
    """
    Manually add a long-term memory.
    Body: {"memory_type": "fact", "content": "User studies CS", "importance": 0.8}
    """
    try:
        memory_system.save_to_ltm(
            user_id=user_id,
            memory_type=memory.get("memory_type", "fact"),
            content=memory.get("content", ""),
            importance=memory.get("importance", 0.5)
        )
        return {"message": "Memory saved successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving LTM: {str(e)}")


@app.get("/memory/{user_id}/session")
async def get_current_session(user_id: str):
    """Get the current active session and its short-term memory."""
    try:
        session_id = memory_system.get_active_session(user_id)
        if session_id is None:
            return {"user_id": user_id, "session_id": None, "messages": [], "active": False}

        messages = memory_system.get_stm(session_id, limit=50)
        return {
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages,
            "active": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")


@app.post("/memory/{user_id}/session/end")
async def end_user_session(user_id: str):
    """End the user's current session (triggers profile evolution and LTM extraction)."""
    try:
        session_id = memory_system.get_active_session(user_id)
        if session_id:
            memory_system.end_session(session_id)
            return {"message": "Session ended", "session_id": session_id}
        return {"message": "No active session found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")


# ====================
# Frontend Serving
# ====================

# Determine the path to the frontend directory
# The frontend/ folder is at the same level as backend/
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
async def serve_frontend():
    """
    Serve the frontend index.html file.
    This is the main entry point for the web UI.
    """
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"message": "Frontend not found. Place index.html in the frontend/ directory."}


# Mount static files from the frontend directory
# This serves CSS, JS, images, etc.
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
