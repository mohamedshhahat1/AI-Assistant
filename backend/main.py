"""
AI Assistant Backend - FastAPI Application
==========================================
This is the main backend server for the AI Assistant project.
It handles chat requests, memory management, and serves the frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

# Import our AI engine components
from ai_engine import IntentDetector, ResponseEngine, MemorySystem


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
# Application Setup
# ====================

# Create the FastAPI application instance
app = FastAPI(
    title="AI Assistant API",
    description="A smart AI assistant with memory and intent detection",
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


@app.on_event("startup")
async def startup_event():
    """
    Initialize AI engine components when the server starts.
    This runs once before the server begins accepting requests.
    """
    global intent_detector, response_engine, memory_system

    print("Initializing AI engine components...")

    # Create instances of each AI component
    intent_detector = IntentDetector()
    response_engine = ResponseEngine()
    memory_system = MemorySystem()

    print("AI engine components initialized successfully!")


# ====================
# API Endpoints
# ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes a user message and returns an AI response.

    Steps:
    1. Load the user's memory (past interactions, preferences)
    2. Detect the intent of the user's message
    3. Generate an appropriate response using intent + memory context
    4. Save this interaction to the user's memory
    5. Return the response with metadata
    """
    try:
        # Step 1: Load user memory for context
        user_memory = memory_system.load_memory(request.user_id)
        memory_used = user_memory is not None and len(user_memory.get("history", [])) > 0

        # Step 2: Detect the intent of the message
        # (e.g., "greeting", "question", "farewell", "request")
        intent_result = intent_detector.detect(request.message)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Step 3: Generate a response using the detected intent and memory context
        response_text = response_engine.generate(
            message=request.message,
            intent=intent,
            memory=user_memory
        )

        # Step 4: Save this interaction to memory for future context
        memory_system.save_interaction(
            user_id=request.user_id,
            message=request.message,
            response=response_text,
            intent=intent
        )

        # Step 5: Return the response
        return ChatResponse(
            response=response_text,
            intent=intent,
            confidence=confidence,
            memory_used=memory_used
        )

    except Exception as e:
        # If anything goes wrong, return a helpful error message
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
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
