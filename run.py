"""
Convenience script to start the AI Assistant application.

For local development:
    python run.py

For production (Render/Railway), the Procfile is used instead.
"""

import os
import uvicorn


def main():
    """Start the AI Assistant server."""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 50)
    print("  🤖 AI Assistant - Starting up...")
    print(f"  URL: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"  Dashboard: http://{host}:{port}/dashboard")
    print("=" * 50)

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=os.environ.get("ENV", "development") == "development"
    )


if __name__ == "__main__":
    main()
