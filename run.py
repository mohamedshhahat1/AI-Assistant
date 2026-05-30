"""Convenience script to start the AI Assistant application."""

import uvicorn


def main():
    print("=" * 50)
    print("  AI Assistant - Starting up...")
    print("  URL: http://0.0.0.0:8000")
    print("=" * 50)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
