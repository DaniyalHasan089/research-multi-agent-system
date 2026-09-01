"""
Entry point — starts the FastAPI app via uvicorn.

Usage:
    python main.py

The server starts at http://localhost:8000
  - Open http://localhost:8000 in your browser to use the app.
  - API docs available at http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Auto-reload on code changes during development
        reload_dirs=["."],    # Watch the repo root for changes
    )
