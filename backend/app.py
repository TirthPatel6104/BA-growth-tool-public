"""
Compatibility wrapper for FastAPI app entrypoint.

Preferred command:
    uvicorn main:app --reload --port 8000

Backward-compatible command:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

from main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
