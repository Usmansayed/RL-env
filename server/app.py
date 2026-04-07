"""
AVA Consciousness Evaluation Environment — server entry point.
Exposes the FastAPI application via uvicorn.
This file is the [project.scripts] entry point required by openenv validate.
"""
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the AVA environment server."""
    from src.ava.server import app  # noqa: F401 — import here to avoid circular issues

    uvicorn.run(
        "src.ava.server:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
