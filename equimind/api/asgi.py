"""
ASGI Entry Point for Gunicorn / Uvicorn deployment.
"""

from equimind.api.server import app

__all__ = ["app"]
