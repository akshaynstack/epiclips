"""
FastAPI routers for the clipping worker.
"""

from app.routers import ai_clipping, detection, health

__all__ = ["health", "detection", "ai_clipping"]



