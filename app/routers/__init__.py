"""
FastAPI routers for the clipping worker.
"""

from app.routers import detection, health

__all__ = ["health", "detection"]

