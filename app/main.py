"""
FastAPI application entry point for the Clipping Detection Worker.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import detection, health
from app.services.face_detector import FaceDetector
from app.services.pose_estimator import PoseEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model instances (loaded on startup)
face_detector: FaceDetector | None = None
pose_estimator: PoseEstimator | None = None

# Semaphore for limiting concurrent detection jobs
# This prevents memory/CPU overload when multiple jobs arrive
job_semaphore: asyncio.Semaphore | None = None


def get_face_detector() -> FaceDetector:
    """Get the global face detector instance."""
    if face_detector is None:
        raise RuntimeError("Face detector not initialized")
    return face_detector


def get_pose_estimator() -> PoseEstimator:
    """Get the global pose estimator instance."""
    if pose_estimator is None:
        raise RuntimeError("Pose estimator not initialized")
    return pose_estimator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    Loads ML models on startup and cleans up on shutdown.
    """
    global face_detector, pose_estimator, job_semaphore

    settings = get_settings()
    logger.info("Starting Clipping Detection Worker...")

    # Create temp directory
    os.makedirs(settings.temp_directory, exist_ok=True)
    logger.info(f"Temp directory: {settings.temp_directory}")

    # Initialize job semaphore for concurrency control
    # This limits how many detection jobs can run simultaneously
    job_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    logger.info(f"Max concurrent jobs: {settings.max_concurrent_jobs}")

    # Load ML models
    logger.info("Loading YOLO face detection model...")
    face_detector = FaceDetector(
        model_path=settings.yolo_model_path,
        confidence_threshold=settings.face_confidence_threshold,
    )
    logger.info("YOLO model loaded successfully")

    logger.info("Loading MediaPipe pose estimation model...")
    pose_estimator = PoseEstimator(
        confidence_threshold=settings.pose_confidence_threshold,
    )
    logger.info("MediaPipe model loaded successfully")

    # Store in app state for dependency injection
    app.state.face_detector = face_detector
    app.state.pose_estimator = pose_estimator
    app.state.job_semaphore = job_semaphore

    logger.info("All models loaded. Worker ready to accept requests.")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Clipping Detection Worker...")
    face_detector = None
    pose_estimator = None
    job_semaphore = None
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="ViewCreator Clipping Worker",
    description="Video detection service using YOLO face detection, MediaPipe pose estimation, and DeepSORT tracking",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(detection.router, prefix="/detect", tags=["Detection"])


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "viewcreator-clipping-worker",
        "version": "1.0.0",
        "status": "running",
    }

