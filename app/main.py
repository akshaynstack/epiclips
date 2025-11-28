"""
FastAPI application entry point for ViewCreator Genesis.

Genesis is ViewCreator's media processing engine, providing:
1. Video detection (YOLO face detection, MediaPipe pose estimation, DeepSORT tracking)
2. Full AI clipping pipeline (transcription, intelligence planning, rendering)
"""

import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import ai_clipping, detection, health
from app.services.detection_pipeline import DetectionPipeline
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
detection_pipeline: DetectionPipeline | None = None

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


def get_detection_pipeline() -> DetectionPipeline:
    """Get the global detection pipeline instance."""
    if detection_pipeline is None:
        raise RuntimeError("Detection pipeline not initialized")
    return detection_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    Loads ML models on startup and cleans up on shutdown.
    """
    global face_detector, pose_estimator, detection_pipeline, job_semaphore

    settings = get_settings()
    logger.info("Starting ViewCreator Genesis...")

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

    # Initialize detection pipeline (used by AI clipping)
    logger.info("Initializing detection pipeline...")
    detection_pipeline = DetectionPipeline(
        face_detector=face_detector,
        pose_estimator=pose_estimator,
    )
    logger.info("Detection pipeline initialized")

    # Store in app state for dependency injection
    app.state.face_detector = face_detector
    app.state.pose_estimator = pose_estimator
    app.state.detection_pipeline = detection_pipeline
    app.state.job_semaphore = job_semaphore

    # Verify external tools
    _verify_external_tools()

    logger.info("All models loaded. Genesis ready to accept requests.")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down ViewCreator Genesis...")
    face_detector = None
    pose_estimator = None
    detection_pipeline = None
    job_semaphore = None
    
    # Clean up temp directory
    if os.path.isdir(settings.temp_directory):
        try:
            shutil.rmtree(settings.temp_directory)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
    
    logger.info("Shutdown complete")


def _verify_external_tools():
    """Verify that required external tools are available."""
    tools = {
        "ffmpeg": "FFmpeg for video rendering",
        "ffprobe": "FFprobe for video analysis",
        "yt-dlp": "yt-dlp for YouTube downloads",
    }
    
    for tool, description in tools.items():
        if shutil.which(tool):
            logger.info(f"✓ {description} available")
        else:
            logger.warning(f"✗ {description} NOT FOUND - some features may not work")


# Create FastAPI application
app = FastAPI(
    title="ViewCreator Genesis",
    description="""
Genesis - ViewCreator's media processing engine.

AI-powered video processing and content creation service.

## Features

### Detection API (`/detect`)
- YOLO face detection
- MediaPipe pose estimation
- DeepSORT object tracking

### AI Clipping API (`/ai-clipping`)
- YouTube video download via yt-dlp
- Audio transcription via Groq Whisper (216x realtime)
- Intelligent clip planning via Gemini (OpenRouter)
- Smart cropping with face tracking
- Viral-style caption generation
- FFmpeg rendering to 9:16 portrait
- S3 upload of final clips

## Usage

1. Submit a job: `POST /ai-clipping/jobs`
2. Poll status: `GET /ai-clipping/jobs/{job_id}`
3. Retrieve clips from the S3 URLs in the response
    """,
    version="2.0.0",
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
app.include_router(ai_clipping.router, tags=["AI Clipping"])


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    settings = get_settings()
    return {
        "service": "viewcreator-genesis",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "detection": "YOLO + MediaPipe + DeepSORT",
            "ai_clipping": "Transcription + Intelligence + Rendering",
        },
        "docs": "/docs",
    }

