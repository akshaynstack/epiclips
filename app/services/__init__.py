"""
Services for the clipping worker.

Includes:
- Detection services (MediaPipe, Haar Cascade, DeepSORT)
- AI clipping services (transcription, intelligence, rendering)
"""

from app.services.detection_pipeline import DetectionPipeline
from app.services.face_detector import FaceDetector
from app.services.frame_extractor import FrameExtractor
from app.services.pose_estimator import PoseEstimator
from app.services.s3_client import S3Client
from app.services.tracker import ObjectTracker

# AI Clipping services
from app.services.ai_clipping_pipeline import AIClippingPipeline
from app.services.caption_generator import CaptionGeneratorService
from app.services.intelligence_planner import IntelligencePlannerService
from app.services.rendering_service import RenderingService
from app.services.s3_upload_service import S3UploadService
from app.services.transcription_service import TranscriptionService
from app.services.video_downloader import VideoDownloaderService

__all__ = [
    # Detection
    "S3Client",
    "FrameExtractor",
    "FaceDetector",
    "PoseEstimator",
    "ObjectTracker",
    "DetectionPipeline",
    # AI Clipping
    "AIClippingPipeline",
    "VideoDownloaderService",
    "TranscriptionService",
    "IntelligencePlannerService",
    "CaptionGeneratorService",
    "RenderingService",
    "S3UploadService",
]



