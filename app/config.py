"""
Configuration module using Pydantic Settings for environment variable management.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "viewcreator-clipping-worker"
    debug: bool = False
    log_level: str = "INFO"

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: str = "viewcreator-media"

    # Model paths
    yolo_model_path: str = "/app/models/yolov8n-face.pt"
    yolo_model_name: str = "yolov8n-face.pt"

    # Processing settings
    frame_interval_seconds: float = 2.0
    max_concurrent_jobs: int = 2
    temp_directory: str = "/tmp/clipping-worker"

    # Detection settings
    face_confidence_threshold: float = 0.5
    pose_confidence_threshold: float = 0.5
    tracking_max_age: int = 30  # Frames before a track is deleted

    # API settings
    api_timeout_seconds: int = 300
    max_video_duration_seconds: int = 3600  # 1 hour max

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

