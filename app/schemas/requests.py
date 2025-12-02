"""
Request schemas for the detection API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request body for the /detect endpoint."""

    job_id: str = Field(..., description="Unique identifier for this detection job")
    video_s3_key: str = Field(
        ..., description="S3 key of the video to process (e.g., users/{user_id}/videos/source.mp4)"
    )
    frame_interval_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Interval between frame extractions in seconds",
    )
    detect_faces: bool = Field(
        default=True, description="Whether to run MediaPipe face detection"
    )
    detect_poses: bool = Field(
        default=True, description="Whether to run MediaPipe pose estimation"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Optional URL to POST results when processing completes (async mode)",
    )
    start_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional start time in milliseconds to process from",
    )
    end_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional end time in milliseconds to process until",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "video_s3_key": "users/user123/videos/source.mp4",
                "frame_interval_seconds": 2.0,
                "detect_faces": True,
                "detect_poses": True,
                "callback_url": None,
            }
        }


class DetectionStatusRequest(BaseModel):
    """Request to check detection job status."""

    job_id: str = Field(..., description="Job ID to check status for")



