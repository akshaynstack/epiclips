"""
Response schemas for the detection API.

These schemas define the JSON format that will be consumed by the NestJS API.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels."""

    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")


class FaceDetection(BaseModel):
    """Face detection result for a single face in a frame."""

    track_id: int = Field(..., description="Consistent ID for this face across frames")
    bbox: BoundingBox = Field(..., description="Bounding box of the face")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    landmarks: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None,
        description="Facial landmarks (eyes, nose, mouth) as normalized coordinates",
    )


class PoseKeypoints(BaseModel):
    """Body pose keypoints from MediaPipe."""

    # Head
    nose: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    left_ear: Optional[Tuple[float, float]] = None
    right_ear: Optional[Tuple[float, float]] = None

    # Upper body
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None

    # Hands
    left_pinky: Optional[Tuple[float, float]] = None
    right_pinky: Optional[Tuple[float, float]] = None
    left_index: Optional[Tuple[float, float]] = None
    right_index: Optional[Tuple[float, float]] = None
    left_thumb: Optional[Tuple[float, float]] = None
    right_thumb: Optional[Tuple[float, float]] = None

    # Lower body
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None

    # Feet
    left_heel: Optional[Tuple[float, float]] = None
    right_heel: Optional[Tuple[float, float]] = None
    left_foot_index: Optional[Tuple[float, float]] = None
    right_foot_index: Optional[Tuple[float, float]] = None


class PoseDetection(BaseModel):
    """Pose detection result for a single person in a frame."""

    track_id: int = Field(
        ..., description="Consistent ID for this person across frames"
    )
    keypoints: Dict[str, Optional[Tuple[float, float]]] = Field(
        ..., description="Body keypoints as normalized coordinates (0.0-1.0)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall pose detection confidence"
    )
    gesture: Optional[str] = Field(
        default=None,
        description="Detected gesture (e.g., 'pointing', 'waving', 'thumbs_up')",
    )


class FrameDetection(BaseModel):
    """Detection results for a single frame."""

    index: int = Field(..., description="Frame index in the sequence")
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds from video start")
    faces: List[FaceDetection] = Field(
        default_factory=list, description="Detected faces in this frame"
    )
    poses: List[PoseDetection] = Field(
        default_factory=list, description="Detected poses in this frame"
    )


class TrackSummary(BaseModel):
    """Summary of a tracked object across all frames."""

    track_id: int = Field(..., description="Track identifier")
    track_type: str = Field(..., description="Type of track ('face' or 'pose')")
    first_frame: int = Field(..., description="First frame where this track appears")
    last_frame: int = Field(..., description="Last frame where this track appears")
    frame_count: int = Field(..., description="Number of frames this track appears in")
    avg_bbox: Optional[BoundingBox] = Field(
        default=None, description="Average bounding box across all frames"
    )
    avg_confidence: float = Field(..., description="Average detection confidence")


class SourceDimensions(BaseModel):
    """Source video dimensions."""

    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")


class ProcessingSummary(BaseModel):
    """Summary of the detection processing."""

    total_frames: int = Field(..., description="Total number of frames processed")
    faces_detected: int = Field(..., description="Total face detections across all frames")
    poses_detected: int = Field(..., description="Total pose detections across all frames")
    unique_face_tracks: int = Field(..., description="Number of unique face tracks")
    unique_pose_tracks: int = Field(..., description="Number of unique pose tracks")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class DetectionResponse(BaseModel):
    """
    Complete detection response containing all detection data.
    
    This is the main output format consumed by the NestJS API.
    """

    job_id: str = Field(..., description="Job identifier from the request")
    status: str = Field(..., description="Job status: 'completed', 'failed', 'processing'")
    source_dimensions: SourceDimensions = Field(
        ..., description="Source video dimensions"
    )
    frame_interval_ms: int = Field(
        ..., description="Interval between frames in milliseconds"
    )
    frames: List[FrameDetection] = Field(
        default_factory=list, description="Detection results per frame"
    )
    tracks: List[TrackSummary] = Field(
        default_factory=list, description="Summary of all tracked objects"
    )
    summary: ProcessingSummary = Field(..., description="Processing summary statistics")
    results_s3_key: Optional[str] = Field(
        default=None, description="S3 key where full results JSON is stored"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if status is 'failed'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "source_dimensions": {"width": 1920, "height": 1080},
                "frame_interval_ms": 2000,
                "frames": [
                    {
                        "index": 0,
                        "timestamp_ms": 0,
                        "faces": [
                            {
                                "track_id": 1,
                                "bbox": {"x": 423, "y": 156, "width": 187, "height": 234},
                                "confidence": 0.94,
                            }
                        ],
                        "poses": [
                            {
                                "track_id": 1,
                                "keypoints": {
                                    "nose": [0.45, 0.22],
                                    "left_shoulder": [0.38, 0.35],
                                    "right_shoulder": [0.52, 0.35],
                                },
                                "confidence": 0.87,
                                "gesture": None,
                            }
                        ],
                    }
                ],
                "tracks": [
                    {
                        "track_id": 1,
                        "track_type": "face",
                        "first_frame": 0,
                        "last_frame": 149,
                        "frame_count": 150,
                        "avg_bbox": {"x": 450, "y": 160, "width": 190, "height": 240},
                        "avg_confidence": 0.91,
                    }
                ],
                "summary": {
                    "total_frames": 150,
                    "faces_detected": 142,
                    "poses_detected": 145,
                    "unique_face_tracks": 1,
                    "unique_pose_tracks": 1,
                    "processing_time_ms": 4523,
                },
                "results_s3_key": "users/user123/detection/results.json",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether the service is ready to accept requests")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    face_detector: str = Field(..., description="Face detector status")
    pose_estimator: str = Field(..., description="Pose estimator status")



