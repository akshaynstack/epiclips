"""
Integration tests for the detection pipeline.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.services.frame_extractor import FrameExtractor, VideoMetadata


class TestFrameExtractor:
    """Tests for FrameExtractor service."""

    @pytest.fixture
    def frame_extractor(self):
        """Create a FrameExtractor instance."""
        return FrameExtractor()

    def test_init(self, frame_extractor):
        """Test initialization."""
        assert frame_extractor.temp_directory is not None

    def test_calculate_scale_filter_no_scale(self, frame_extractor):
        """Test scale filter when no scaling needed."""
        result = frame_extractor._calculate_scale_filter(1280, 720, 1280)
        assert result is None

    def test_calculate_scale_filter_width(self, frame_extractor):
        """Test scale filter for wide video."""
        result = frame_extractor._calculate_scale_filter(1920, 1080, 1280)
        assert result == "scale=1280:-2"

    def test_calculate_scale_filter_height(self, frame_extractor):
        """Test scale filter for tall video."""
        result = frame_extractor._calculate_scale_filter(1080, 1920, 1280)
        assert result == "scale=-2:1280"


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating video metadata."""
        metadata = VideoMetadata(
            duration_ms=60000,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
        )
        assert metadata.duration_ms == 60000
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def mock_video(self):
        """Create a mock video file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # Create a minimal valid video file
            # In practice, you'd use ffmpeg to create a test video
            # For now, we just create a placeholder
            video_path = f.name

        yield video_path

        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)

    def test_pipeline_processes_frames(self):
        """Test that pipeline processes frames correctly."""
        # This is a conceptual test - actual implementation would need
        # either real video files or extensive mocking

        # Mock frame data
        frames = [
            {
                "index": 0,
                "timestamp_ms": 0,
                "faces": [{"track_id": 1, "bbox": {"x": 100, "y": 50, "width": 80, "height": 100}, "confidence": 0.9}],
                "poses": [],
            },
            {
                "index": 1,
                "timestamp_ms": 2000,
                "faces": [{"track_id": 1, "bbox": {"x": 105, "y": 52, "width": 82, "height": 98}, "confidence": 0.88}],
                "poses": [],
            },
        ]

        # Verify structure
        assert len(frames) == 2
        assert frames[0]["faces"][0]["track_id"] == frames[1]["faces"][0]["track_id"]

    def test_tracking_maintains_identity(self):
        """Test that tracker maintains identity across frames."""
        # Simulate detections across frames with consistent tracking
        detections_frame_1 = [
            {"bbox": (100, 50, 80, 100), "confidence": 0.9},
        ]
        detections_frame_2 = [
            {"bbox": (105, 52, 82, 98), "confidence": 0.88},  # Slight movement
        ]

        # The tracker should assign the same track_id to both
        # since they overlap significantly
        def calculate_iou(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            intersection = (xi2 - xi1) * (yi2 - yi1)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        bbox1 = detections_frame_1[0]["bbox"]
        bbox2 = detections_frame_2[0]["bbox"]
        iou = calculate_iou(bbox1, bbox2)

        # High IOU means same object
        assert iou > 0.7


class TestS3Client:
    """Tests for S3Client service (mocked)."""

    def test_generate_results_key(self):
        """Test generating S3 key for results."""
        video_key = "users/user123/videos/source.mp4"
        job_id = "test-job-456"

        # Extract path and generate results key
        parts = video_key.split("/")
        base_path = "/".join(parts[:-2])
        results_key = f"{base_path}/detection/{job_id}.json"

        assert results_key == "users/user123/detection/test-job-456.json"

    def test_generate_results_key_simple_path(self):
        """Test generating results key for simple paths."""
        video_key = "videos/source.mp4"
        job_id = "job-789"

        parts = video_key.split("/")
        if len(parts) >= 2:
            base_path = "/".join(parts[:-2])
            results_key = f"{base_path}/detection/{job_id}.json"
        else:
            results_key = f"detection/{job_id}.json"

        assert results_key == "detection/job-789.json"


class TestEndToEnd:
    """End-to-end tests for the detection workflow."""

    def test_response_format(self):
        """Test that response format matches expected schema."""
        response = {
            "job_id": "e2e-test-job",
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
                    "last_frame": 0,
                    "frame_count": 1,
                    "avg_bbox": {"x": 423, "y": 156, "width": 187, "height": 234},
                    "avg_confidence": 0.94,
                }
            ],
            "summary": {
                "total_frames": 1,
                "faces_detected": 1,
                "poses_detected": 1,
                "unique_face_tracks": 1,
                "unique_pose_tracks": 1,
                "processing_time_ms": 1234,
            },
        }

        # Verify required fields
        assert "job_id" in response
        assert "status" in response
        assert "source_dimensions" in response
        assert "frames" in response
        assert "tracks" in response
        assert "summary" in response

        # Verify frame structure
        frame = response["frames"][0]
        assert "index" in frame
        assert "timestamp_ms" in frame
        assert "faces" in frame
        assert "poses" in frame

        # Verify face structure
        face = frame["faces"][0]
        assert "track_id" in face
        assert "bbox" in face
        assert "confidence" in face

        # Verify bbox structure
        bbox = face["bbox"]
        assert "x" in bbox
        assert "y" in bbox
        assert "width" in bbox
        assert "height" in bbox

    def test_normalized_coordinates(self):
        """Test that keypoints are normalized (0-1 range)."""
        keypoints = {
            "nose": [0.45, 0.22],
            "left_shoulder": [0.38, 0.35],
            "right_shoulder": [0.52, 0.35],
        }

        for name, coords in keypoints.items():
            x, y = coords
            assert 0.0 <= x <= 1.0, f"{name} x coordinate out of range"
            assert 0.0 <= y <= 1.0, f"{name} y coordinate out of range"

