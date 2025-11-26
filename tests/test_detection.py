"""
Unit tests for detection services.
"""

import numpy as np
import pytest

from app.schemas.requests import DetectionRequest
from app.schemas.responses import (
    BoundingBox,
    DetectionResponse,
    FaceDetection,
    FrameDetection,
    ProcessingSummary,
    SourceDimensions,
)


class TestBoundingBox:
    """Tests for BoundingBox schema."""

    def test_create_bounding_box(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x=100, y=200, width=150, height=200)
        assert bbox.x == 100
        assert bbox.y == 200
        assert bbox.width == 150
        assert bbox.height == 200

    def test_bounding_box_from_dict(self):
        """Test creating from dictionary."""
        data = {"x": 50, "y": 100, "width": 200, "height": 250}
        bbox = BoundingBox(**data)
        assert bbox.x == 50
        assert bbox.height == 250


class TestFaceDetection:
    """Tests for FaceDetection schema."""

    def test_create_face_detection(self):
        """Test creating a face detection."""
        bbox = BoundingBox(x=100, y=50, width=80, height=100)
        face = FaceDetection(track_id=1, bbox=bbox, confidence=0.95)
        assert face.track_id == 1
        assert face.confidence == 0.95
        assert face.bbox.x == 100

    def test_face_detection_with_landmarks(self):
        """Test face detection with landmarks."""
        bbox = BoundingBox(x=100, y=50, width=80, height=100)
        landmarks = {
            "left_eye": (0.3, 0.4),
            "right_eye": (0.7, 0.4),
            "nose": (0.5, 0.6),
        }
        face = FaceDetection(
            track_id=1, bbox=bbox, confidence=0.92, landmarks=landmarks
        )
        assert face.landmarks is not None
        assert face.landmarks["nose"] == (0.5, 0.6)


class TestDetectionRequest:
    """Tests for DetectionRequest schema."""

    def test_create_request(self):
        """Test creating a detection request."""
        request = DetectionRequest(
            job_id="test-job-123",
            video_s3_key="users/user1/videos/test.mp4",
        )
        assert request.job_id == "test-job-123"
        assert request.frame_interval_seconds == 2.0  # default
        assert request.detect_faces is True  # default
        assert request.detect_poses is True  # default

    def test_request_with_custom_options(self):
        """Test request with custom options."""
        request = DetectionRequest(
            job_id="custom-job",
            video_s3_key="videos/source.mp4",
            frame_interval_seconds=1.5,
            detect_faces=True,
            detect_poses=False,
            start_time_ms=5000,
            end_time_ms=30000,
        )
        assert request.frame_interval_seconds == 1.5
        assert request.detect_poses is False
        assert request.start_time_ms == 5000
        assert request.end_time_ms == 30000

    def test_request_validation(self):
        """Test request validation."""
        # Frame interval must be between 0.5 and 10.0
        with pytest.raises(ValueError):
            DetectionRequest(
                job_id="test",
                video_s3_key="test.mp4",
                frame_interval_seconds=0.1,  # Too small
            )


class TestDetectionResponse:
    """Tests for DetectionResponse schema."""

    def test_create_response(self):
        """Test creating a detection response."""
        response = DetectionResponse(
            job_id="test-job",
            status="completed",
            source_dimensions=SourceDimensions(width=1920, height=1080),
            frame_interval_ms=2000,
            frames=[],
            tracks=[],
            summary=ProcessingSummary(
                total_frames=100,
                faces_detected=95,
                poses_detected=90,
                unique_face_tracks=1,
                unique_pose_tracks=1,
                processing_time_ms=5000,
            ),
        )
        assert response.status == "completed"
        assert response.source_dimensions.width == 1920
        assert response.summary.total_frames == 100

    def test_response_with_frames(self):
        """Test response with frame detections."""
        frame = FrameDetection(
            index=0,
            timestamp_ms=0,
            faces=[
                FaceDetection(
                    track_id=1,
                    bbox=BoundingBox(x=100, y=50, width=80, height=100),
                    confidence=0.9,
                )
            ],
            poses=[],
        )
        response = DetectionResponse(
            job_id="test",
            status="completed",
            source_dimensions=SourceDimensions(width=1280, height=720),
            frame_interval_ms=1000,
            frames=[frame],
            tracks=[],
            summary=ProcessingSummary(
                total_frames=1,
                faces_detected=1,
                poses_detected=0,
                unique_face_tracks=1,
                unique_pose_tracks=0,
                processing_time_ms=100,
            ),
        )
        assert len(response.frames) == 1
        assert len(response.frames[0].faces) == 1
        assert response.frames[0].faces[0].track_id == 1


class TestFaceDetectorService:
    """Tests for FaceDetector service (requires mocking)."""

    def test_bbox_to_center(self):
        """Test converting bbox to center coordinates."""
        # Simulating what the detector does
        bbox = (100, 50, 80, 100)  # x, y, w, h
        source_width = 1920
        source_height = 1080

        center_x = (bbox[0] + bbox[2] / 2) / source_width
        center_y = (bbox[1] + bbox[3] / 2) / source_height

        assert 0.0 < center_x < 1.0
        assert 0.0 < center_y < 1.0
        assert abs(center_x - 0.073) < 0.01  # (100 + 40) / 1920


class TestTrackerService:
    """Tests for ObjectTracker service (requires mocking)."""

    def test_iou_calculation(self):
        """Test IOU calculation between bboxes."""
        # Simulating IOU calculation
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

        # Identical boxes
        bbox1 = (100, 100, 50, 50)
        assert calculate_iou(bbox1, bbox1) == 1.0

        # Non-overlapping boxes
        bbox2 = (200, 200, 50, 50)
        assert calculate_iou(bbox1, bbox2) == 0.0

        # Partially overlapping boxes
        bbox3 = (125, 125, 50, 50)
        iou = calculate_iou(bbox1, bbox3)
        assert 0.0 < iou < 1.0


class TestTrackSummary:
    """Tests for track summary generation."""

    def test_average_bbox_calculation(self):
        """Test calculating average bounding box from track history."""
        bboxes = [
            (100, 50, 80, 100),
            (105, 52, 82, 98),
            (95, 48, 78, 102),
        ]

        avg_x = sum(b[0] for b in bboxes) // len(bboxes)
        avg_y = sum(b[1] for b in bboxes) // len(bboxes)
        avg_w = sum(b[2] for b in bboxes) // len(bboxes)
        avg_h = sum(b[3] for b in bboxes) // len(bboxes)

        assert avg_x == 100
        assert avg_y == 50
        assert avg_w == 80
        assert avg_h == 100

    def test_average_confidence_calculation(self):
        """Test calculating average confidence."""
        confidences = [0.9, 0.85, 0.92, 0.88]
        avg = sum(confidences) / len(confidences)
        assert abs(avg - 0.8875) < 0.001



