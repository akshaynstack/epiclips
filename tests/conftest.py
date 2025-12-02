"""
Pytest configuration and fixtures.
"""

import os
import sys

import pytest

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def sample_image():
    """Create a sample test image."""
    import numpy as np

    # Create a simple test image (640x480, 3 channels)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simple "face" (circle)
    center = (320, 240)
    radius = 100
    color = (200, 180, 160)  # Skin-like color

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2:
                image[y, x] = color

    return image


@pytest.fixture(scope="session")
def sample_bbox():
    """Sample bounding box for testing."""
    return {
        "x": 220,
        "y": 140,
        "width": 200,
        "height": 200,
    }


@pytest.fixture
def mock_s3_client(mocker):
    """Mock S3 client for testing."""
    mock = mocker.MagicMock()
    mock.download_video.return_value = "/tmp/test_video.mp4"
    mock.upload_json.return_value = "detection/test.json"
    return mock


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return {
        "aws_region": "us-east-1",
        "s3_bucket": "test-bucket",
        "frame_interval_seconds": 2.0,
        "face_confidence_threshold": 0.5,
        "pose_confidence_threshold": 0.5,
        "temp_directory": "/tmp/test-worker",
        "max_workers": 4,
        "max_render_workers": 3,
    }



