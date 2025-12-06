"""
Test script for the new FaceTracker and improved face detection.

This script validates:
1. FaceTracker can track faces across frames
2. Dominant face selection works correctly
3. Adaptive and multi-scale detection functions
4. Integration with smart layout detector

Usage:
    python tests/test_face_tracking.py [video_path]
    
If no video path is provided, it will use a test pattern.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_face_tracker_basic():
    """Test basic FaceTracker functionality."""
    from app.services.face_tracker import FaceTracker, TrackingResult
    
    logger.info("Testing FaceTracker basic functionality...")
    
    tracker = FaceTracker()
    
    # Simulate processing frames with face detections
    # Frame 0: One face at (100, 100, 50, 50)
    result1 = tracker.process_frame(
        detections=[(100, 100, 50, 50, 0.9)],
        frame_idx=0,
        timestamp_ms=0,
        frame_width=1920,
        frame_height=1080,
    )
    assert result1.num_detections == 1, f"Expected 1 detection, got {result1.num_detections}"
    assert result1.num_active_tracks == 1, f"Expected 1 track, got {result1.num_active_tracks}"
    
    # Frame 1: Same face moved slightly
    result2 = tracker.process_frame(
        detections=[(105, 102, 52, 51, 0.85)],
        frame_idx=1,
        timestamp_ms=33,
        frame_width=1920,
        frame_height=1080,
    )
    assert result2.num_active_tracks == 1, "Should still be 1 track (same face)"
    
    # Frame 2-6: Continue tracking to build up track length
    for i in range(2, 7):
        tracker.process_frame(
            detections=[(100 + i * 2, 100 + i, 50, 50, 0.9)],
            frame_idx=i,
            timestamp_ms=i * 33,
            frame_width=1920,
            frame_height=1080,
        )
    
    # Get final result
    final_result = tracker.get_result()
    
    assert len(final_result.tracks) >= 1, f"Expected at least 1 valid track, got {len(final_result.tracks)}"
    assert final_result.dominant_track is not None, "Should have a dominant track"
    assert final_result.dominant_track.frames_visible >= 5, "Dominant track should have many visible frames"
    
    logger.info(f"✓ Basic tracking test passed: {len(final_result.tracks)} tracks, dominant={final_result.dominant_track_id}")
    logger.info(f"  Dominant face visibility: {final_result.dominant_face_visibility:.1%}")
    logger.info(f"  Dominance score: {final_result.dominant_track.dominance_score:.3f}")
    
    return True


async def test_face_tracker_multiple_faces():
    """Test tracking multiple faces and selecting dominant."""
    from app.services.face_tracker import FaceTracker
    
    logger.info("Testing multiple face tracking...")
    
    tracker = FaceTracker()
    
    # Simulate 10 frames with 2 faces
    # Face 1: Large, bottom-center (like a talking head)
    # Face 2: Small, top-left (like a thumbnail)
    for i in range(10):
        tracker.process_frame(
            detections=[
                (800, 600, 300, 300, 0.95),  # Large face, bottom-center
                (50, 50, 80, 80, 0.7),        # Small face, top-left
            ],
            frame_idx=i,
            timestamp_ms=i * 100,
            frame_width=1920,
            frame_height=1080,
        )
    
    result = tracker.get_result()
    
    assert len(result.tracks) >= 2, f"Expected at least 2 tracks, got {len(result.tracks)}"
    assert result.dominant_track is not None, "Should have a dominant track"
    
    # The large centered face should be dominant
    dom_bbox = result.dominant_track.smoothed_bbox
    dom_x, dom_y, dom_w, dom_h = dom_bbox
    
    logger.info(f"✓ Multiple face tracking passed: {len(result.tracks)} tracks")
    logger.info(f"  Dominant face: {dom_w}x{dom_h} at ({dom_x}, {dom_y})")
    logger.info(f"  Dominance score: {result.dominant_track.dominance_score:.3f}")
    
    # The dominant should be the larger face
    assert dom_w > 200, f"Dominant face should be large, got width={dom_w}"
    
    return True


async def test_face_detector_multiscale():
    """Test multi-scale face detection."""
    from app.services.face_detector import FaceDetector
    import numpy as np
    
    logger.info("Testing multi-scale face detection...")
    
    detector = FaceDetector(confidence_threshold=0.4)
    
    if not detector.is_ready():
        logger.warning("Face detector not ready (missing backends)")
        return False
    
    # Create a simple test image (blank for now, just testing the API)
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    test_image.fill(128)  # Gray background
    
    # Test adaptive detection
    result = detector.detect_faces_adaptive(
        test_image,
        frame_index=0,
        timestamp_ms=0,
        expected_face_count=1,
    )
    
    logger.info(f"✓ Adaptive detection API works: {len(result.detections)} faces found")
    
    # Test multi-scale detection
    result_ms = detector.detect_faces_multiscale(
        test_image,
        frame_index=0,
        timestamp_ms=0,
        scales=[1.0, 1.5],
    )
    
    logger.info(f"✓ Multi-scale detection API works: {len(result_ms.detections)} faces found")
    
    # Get model info
    info = detector.get_model_info()
    logger.info(f"  Detector backends: {info['backends']}")
    logger.info(f"  Min face area ratio: {info['min_face_area_ratio']}")
    
    detector.close()
    return True


async def test_video_tracking(video_path: str):
    """Test face tracking on a real video."""
    from app.services.face_tracker import track_faces_in_video, get_dominant_face_crop_region
    
    logger.info(f"Testing video tracking on: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Track faces for first 10 seconds
    result = await track_faces_in_video(
        video_path=video_path,
        start_ms=0,
        end_ms=10000,
        sample_fps=5.0,
    )
    
    logger.info(f"✓ Video tracking complete:")
    logger.info(f"  Total frames processed: {result.total_frames}")
    logger.info(f"  Video dimensions: {result.frame_width}x{result.frame_height}")
    logger.info(f"  Number of tracks: {len(result.tracks)}")
    logger.info(f"  Avg faces per frame: {result.avg_faces_per_frame:.2f}")
    
    if result.dominant_track:
        logger.info(f"  Dominant track ID: {result.dominant_track_id}")
        logger.info(f"  Dominant face visibility: {result.dominant_face_visibility:.1%}")
        logger.info(f"  Dominance score: {result.dominant_track.dominance_score:.3f}")
        
        # Test crop region calculation
        crop = get_dominant_face_crop_region(
            result,
            timestamp_ms=5000,
            target_aspect_ratio=9/16,
        )
        if crop:
            logger.info(f"  Crop region at 5s: {crop}")
    else:
        logger.warning("  No dominant face found in video")
    
    return True


async def test_smart_layout_with_tracking(video_path: str):
    """Test smart layout detector with face tracking integration."""
    from app.services.smart_layout_detector import SmartLayoutDetector
    
    logger.info(f"Testing smart layout with tracking on: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    detector = SmartLayoutDetector(use_speaker_driven_layout=False, use_vlm_detection=False)
    
    # Test the new analyze_with_face_tracking method
    layout, tracking = await detector.analyze_with_face_tracking(
        video_path=video_path,
        start_ms=0,
        end_ms=10000,
        frame_width=1920,
        frame_height=1080,
        sample_fps=5.0,
    )
    
    logger.info(f"✓ Smart layout with tracking complete:")
    logger.info(f"  Dominant layout: {layout.dominant_layout}")
    logger.info(f"  Has transitions: {layout.has_transitions}")
    logger.info(f"  Number of segments: {len(layout.layout_segments)}")
    logger.info(f"  Corner facecam bbox: {layout.corner_facecam_bbox}")
    
    if tracking:
        logger.info(f"  Tracking: {len(tracking.tracks)} tracks, dominant visibility={tracking.dominant_face_visibility:.1%}")
    
    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Face Tracking Test Suite")
    logger.info("=" * 60)
    
    # Basic tests (no video required)
    tests_passed = 0
    tests_failed = 0
    
    try:
        if await test_face_tracker_basic():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        logger.error(f"✗ Basic tracking test failed: {e}")
        tests_failed += 1
    
    try:
        if await test_face_tracker_multiple_faces():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        logger.error(f"✗ Multiple face tracking test failed: {e}")
        tests_failed += 1
    
    try:
        if await test_face_detector_multiscale():
            tests_passed += 1
        else:
            tests_failed += 1
    except Exception as e:
        logger.error(f"✗ Multi-scale detection test failed: {e}")
        tests_failed += 1
    
    # Video tests (if video path provided)
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        try:
            if await test_video_tracking(video_path):
                tests_passed += 1
            else:
                tests_failed += 1
        except Exception as e:
            logger.error(f"✗ Video tracking test failed: {e}")
            tests_failed += 1
        
        try:
            if await test_smart_layout_with_tracking(video_path):
                tests_passed += 1
            else:
                tests_failed += 1
        except Exception as e:
            logger.error(f"✗ Smart layout test failed: {e}")
            tests_failed += 1
    else:
        logger.info("\n[Skipping video tests - no video path provided]")
        logger.info("Usage: python tests/test_face_tracking.py [video_path]")
    
    logger.info("=" * 60)
    logger.info(f"Tests complete: {tests_passed} passed, {tests_failed} failed")
    logger.info("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
