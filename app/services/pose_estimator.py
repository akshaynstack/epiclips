"""
Pose estimation service using MediaPipe.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe pose landmark indices
POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class PoseDetectionResult:
    """Result of pose detection for a single person."""

    keypoints: Dict[str, Optional[Tuple[float, float]]]
    confidence: float
    visibility: Dict[str, float] = field(default_factory=dict)
    gesture: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


@dataclass
class FramePoseDetections:
    """Pose detections for a single frame."""

    frame_index: int
    timestamp_ms: int
    detections: List[PoseDetectionResult] = field(default_factory=list)


class PoseEstimator:
    """
    MediaPipe-based pose estimation service.
    
    Extracts body keypoints and detects gestures.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Initialize pose estimator.
        
        Args:
            confidence_threshold: Minimum confidence for pose detection
            model_complexity: MediaPipe model complexity (0, 1, or 2)
        """
        self.confidence_threshold = confidence_threshold
        self.model_complexity = model_complexity
        self._pose = None
        self._ready = False
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the MediaPipe pose model."""
        try:
            import mediapipe as mp
            
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=self.confidence_threshold,
            )
            self._ready = True
            logger.info(f"MediaPipe Pose model loaded (complexity={self.model_complexity})")

        except Exception as e:
            logger.error(f"Failed to load MediaPipe Pose model: {e}")
            self._ready = False
            raise

    def is_ready(self) -> bool:
        """Check if the estimator is ready."""
        return self._ready and self._pose is not None

    def estimate_pose(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FramePoseDetections:
        """
        Estimate pose in a single image.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FramePoseDetections with detected poses
        """
        if not self.is_ready():
            raise RuntimeError("Pose estimator not ready")

        detections = []
        
        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run pose estimation
            results = self._pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = {}
                visibility = {}
                
                for name, idx in POSE_LANDMARKS.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    # Normalize coordinates (MediaPipe returns normalized coords)
                    keypoints[name] = (landmark.x, landmark.y)
                    visibility[name] = landmark.visibility
                
                # Calculate average confidence from visibility
                avg_confidence = sum(visibility.values()) / len(visibility)
                
                # Calculate bounding box from keypoints
                bbox = self._calculate_bounding_box(keypoints, image.shape[:2])
                
                # Detect gesture
                gesture = self._detect_gesture(keypoints, visibility)
                
                if avg_confidence >= self.confidence_threshold:
                    detections.append(PoseDetectionResult(
                        keypoints=keypoints,
                        confidence=avg_confidence,
                        visibility=visibility,
                        gesture=gesture,
                        bounding_box=bbox,
                    ))

        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")

        return FramePoseDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    def estimate_pose_batch(
        self,
        images: List[Tuple[int, int, np.ndarray]],  # (index, timestamp_ms, image)
    ) -> List[FramePoseDetections]:
        """
        Estimate poses in a batch of images.
        
        Args:
            images: List of (frame_index, timestamp_ms, image) tuples
            
        Returns:
            List of FramePoseDetections for each image
        """
        results = []
        
        for frame_index, timestamp_ms, image in images:
            detection = self.estimate_pose(image, frame_index, timestamp_ms)
            results.append(detection)

        logger.info(f"Processed batch of {len(images)} frames, "
                   f"total poses: {sum(len(r.detections) for r in results)}")
        
        return results

    def estimate_from_file(
        self,
        file_path: str,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FramePoseDetections:
        """
        Estimate pose in an image file.
        
        Args:
            file_path: Path to the image file
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FramePoseDetections with detected poses
        """
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to read image: {file_path}")
            return FramePoseDetections(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detections=[],
            )
        
        return self.estimate_pose(image, frame_index, timestamp_ms)

    def _calculate_bounding_box(
        self,
        keypoints: Dict[str, Optional[Tuple[float, float]]],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box from keypoints."""
        valid_points = [p for p in keypoints.values() if p is not None]
        if not valid_points:
            return None
        
        h, w = image_shape
        xs = [int(p[0] * w) for p in valid_points]
        ys = [int(p[1] * h) for p in valid_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add padding
        padding = int(min(w, h) * 0.05)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _detect_gesture(
        self,
        keypoints: Dict[str, Optional[Tuple[float, float]]],
        visibility: Dict[str, float],
    ) -> Optional[str]:
        """
        Detect simple gestures from keypoints.
        
        Detects:
        - pointing: One arm extended
        - waving: Hand above shoulder with arm extended
        - thumbs_up: Thumb significantly above other fingers
        - arms_crossed: Wrists near opposite shoulders
        """
        try:
            # Get key points
            left_wrist = keypoints.get("left_wrist")
            right_wrist = keypoints.get("right_wrist")
            left_shoulder = keypoints.get("left_shoulder")
            right_shoulder = keypoints.get("right_shoulder")
            left_elbow = keypoints.get("left_elbow")
            right_elbow = keypoints.get("right_elbow")
            nose = keypoints.get("nose")
            
            if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return None
            
            # Calculate distances and angles
            # Pointing: arm extended (elbow relatively straight)
            # Waving: hand above head/shoulder
            
            # Check if either hand is above head
            if nose and left_wrist:
                if left_wrist[1] < nose[1] and visibility.get("left_wrist", 0) > 0.5:
                    return "waving"
            
            if nose and right_wrist:
                if right_wrist[1] < nose[1] and visibility.get("right_wrist", 0) > 0.5:
                    return "waving"
            
            # Check for pointing (hand extended to side)
            if left_shoulder and left_wrist:
                dx = abs(left_wrist[0] - left_shoulder[0])
                dy = abs(left_wrist[1] - left_shoulder[1])
                if dx > 0.3 and dy < 0.2:  # Arm extended horizontally
                    return "pointing"
            
            if right_shoulder and right_wrist:
                dx = abs(right_wrist[0] - right_shoulder[0])
                dy = abs(right_wrist[1] - right_shoulder[1])
                if dx > 0.3 and dy < 0.2:
                    return "pointing"
            
            return None

        except Exception as e:
            logger.debug(f"Gesture detection failed: {e}")
            return None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "loaded": self._ready,
            "model_type": "MediaPipe Pose",
            "model_complexity": self.model_complexity,
            "confidence_threshold": self.confidence_threshold,
        }

