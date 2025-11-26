"""
Face detection service using YOLOv8.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionResult:
    """Result of face detection on a single frame."""

    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[dict] = None  # Facial landmarks if available
    embedding: Optional[np.ndarray] = None  # Face embedding for tracking


@dataclass
class FrameFaceDetections:
    """Face detections for a single frame."""

    frame_index: int
    timestamp_ms: int
    detections: List[FaceDetectionResult] = field(default_factory=list)


class FaceDetector:
    """
    YOLO-based face detection service.
    
    Uses YOLOv8-face model for accurate face detection.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize face detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._ready = False
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            # Workaround for PyTorch 2.6+ strict weights_only loading
            # Monkey-patch torch.load to use weights_only=False for YOLO models
            import torch
            _original_torch_load = torch.load
            
            def _patched_torch_load(*args, **kwargs):
                # Force weights_only=False for ultralytics model loading
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _patched_torch_load
            
            from ultralytics import YOLO

            # Check if model file exists, otherwise use default yolov8n
            if os.path.exists(self.model_path):
                logger.info(f"Loading YOLO model from {self.model_path}")
                self._model = YOLO(self.model_path)
            else:
                # Use standard YOLOv8n model (will auto-download)
                # For face detection, we'll detect "person" class and focus on upper body
                logger.warning(f"Model {self.model_path} not found, using yolov8n.pt")
                self._model = YOLO("yolov8n.pt")

            # Restore original torch.load
            torch.load = _original_torch_load

            # Move to device
            self._model.to(self.device)
            self._ready = True
            logger.info(f"YOLO model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._ready = False
            raise

    def is_ready(self) -> bool:
        """Check if the detector is ready."""
        return self._ready and self._model is not None

    def detect_faces(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FrameFaceDetections:
        """
        Detect faces in a single image.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameFaceDetections with all detected faces
        """
        if not self.is_ready():
            raise RuntimeError("Face detector not ready")

        detections = []
        
        try:
            # Run YOLO inference
            results = self._model(
                image,
                conf=self.confidence_threshold,
                verbose=False,
            )

            if results and len(results) > 0:
                result = results[0]
                
                # Process detections
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # For standard YOLO, class 0 is "person"
                    # For face-specific models, class 0 is "face"
                    if cls_id == 0 and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        
                        # For person detection, estimate face region as upper 30% of bbox
                        # This is a fallback when using non-face-specific model
                        if not self._is_face_model():
                            # Adjust to face region (upper portion of person bbox)
                            face_h = int(h * 0.3)
                            # Center horizontally, take upper portion
                            face_w = min(face_h, w)
                            face_x = x + (w - face_w) // 2
                            face_y = y
                            x, y, w, h = face_x, face_y, face_w, face_h
                        
                        detections.append(FaceDetectionResult(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                        ))

        except Exception as e:
            logger.error(f"Face detection failed: {e}")

        return FrameFaceDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    def detect_faces_batch(
        self,
        images: List[Tuple[int, int, np.ndarray]],  # (index, timestamp_ms, image)
    ) -> List[FrameFaceDetections]:
        """
        Detect faces in a batch of images.
        
        Args:
            images: List of (frame_index, timestamp_ms, image) tuples
            
        Returns:
            List of FrameFaceDetections for each image
        """
        results = []
        
        for frame_index, timestamp_ms, image in images:
            detection = self.detect_faces(image, frame_index, timestamp_ms)
            results.append(detection)

        logger.info(f"Processed batch of {len(images)} frames, "
                   f"total faces: {sum(len(r.detections) for r in results)}")
        
        return results

    def detect_from_file(
        self,
        file_path: str,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FrameFaceDetections:
        """
        Detect faces in an image file.
        
        Args:
            file_path: Path to the image file
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameFaceDetections with all detected faces
        """
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to read image: {file_path}")
            return FrameFaceDetections(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detections=[],
            )
        
        return self.detect_faces(image, frame_index, timestamp_ms)

    def _is_face_model(self) -> bool:
        """Check if the loaded model is a face-specific model."""
        # Check model name or number of classes
        if self._model is None:
            return False
        
        model_name = self.model_path.lower()
        return "face" in model_name

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "device": self.device,
            "is_face_model": self._is_face_model(),
            "confidence_threshold": self.confidence_threshold,
        }

