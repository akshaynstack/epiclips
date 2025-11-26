"""
Face detection service with multi-tier detection fallback.

Detection Priority:
1. MediaPipe FaceDetection (purpose-built, most accurate for faces)
2. YOLOv8-face model (if available)
3. YOLOv8 person -> face estimate (fallback)
4. Haar Cascade (final fallback)

Based on epiriumaiclips architecture for superior face tracking.
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
    detection_method: str = "unknown"  # Which detector found this face


@dataclass
class FrameFaceDetections:
    """Face detections for a single frame."""

    frame_index: int
    timestamp_ms: int
    detections: List[FaceDetectionResult] = field(default_factory=list)


class FaceDetector:
    """
    Multi-tier face detection service with intelligent fallback.

    Uses a priority chain of detectors for maximum reliability:
    1. MediaPipe FaceDetection short-range (best for close faces)
    2. MediaPipe FaceDetection full-range (better for small/distant faces)
    3. YOLOv8-face model (if available)
    4. YOLOv8 person detection with face estimation
    5. Haar Cascade (classical CV fallback)

    Also provides outlier rejection to filter false positives.
    """

    # Face size bounds as fraction of frame area
    # Lowered from 0.5% to 0.1% to detect small webcam faces in screen shares
    MIN_FACE_AREA_RATIO = 0.001  # 0.1% of frame (allows ~45x45 face in 1080p)
    MAX_FACE_AREA_RATIO = 0.30   # 30% of frame

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize face detector with multi-tier fallback.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Detection backends
        self._mediapipe_detector = None
        self._mediapipe_detector_fullrange = None  # Full-range model for small/distant faces
        self._yolo_model = None
        self._haar_cascade = None
        self._dnn_net = None

        self._ready = False
        self._load_detectors()

    def _load_detectors(self) -> None:
        """Load all available detection backends."""
        detectors_loaded = []
        
        # 1. MediaPipe FaceDetection short-range (primary - best for close faces)
        try:
            import mediapipe as mp
            self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (better for close faces)
                min_detection_confidence=self.confidence_threshold,
            )
            detectors_loaded.append("MediaPipe FaceDetection (short-range)")
            logger.info("MediaPipe FaceDetection short-range loaded (primary detector)")

            # 1b. Also load full-range model for small/distant faces (like webcam overlays)
            self._mediapipe_detector_fullrange = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 for full-range (better for small/distant faces)
                min_detection_confidence=max(0.3, self.confidence_threshold - 0.2),  # Lower threshold
            )
            detectors_loaded.append("MediaPipe FaceDetection (full-range)")
            logger.info("MediaPipe FaceDetection full-range loaded (fallback for small faces)")
        except ImportError:
            logger.warning("MediaPipe not available, will use fallback detectors")
        except Exception as e:
            logger.warning(f"MediaPipe FaceDetection failed to initialize: {e}")

        # 2. YOLO model (secondary)
        try:
            import torch
            _original_torch_load = torch.load
            
            def _patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _patched_torch_load
            
            from ultralytics import YOLO

            if os.path.exists(self.model_path):
                logger.info(f"Loading YOLO model from {self.model_path}")
                self._yolo_model = YOLO(self.model_path)
                detectors_loaded.append(f"YOLO ({self.model_path})")
            else:
                logger.info(f"Model {self.model_path} not found, using yolov8n.pt")
                self._yolo_model = YOLO("yolov8n.pt")
                detectors_loaded.append("YOLO (yolov8n.pt person detection)")

            torch.load = _original_torch_load
            self._yolo_model.to(self.device)
            logger.info(f"YOLO model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")

        # 3. OpenCV DNN face detector (tertiary fallback)
        try:
            # Try to load OpenCV's DNN face detector
            prototxt_path = cv2.data.haarcascades.replace('haarcascades', 'opencv_face_detector.pbtxt')
            model_path = cv2.data.haarcascades.replace('haarcascades', 'opencv_face_detector_uint8.pb')
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self._dnn_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                detectors_loaded.append("OpenCV DNN")
                logger.info("OpenCV DNN face detector loaded")
        except Exception as e:
            logger.debug(f"OpenCV DNN face detector not available: {e}")

        # 4. Haar Cascade (final fallback)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self._haar_cascade.empty():
                self._haar_cascade = None
                logger.warning("Haar cascade failed to load")
            else:
                detectors_loaded.append("Haar Cascade")
                logger.info("Haar Cascade face detector loaded (final fallback)")
        except Exception as e:
            logger.warning(f"Haar cascade failed to load: {e}")

        self._ready = len(detectors_loaded) > 0
        logger.info(f"Face detector ready with {len(detectors_loaded)} backends: {detectors_loaded}")

    def is_ready(self) -> bool:
        """Check if at least one detector is ready."""
        return self._ready

    def detect_faces(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FrameFaceDetections:
        """
        Detect faces using multi-tier detection with fallback.
        
        Tries detectors in priority order until faces are found.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameFaceDetections with all detected faces
        """
        if not self.is_ready():
            raise RuntimeError("No face detection backends available")

        height, width = image.shape[:2]
        frame_area = width * height
        detections = []

        # Try MediaPipe short-range first (best for close faces)
        if self._mediapipe_detector is not None and not detections:
            try:
                detections = self._detect_with_mediapipe(image, width, height, use_fullrange=False)
                if detections:
                    logger.debug(f"MediaPipe short-range detected {len(detections)} faces")
            except Exception as e:
                logger.warning(f"MediaPipe short-range detection failed: {e}")

        # Try MediaPipe full-range if short-range found nothing (better for small webcam faces)
        if self._mediapipe_detector_fullrange is not None and not detections:
            try:
                detections = self._detect_with_mediapipe(image, width, height, use_fullrange=True)
                if detections:
                    logger.debug(f"MediaPipe full-range detected {len(detections)} faces")
            except Exception as e:
                logger.warning(f"MediaPipe full-range detection failed: {e}")

        # Try YOLO if MediaPipe found nothing
        if self._yolo_model is not None and not detections:
            try:
                detections = self._detect_with_yolo(image, width, height)
                if detections:
                    logger.debug(f"YOLO detected {len(detections)} faces")
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")

        # Try OpenCV DNN if still nothing
        if self._dnn_net is not None and not detections:
            try:
                detections = self._detect_with_dnn(image, width, height)
                if detections:
                    logger.debug(f"OpenCV DNN detected {len(detections)} faces")
            except Exception as e:
                logger.warning(f"OpenCV DNN detection failed: {e}")

        # Final fallback: Haar Cascade
        if self._haar_cascade is not None and not detections:
            try:
                detections = self._detect_with_haar(image, width, height)
                if detections:
                    logger.debug(f"Haar Cascade detected {len(detections)} faces")
            except Exception as e:
                logger.warning(f"Haar Cascade detection failed: {e}")

        # Filter by face size bounds
        pre_filter_count = len(detections)
        detections = self._filter_by_size(detections, frame_area)

        # Log when no faces found (helps diagnose detection issues)
        if not detections:
            if pre_filter_count > 0:
                logger.info(f"Frame {frame_index}: {pre_filter_count} faces detected but ALL filtered by size bounds")
            else:
                logger.debug(f"Frame {frame_index}: No faces detected by any backend (frame size: {width}x{height})")

        return FrameFaceDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    def _detect_with_mediapipe(
        self,
        image: np.ndarray,
        width: int,
        height: int,
        use_fullrange: bool = False,
    ) -> List[FaceDetectionResult]:
        """Detect faces using MediaPipe FaceDetection."""
        detections = []

        # Select detector based on range
        detector = self._mediapipe_detector_fullrange if use_fullrange else self._mediapipe_detector
        if detector is None:
            return detections

        # MediaPipe expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Minimum face size filter
                if w > 30 and h > 30:
                    detections.append(FaceDetectionResult(
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        detection_method="mediapipe_fullrange" if use_fullrange else "mediapipe",
                    ))

        return detections

    def _detect_with_yolo(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> List[FaceDetectionResult]:
        """Detect faces using YOLO model."""
        detections = []
        
        results = self._yolo_model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        if results and len(results) > 0:
            result = results[0]
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Class 0 is person in standard YOLO, face in face models
                if cls_id == 0 and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    
                    # If not a face-specific model, estimate face as upper 30% of person
                    if not self._is_face_model():
                        face_h = int(h * 0.3)
                        face_w = min(face_h, w)
                        face_x = x + (w - face_w) // 2
                        face_y = y
                        x, y, w, h = face_x, face_y, face_w, face_h
                    
                    if w > 30 and h > 30:
                        detections.append(FaceDetectionResult(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            detection_method="yolo" if self._is_face_model() else "yolo_person",
                        ))
        
        return detections

    def _detect_with_dnn(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> List[FaceDetectionResult]:
        """Detect faces using OpenCV DNN face detector."""
        detections = []
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self._dnn_net.setInput(blob)
        dnn_results = self._dnn_net.forward()
        
        for i in range(dnn_results.shape[2]):
            confidence = dnn_results[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(dnn_results[0, 0, i, 3] * width)
                y1 = int(dnn_results[0, 0, i, 4] * height)
                x2 = int(dnn_results[0, 0, i, 5] * width)
                y2 = int(dnn_results[0, 0, i, 6] * height)
                
                w = x2 - x1
                h = y2 - y1
                
                if w > 30 and h > 30:
                    detections.append(FaceDetectionResult(
                        bbox=(x1, y1, w, h),
                        confidence=float(confidence),
                        detection_method="opencv_dnn",
                    ))
        
        return detections

    def _detect_with_haar(
        self,
        image: np.ndarray,
        width: int,
        height: int,
    ) -> List[FaceDetectionResult]:
        """Detect faces using Haar Cascade (final fallback)."""
        detections = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(40, 40),
            maxSize=(int(width * 0.7), int(height * 0.7)),
        )
        
        for (x, y, w, h) in faces:
            # Estimate confidence based on face size and position
            face_area = w * h
            frame_area = width * height
            relative_size = face_area / frame_area
            confidence = min(0.9, 0.3 + relative_size * 2)  # Rough confidence estimate
            
            detections.append(FaceDetectionResult(
                bbox=(x, y, w, h),
                confidence=confidence,
                detection_method="haar_cascade",
            ))
        
        return detections

    def _filter_by_size(
        self,
        detections: List[FaceDetectionResult],
        frame_area: int,
    ) -> List[FaceDetectionResult]:
        """Filter detections by face size bounds."""
        filtered = []
        
        for det in detections:
            x, y, w, h = det.bbox
            face_area = w * h
            ratio = face_area / frame_area
            
            if self.MIN_FACE_AREA_RATIO <= ratio <= self.MAX_FACE_AREA_RATIO:
                filtered.append(det)
            else:
                logger.debug(f"Filtered face with area ratio {ratio:.3f} (bounds: {self.MIN_FACE_AREA_RATIO}-{self.MAX_FACE_AREA_RATIO})")
        
        return filtered

    def filter_outliers(
        self,
        detections: List[FaceDetectionResult],
        std_threshold: float = 2.0,
    ) -> List[FaceDetectionResult]:
        """
        Filter out outlier face detections using statistical analysis.
        
        Removes faces that are more than `std_threshold` standard deviations
        away from the median position. This helps eliminate false positives.
        
        Args:
            detections: List of face detections to filter
            std_threshold: Number of standard deviations for outlier threshold
            
        Returns:
            Filtered list with outliers removed
        """
        if len(detections) < 3:
            return detections
        
        try:
            # Calculate center positions
            centers = []
            for det in detections:
                x, y, w, h = det.bbox
                center_x = x + w / 2
                center_y = y + h / 2
                centers.append((center_x, center_y))
            
            x_positions = [c[0] for c in centers]
            y_positions = [c[1] for c in centers]
            
            # Calculate median and standard deviation
            median_x = np.median(x_positions)
            median_y = np.median(y_positions)
            std_x = np.std(x_positions)
            std_y = np.std(y_positions)
            
            # Filter outliers
            filtered = []
            for det, (cx, cy) in zip(detections, centers):
                # Check if within threshold
                x_ok = std_x == 0 or abs(cx - median_x) <= std_threshold * std_x
                y_ok = std_y == 0 or abs(cy - median_y) <= std_threshold * std_y
                
                if x_ok and y_ok:
                    filtered.append(det)
                else:
                    logger.debug(f"Filtered outlier face at ({cx:.0f}, {cy:.0f}), median=({median_x:.0f}, {median_y:.0f})")
            
            if filtered:
                logger.debug(f"Outlier rejection: {len(detections)} -> {len(filtered)} faces")
                return filtered
            else:
                # Don't filter everything - return original if all would be filtered
                return detections
                
        except Exception as e:
            logger.warning(f"Outlier filtering failed: {e}")
            return detections

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

        total_faces = sum(len(r.detections) for r in results)
        logger.info(f"Processed batch of {len(images)} frames, total faces: {total_faces}")
        
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
        """Check if the loaded YOLO model is face-specific."""
        if self._yolo_model is None:
            return False
        
        model_name = self.model_path.lower()
        return "face" in model_name

    def get_model_info(self) -> dict:
        """Get information about loaded detection backends."""
        backends = []

        if self._mediapipe_detector:
            backends.append("mediapipe_shortrange")
        if self._mediapipe_detector_fullrange:
            backends.append("mediapipe_fullrange")
        if self._yolo_model:
            backends.append("yolo" if self._is_face_model() else "yolo_person")
        if self._dnn_net is not None:
            backends.append("opencv_dnn")
        if self._haar_cascade is not None:
            backends.append("haar_cascade")

        return {
            "ready": self._ready,
            "backends": backends,
            "primary": backends[0] if backends else None,
            "yolo_model_path": self.model_path if self._yolo_model else None,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "min_face_area_ratio": self.MIN_FACE_AREA_RATIO,
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._mediapipe_detector:
            try:
                self._mediapipe_detector.close()
            except Exception:
                pass
        if self._mediapipe_detector_fullrange:
            try:
                self._mediapipe_detector_fullrange.close()
            except Exception:
                pass
        self._mediapipe_detector = None
        self._mediapipe_detector_fullrange = None
        self._yolo_model = None
        self._dnn_net = None
        self._haar_cascade = None
        self._ready = False
