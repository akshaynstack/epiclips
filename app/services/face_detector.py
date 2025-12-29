"""
Face detection service with multi-tier detection fallback.

Detection Priority:
1. MediaPipe FaceDetection short-range (purpose-built, most accurate for close-up faces)
2. MediaPipe FaceDetection full-range (for small/distant faces)
3. Haar Cascade (final fallback for edge cases)

Multi-tier architecture for robust face tracking with minimal dependencies.
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
    3. Haar Cascade (classical CV fallback)

    Also provides outlier rejection to filter false positives.
    """

    # Face size bounds as fraction of frame area
    # Lowered to 0.03% to detect very small webcam overlay faces in screen shares
    # 0.0003 = ~25x25 face in 1080p, 0.001 = ~45x45 face in 1080p
    MIN_FACE_AREA_RATIO = 0.0003  # 0.03% of frame (allows ~25x25 face in 1080p)
    MAX_FACE_AREA_RATIO = 0.30    # 30% of frame

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize face detector with multi-tier fallback.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Detection backends
        self._mediapipe_detector = None
        self._mediapipe_detector_fullrange = None  # Full-range model for small/distant faces
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

    def _reset_mediapipe_detectors(self) -> None:
        """
        Reset MediaPipe detectors to handle timestamp errors in concurrent usage.
        
        MediaPipe's internal state can become corrupted when used concurrently,
        causing "Packet timestamp mismatch" errors. Creating fresh instances resolves this.
        """
        try:
            import mediapipe as mp
            
            # Close existing detectors if they have close methods
            if self._mediapipe_detector is not None:
                try:
                    self._mediapipe_detector.close()
                except Exception:
                    pass
            if self._mediapipe_detector_fullrange is not None:
                try:
                    self._mediapipe_detector_fullrange.close()
                except Exception:
                    pass
            
            # Create fresh instances
            self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.confidence_threshold,
            )
            self._mediapipe_detector_fullrange = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=max(0.3, self.confidence_threshold - 0.2),
            )
            logger.debug("MediaPipe detectors reset successfully")
        except Exception as e:
            logger.warning(f"Failed to reset MediaPipe detectors: {e}")

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
        short_range_detections = []
        full_range_detections = []

        # OPTIMIZED: Run short-range first, only use full-range as fallback
        # This saves ~20-30% CPU by not running both detectors on every frame
        
        # Try MediaPipe short-range first (best for close faces - most common case)
        if self._mediapipe_detector is not None:
            try:
                short_range_detections = self._detect_with_mediapipe(image, width, height, use_fullrange=False)
                if short_range_detections:
                    logger.debug(f"MediaPipe short-range detected {len(short_range_detections)} faces")
            except Exception as e:
                # MediaPipe timestamp error often happens with concurrent usage
                # Reset detector and try again with fresh instance
                if "timestamp" in str(e).lower():
                    logger.debug("MediaPipe short-range timestamp error, recreating detector...")
                    self._reset_mediapipe_detectors()
                    try:
                        short_range_detections = self._detect_with_mediapipe(image, width, height, use_fullrange=False)
                    except Exception:
                        pass  # Fall through to other detectors
                else:
                    logger.warning(f"MediaPipe short-range detection failed: {e}")

        # Only try full-range if short-range found NO faces (fallback for small/distant faces)
        # This optimization saves ~20-30% CPU on frames with easily-detected faces
        if not short_range_detections and self._mediapipe_detector_fullrange is not None:
            try:
                full_range_detections = self._detect_with_mediapipe(image, width, height, use_fullrange=True)
                if full_range_detections:
                    logger.debug(f"MediaPipe full-range (fallback) detected {len(full_range_detections)} faces")
            except Exception as e:
                if "timestamp" in str(e).lower():
                    logger.debug("MediaPipe full-range timestamp error, recreating detector...")
                    self._reset_mediapipe_detectors()
                    try:
                        full_range_detections = self._detect_with_mediapipe(image, width, height, use_fullrange=True)
                    except Exception:
                        pass  # Fall through to other detectors
                else:
                    logger.warning(f"MediaPipe full-range detection failed: {e}")
        
        # Merge detections (in optimized mode, usually only one list has results)
        detections = self._merge_detections(short_range_detections, full_range_detections, width, height)
        if detections:
            logger.debug(f"MediaPipe: {len(detections)} faces (short-range: {len(short_range_detections)}, full-range fallback: {len(full_range_detections)})")

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
        """
        Detect faces using MediaPipe FaceDetection.
        
        Creates a fresh detector instance per call to avoid graph corruption
        in concurrent usage scenarios. The 'Empty packets' error occurs when
        multiple threads share a MediaPipe graph instance.
        """
        detections = []

        try:
            import mediapipe as mp
            
            # Create fresh detector instance for this call (thread-safe)
            # This avoids MediaPipe graph corruption under concurrent usage
            if use_fullrange:
                detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # Full-range model for small/distant faces
                    min_detection_confidence=max(0.3, self.confidence_threshold - 0.2),
                )
            else:
                detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,  # Short-range model for close faces
                    min_detection_confidence=self.confidence_threshold,
                )
            
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
                    
                    # Minimum face size filter (reduced for small webcam overlays)
                    if w > 20 and h > 20:
                        detections.append(FaceDetectionResult(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            detection_method="mediapipe_fullrange" if use_fullrange else "mediapipe",
                        ))
            
            # Clean up detector resources immediately
            detector.close()
            
        except ImportError:
            logger.warning("MediaPipe not available for detection")
        except Exception as e:
            logger.warning(f"MediaPipe {'full-range' if use_fullrange else 'short-range'} detection failed: {e}")

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
                
                # Minimum face size filter (reduced for small webcam overlays)
                if w > 20 and h > 20:
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
                # Log with actual dimensions for easier debugging
                reason = "too small" if ratio < self.MIN_FACE_AREA_RATIO else "too large"
                logger.debug(
                    f"Filtered face ({reason}): {w}x{h}px at ({x},{y}), "
                    f"area_ratio={ratio:.5f} (bounds: {self.MIN_FACE_AREA_RATIO}-{self.MAX_FACE_AREA_RATIO})"
                )
        
        return filtered

    def _merge_detections(
        self,
        short_range: List[FaceDetectionResult],
        full_range: List[FaceDetectionResult],
        frame_width: int,
        frame_height: int,
    ) -> List[FaceDetectionResult]:
        """
        Merge face detections from short-range and full-range MediaPipe models.
        
        Removes duplicates by checking for overlapping bounding boxes.
        Keeps the detection with higher confidence when faces overlap.
        
        This is critical for detecting both:
        - Large center faces (short-range is better)
        - Small corner webcam faces (full-range is better)
        
        Args:
            short_range: Detections from short-range model (close faces)
            full_range: Detections from full-range model (distant/small faces)
            frame_width: Frame width for IoU calculation
            frame_height: Frame height for IoU calculation
            
        Returns:
            Merged list with duplicates removed
        """
        if not full_range:
            return short_range
        if not short_range:
            return full_range
        
        # Start with all short-range detections (typically higher quality for close faces)
        merged = list(short_range)
        
        # Add full-range detections that don't overlap significantly with short-range
        for fr_det in full_range:
            is_duplicate = False
            fr_x, fr_y, fr_w, fr_h = fr_det.bbox
            
            for sr_det in short_range:
                sr_x, sr_y, sr_w, sr_h = sr_det.bbox
                
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(
                    (fr_x, fr_y, fr_w, fr_h),
                    (sr_x, sr_y, sr_w, sr_h)
                )
                
                # If significant overlap (IoU > 0.3), consider it a duplicate
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(fr_det)
                logger.debug(f"Added full-range face detection: {fr_det.bbox} (not overlapping with short-range)")
        
        return merged

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First box as (x, y, width, height)
            box2: Second box as (x, y, width, height)
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union

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

    def get_model_info(self) -> dict:
        """Get information about loaded detection backends."""
        backends = []

        if self._mediapipe_detector:
            backends.append("mediapipe_shortrange")
        if self._mediapipe_detector_fullrange:
            backends.append("mediapipe_fullrange")
        if self._dnn_net is not None:
            backends.append("opencv_dnn")
        if self._haar_cascade is not None:
            backends.append("haar_cascade")

        return {
            "ready": self._ready,
            "backends": backends,
            "primary": backends[0] if backends else None,
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
        self._dnn_net = None
        self._haar_cascade = None
        self._ready = False

    def detect_faces_adaptive(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
        previous_detections: Optional[List[FaceDetectionResult]] = None,
        expected_face_count: int = 0,
    ) -> FrameFaceDetections:
        """
        Detect faces with adaptive thresholds based on previous detections.
        
        This method adjusts detection parameters based on:
        1. Previous face detections (if faces were found before, try harder)
        2. Expected face count (if we expect faces, lower thresholds)
        
        This is useful for tracking scenarios where we know faces exist
        but may be temporarily hard to detect (motion blur, occlusion).
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            previous_detections: Previous frame's detections for adaptive search
            expected_face_count: Expected number of faces (0 = unknown)
            
        Returns:
            FrameFaceDetections with all detected faces
        """
        # First try normal detection
        result = self.detect_faces(image, frame_index, timestamp_ms)
        
        # If we got faces, return them
        if result.detections:
            return result
        
        # If we expected faces or had previous detections, try harder
        if previous_detections or expected_face_count > 0:
            logger.debug(f"Frame {frame_index}: No faces found, trying multi-scale detection...")
            return self.detect_faces_multiscale(image, frame_index, timestamp_ms)
        
        return result

    def detect_faces_multiscale(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
        scales: Optional[List[float]] = None,
    ) -> FrameFaceDetections:
        """
        Detect faces using multiple image scales for better small face detection.
        
        This method upscales the image to detect very small faces that might
        be missed at native resolution. Useful for small webcam overlays.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            scales: List of scale factors to try (default: [1.0, 1.5, 2.0])
            
        Returns:
            FrameFaceDetections with all detected faces (merged across scales)
        """
        if scales is None:
            scales = [1.0, 1.5, 2.0]
        
        height, width = image.shape[:2]
        frame_area = width * height
        all_detections: List[FaceDetectionResult] = []
        
        for scale in scales:
            if scale == 1.0:
                # Use normal detection at native scale
                scaled_image = image
                scaled_width, scaled_height = width, height
            else:
                # Upscale the image
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                
                # Skip if upscaled image is too large (>4K)
                if scaled_width > 3840 or scaled_height > 2160:
                    continue
                
                scaled_image = cv2.resize(
                    image,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Detect faces at this scale
            if self._mediapipe_detector_fullrange is not None:
                try:
                    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
                    results = self._mediapipe_detector_fullrange.process(rgb_image)
                    
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            confidence = detection.score[0]
                            
                            # Convert to absolute coords at scaled size, then back to original
                            x = int(bbox.xmin * scaled_width / scale)
                            y = int(bbox.ymin * scaled_height / scale)
                            w = int(bbox.width * scaled_width / scale)
                            h = int(bbox.height * scaled_height / scale)
                            
                            if w > 15 and h > 15:  # Even lower threshold for multiscale
                                all_detections.append(FaceDetectionResult(
                                    bbox=(x, y, w, h),
                                    confidence=confidence,
                                    detection_method=f"mediapipe_fullrange_x{scale}",
                                ))
                                logger.debug(f"Multiscale ({scale}x) detected face: {w}x{h} at ({x},{y})")
                except Exception as e:
                    logger.debug(f"Multiscale detection at {scale}x failed: {e}")
        
        if not all_detections:
            return FrameFaceDetections(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detections=[],
            )
        
        # Merge overlapping detections from different scales
        merged = self._merge_multiscale_detections(all_detections, width, height)
        
        # Filter by size bounds
        merged = self._filter_by_size(merged, frame_area)
        
        logger.debug(f"Multiscale detection found {len(merged)} unique faces across {len(scales)} scales")
        
        return FrameFaceDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=merged,
        )

    def _merge_multiscale_detections(
        self,
        detections: List[FaceDetectionResult],
        frame_width: int,
        frame_height: int,
    ) -> List[FaceDetectionResult]:
        """
        Merge face detections from multiple scales, removing duplicates.
        
        Uses NMS (Non-Maximum Suppression) to keep the best detection
        for each unique face.
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept: List[FaceDetectionResult] = []
        
        for det in sorted_dets:
            is_duplicate = False
            
            for kept_det in kept:
                iou = self._calculate_iou(det.bbox, kept_det.bbox)
                if iou > 0.4:  # Slightly higher threshold for NMS
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept

    def detect_faces_in_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> FrameFaceDetections:
        """
        Detect faces in a specific region of the image.
        
        This is useful for:
        1. Re-detecting faces near where they were last seen
        2. Focusing on known webcam overlay areas
        3. Reducing false positives from main screen content
        
        Args:
            image: Full image as numpy array (BGR format)
            region: Region to search (x, y, width, height)
            frame_index: Index of the frame
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameFaceDetections with faces found in the region
        """
        x, y, w, h = region
        height, width = image.shape[:2]
        
        # Clamp region to image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        if w < 30 or h < 30:
            return FrameFaceDetections(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detections=[],
            )
        
        # Crop region
        region_image = image[y:y+h, x:x+w]
        
        # Add padding to help detection (face detectors often need context)
        pad = 20
        padded_x = max(0, x - pad)
        padded_y = max(0, y - pad)
        padded_w = min(w + 2 * pad, width - padded_x)
        padded_h = min(h + 2 * pad, height - padded_y)
        
        region_image = image[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w]
        
        # Detect faces in region
        result = self.detect_faces(region_image, frame_index, timestamp_ms)
        
        # Adjust coordinates back to full image
        adjusted_detections = []
        for det in result.detections:
            dx, dy, dw, dh = det.bbox
            adjusted_detections.append(FaceDetectionResult(
                bbox=(dx + padded_x, dy + padded_y, dw, dh),
                confidence=det.confidence,
                landmarks=det.landmarks,
                embedding=det.embedding,
                detection_method=det.detection_method + "_region",
            ))
        
        return FrameFaceDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=adjusted_detections,
        )