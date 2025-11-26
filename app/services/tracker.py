"""
Object tracking service using DeepSORT for maintaining identity across frames.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object across frames."""

    track_id: int
    track_type: str  # 'face' or 'pose'
    first_frame: int
    last_frame: int
    frame_count: int
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    @property
    def avg_bbox(self) -> Tuple[int, int, int, int]:
        """Calculate average bounding box."""
        if not self.bboxes:
            return (0, 0, 0, 0)
        
        avg_x = int(sum(b[0] for b in self.bboxes) / len(self.bboxes))
        avg_y = int(sum(b[1] for b in self.bboxes) / len(self.bboxes))
        avg_w = int(sum(b[2] for b in self.bboxes) / len(self.bboxes))
        avg_h = int(sum(b[3] for b in self.bboxes) / len(self.bboxes))
        
        return (avg_x, avg_y, avg_w, avg_h)

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class ObjectTracker:
    """
    Object tracker using DeepSORT algorithm.
    
    Maintains consistent IDs for detected objects across frames.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
    ):
        """
        Initialize object tracker.
        
        Args:
            max_age: Maximum frames to keep a track without updates
            n_init: Number of consecutive detections before a track is confirmed
            max_iou_distance: Maximum IOU distance for matching
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        
        self._face_tracker = None
        self._pose_tracker = None
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        
        self._init_trackers()

    def _init_trackers(self) -> None:
        """Initialize DeepSORT trackers."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            # Tracker for faces
            self._face_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
            )
            
            # Separate tracker for poses (if needed)
            self._pose_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
            )
            
            logger.info("DeepSORT trackers initialized")

        except ImportError:
            logger.warning("DeepSORT not available, using simple IOU tracker")
            self._face_tracker = None
            self._pose_tracker = None

    def update_faces(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],  # (bbox, confidence)
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """
        Update face tracker with new detections.
        
        Args:
            detections: List of (bbox, confidence) tuples
            frame: Current frame (for appearance features)
            frame_index: Current frame index
            
        Returns:
            List of (track_id, bbox, confidence) tuples
        """
        if not detections:
            return []

        if self._face_tracker is None:
            # Fallback to simple assignment
            return self._simple_track_update(detections, frame_index, "face")

        try:
            # Convert detections to DeepSORT format
            # DeepSORT expects: [([left, top, w, h], confidence, class)]
            ds_detections = []
            for bbox, conf in detections:
                x, y, w, h = bbox
                ds_detections.append(([x, y, w, h], conf, "face"))

            # Update tracker
            tracks = self._face_tracker.update_tracks(ds_detections, frame=frame)
            
            results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                bbox = (int(ltrb[0]), int(ltrb[1]), 
                       int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1]))
                conf = track.det_conf if track.det_conf else 0.5
                
                # Update track history
                self._update_track_history(track_id, "face", frame_index, bbox, conf)
                
                results.append((track_id, bbox, conf))
            
            return results

        except Exception as e:
            logger.error(f"Face tracking failed: {e}")
            return self._simple_track_update(detections, frame_index, "face")

    def update_poses(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],  # (bbox, confidence)
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """
        Update pose tracker with new detections.
        
        Args:
            detections: List of (bbox, confidence) tuples
            frame: Current frame
            frame_index: Current frame index
            
        Returns:
            List of (track_id, bbox, confidence) tuples
        """
        if not detections:
            return []

        if self._pose_tracker is None:
            return self._simple_track_update(detections, frame_index, "pose")

        try:
            ds_detections = []
            for bbox, conf in detections:
                x, y, w, h = bbox
                ds_detections.append(([x, y, w, h], conf, "pose"))

            tracks = self._pose_tracker.update_tracks(ds_detections, frame=frame)
            
            results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                # Offset pose IDs to avoid collision with face IDs
                track_id = track.track_id + 10000
                ltrb = track.to_ltrb()
                bbox = (int(ltrb[0]), int(ltrb[1]),
                       int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1]))
                conf = track.det_conf if track.det_conf else 0.5
                
                self._update_track_history(track_id, "pose", frame_index, bbox, conf)
                
                results.append((track_id, bbox, conf))
            
            return results

        except Exception as e:
            logger.error(f"Pose tracking failed: {e}")
            return self._simple_track_update(detections, frame_index, "pose")

    def _simple_track_update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        frame_index: int,
        track_type: str,
    ) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """
        Simple IOU-based tracking fallback.
        
        Uses basic IOU matching when DeepSORT is not available.
        """
        results = []
        
        for bbox, conf in detections:
            # Find best matching existing track
            best_match = None
            best_iou = 0.0
            
            for track_id, track in self._tracks.items():
                if track.track_type != track_type:
                    continue
                if track.last_frame < frame_index - self.max_age:
                    continue
                
                # Calculate IOU with last known bbox
                if track.bboxes:
                    iou = self._calculate_iou(bbox, track.bboxes[-1])
                    if iou > best_iou and iou > (1 - self.max_iou_distance):
                        best_iou = iou
                        best_match = track_id
            
            if best_match is not None:
                track_id = best_match
            else:
                track_id = self._next_id
                self._next_id += 1
            
            self._update_track_history(track_id, track_type, frame_index, bbox, conf)
            results.append((track_id, bbox, conf))
        
        return results

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
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

    def _update_track_history(
        self,
        track_id: int,
        track_type: str,
        frame_index: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        """Update track history with new detection."""
        if track_id not in self._tracks:
            self._tracks[track_id] = Track(
                track_id=track_id,
                track_type=track_type,
                first_frame=frame_index,
                last_frame=frame_index,
                frame_count=1,
                bboxes=[bbox],
                confidences=[confidence],
            )
        else:
            track = self._tracks[track_id]
            track.last_frame = frame_index
            track.frame_count += 1
            track.bboxes.append(bbox)
            track.confidences.append(confidence)

    def get_all_tracks(self) -> List[Track]:
        """Get all tracked objects."""
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_id = 1
        
        # Re-initialize DeepSORT trackers
        self._init_trackers()
        
        logger.info("Tracker state reset")

