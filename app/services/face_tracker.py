"""
Face Tracker Service - Frame-by-frame face tracking with temporal consistency.

This service provides robust face tracking for video clips:
1. Tracks faces across frames using centroid-based tracking (similar to SORT/DeepSort)
2. Identifies the MAIN/dominant person based on persistence, size, and position
3. Uses adaptive thresholds based on face detection history
4. Smooths face bounding boxes to reduce jitter

Key algorithm: Simple Online and Realtime Tracking (SORT) without deep features
- Uses IoU (Intersection over Union) for face matching
- Maintains face tracks with temporal smoothing
- Handles face occlusion and disappearance gracefully

This is the "top notch clipping agent algorithm which analyzes frame by frame"
that tracks the MAIN person throughout the clip.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Tracking thresholds
IOU_THRESHOLD = 0.3  # Minimum IoU to match faces between frames
MAX_FRAMES_DISAPPEARED = 30  # Max frames a face can be missing before dropped
MIN_TRACK_LENGTH = 5  # Minimum frames in a track to be considered valid

# Dominant face selection
# The dominant face is the MAIN person we want to track for split-screen
DOMINANT_FACE_MIN_AREA_RATIO = 0.005  # 0.5% of frame - must be visible
DOMINANT_FACE_PERSISTENCE_WEIGHT = 0.4  # Weight for how long face has been visible
DOMINANT_FACE_SIZE_WEIGHT = 0.3  # Weight for face size
DOMINANT_FACE_CENTER_WEIGHT = 0.2  # Weight for being centered
DOMINANT_FACE_BOTTOM_WEIGHT = 0.1  # Weight for being in bottom (typical webcam position)

# Bounding box smoothing (exponential moving average)
BBOX_SMOOTHING_ALPHA = 0.3  # Higher = more responsive, lower = smoother


@dataclass
class FaceTrack:
    """
    Represents a tracked face across multiple frames.
    
    This is the core data structure for face tracking. Each track has:
    - Unique ID for the tracked face
    - History of bounding boxes across frames
    - Smoothed bounding box for stable rendering
    - Metrics for determining if this is the dominant face
    """
    track_id: int
    
    # Bounding box history: list of (frame_idx, x, y, w, h, confidence)
    bbox_history: list[tuple[int, int, int, int, int, float]] = field(default_factory=list)
    
    # Smoothed bounding box (x, y, w, h) - updated every frame
    smoothed_bbox: Optional[tuple[int, int, int, int]] = None
    
    # Tracking state
    frames_visible: int = 0
    frames_disappeared: int = 0
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    
    # Computed metrics (updated on each frame)
    avg_area: float = 0.0
    avg_center_x: float = 0.5  # Normalized 0-1
    avg_center_y: float = 0.5  # Normalized 0-1
    persistence_ratio: float = 0.0  # frames_visible / total_frames_tracked
    
    # Dominant face score (0-1) - higher = more likely to be the main person
    dominance_score: float = 0.0
    
    def update_metrics(self, frame_width: int, frame_height: int, total_frames: int, preferred_position: Optional[str] = None) -> None:
        """Update computed metrics based on bbox history."""
        if not self.bbox_history:
            return
        
        # Calculate averages from history (last 30 frames or all if less)
        recent_history = self.bbox_history[-30:]
        
        areas = []
        centers_x = []
        centers_y = []
        
        for frame_idx, x, y, w, h, conf in recent_history:
            area = w * h
            areas.append(area)
            centers_x.append((x + w / 2) / frame_width if frame_width > 0 else 0.5)
            centers_y.append((y + h / 2) / frame_height if frame_height > 0 else 0.5)
        
        self.avg_area = sum(areas) / len(areas) if areas else 0
        self.avg_center_x = sum(centers_x) / len(centers_x) if centers_x else 0.5
        self.avg_center_y = sum(centers_y) / len(centers_y) if centers_y else 0.5
        
        # Persistence ratio
        total_tracked = self.last_seen_frame - self.first_seen_frame + 1
        self.persistence_ratio = self.frames_visible / total_tracked if total_tracked > 0 else 0
        
        # Calculate dominance score
        frame_area = frame_width * frame_height if frame_width > 0 and frame_height > 0 else 1
        area_ratio = self.avg_area / frame_area
        
        # Size score (larger = higher, capped at 30% of frame)
        size_score = min(area_ratio / 0.30, 1.0)
        
        # Center score (closer to horizontal center = higher)
        center_score = 1.0 - abs(self.avg_center_x - 0.5) * 2
        
        # Bottom score (lower in frame = higher, typical webcam position)
        bottom_score = self.avg_center_y  # 0 at top, 1 at bottom
        
        # Persistence score (more visible frames = higher)
        persistence_score = min(self.persistence_ratio / 0.8, 1.0)  # 80% visibility = max
        
        # Position match score (if preferred_position is set)
        position_match_score = 0.0
        position_disqualified = False
        if preferred_position:
            # Define target centers and STRICT bounds for quadrants
            # A corner webcam should be in the actual corner, not just "closer to corner than center"
            targets = {
                "top-left": {"center": (0.2, 0.2), "x_max": 0.4, "y_max": 0.4, "x_min": 0.0, "y_min": 0.0},
                "top-right": {"center": (0.8, 0.2), "x_min": 0.6, "y_max": 0.4, "x_max": 1.0, "y_min": 0.0},
                "bottom-left": {"center": (0.2, 0.8), "x_max": 0.4, "y_min": 0.6, "x_min": 0.0, "y_max": 1.0},
                "bottom-right": {"center": (0.8, 0.8), "x_min": 0.6, "y_min": 0.6, "x_max": 1.0, "y_max": 1.0},
            }
            target = targets.get(preferred_position)
            if target:
                tx, ty = target["center"]
                
                # STRICT BOUNDS CHECK: Face must be in the correct quadrant
                in_x_bounds = target.get("x_min", 0) <= self.avg_center_x <= target.get("x_max", 1)
                in_y_bounds = target.get("y_min", 0) <= self.avg_center_y <= target.get("y_max", 1)
                
                if in_x_bounds and in_y_bounds:
                    # Face is in the correct quadrant - score based on distance from center
                    dist = ((self.avg_center_x - tx)**2 + (self.avg_center_y - ty)**2)**0.5
                    position_match_score = max(0.0, 1.0 - dist * 2)
                else:
                    # Face is NOT in the requested quadrant - heavily penalize it
                    position_match_score = -1.0  # Negative score to demote this face
                    position_disqualified = True
        
        # Combine scores with weights
        if preferred_position:
            if position_disqualified:
                # Face is outside the requested quadrant - give very low score
                # but not zero so we can still use it as fallback if nothing better exists
                self.dominance_score = -0.5
            else:
                # Face is in the correct quadrant - prioritize position match heavily
                self.dominance_score = (
                    0.1 * size_score +
                    0.1 * center_score +
                    0.1 * bottom_score +
                    0.2 * persistence_score +
                    0.5 * position_match_score  # 50% weight on position match
                )
        else:
            self.dominance_score = (
                DOMINANT_FACE_SIZE_WEIGHT * size_score +
                DOMINANT_FACE_CENTER_WEIGHT * center_score +
                DOMINANT_FACE_BOTTOM_WEIGHT * bottom_score +
                DOMINANT_FACE_PERSISTENCE_WEIGHT * persistence_score
            )
    
    def update_smoothed_bbox(self, new_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Update smoothed bounding box using exponential moving average.
        
        This reduces jitter in face tracking for smoother rendering.
        """
        x, y, w, h = new_bbox
        
        if self.smoothed_bbox is None:
            self.smoothed_bbox = new_bbox
            return new_bbox
        
        sx, sy, sw, sh = self.smoothed_bbox
        
        # Exponential moving average
        alpha = BBOX_SMOOTHING_ALPHA
        new_smoothed = (
            int(alpha * x + (1 - alpha) * sx),
            int(alpha * y + (1 - alpha) * sy),
            int(alpha * w + (1 - alpha) * sw),
            int(alpha * h + (1 - alpha) * sh),
        )
        
        self.smoothed_bbox = new_smoothed
        return new_smoothed


@dataclass
class TrackedFrame:
    """Face tracking results for a single frame."""
    frame_idx: int
    timestamp_ms: int
    
    # All tracked faces in this frame
    tracked_faces: list[FaceTrack] = field(default_factory=list)
    
    # The dominant face (main person) in this frame, if any
    dominant_face: Optional[FaceTrack] = None
    dominant_face_bbox: Optional[tuple[int, int, int, int]] = None
    
    # Detection stats
    num_detections: int = 0
    num_active_tracks: int = 0


@dataclass
class TrackingResult:
    """Complete tracking result for a video clip."""
    
    # All face tracks found in the video
    tracks: list[FaceTrack] = field(default_factory=list)
    
    # The dominant/main face track (highest dominance score)
    dominant_track: Optional[FaceTrack] = None
    dominant_track_id: Optional[int] = None
    
    # Per-frame results
    frame_results: list[TrackedFrame] = field(default_factory=list)
    
    # Video info
    total_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    
    # Statistics
    avg_faces_per_frame: float = 0.0
    dominant_face_visibility: float = 0.0  # Percentage of frames with dominant face
    
    def get_dominant_bbox_at_frame(self, frame_idx: int) -> Optional[tuple[int, int, int, int]]:
        """Get the smoothed dominant face bbox at a specific frame."""
        if frame_idx < 0 or frame_idx >= len(self.frame_results):
            return None
        return self.frame_results[frame_idx].dominant_face_bbox
    
    def get_dominant_bbox_at_time(self, timestamp_ms: int, fps: float = 30.0) -> Optional[tuple[int, int, int, int]]:
        """Get the smoothed dominant face bbox at a specific timestamp."""
        if fps <= 0:
            return None
        
        frame_idx = int(timestamp_ms * fps / 1000)
        return self.get_dominant_bbox_at_frame(frame_idx)


class FaceTracker:
    """
    Frame-by-frame face tracker with temporal consistency.
    
    This tracker:
    1. Takes raw face detections from FaceDetector
    2. Associates faces across frames using IoU matching
    3. Maintains stable track IDs for each person
    4. Identifies the DOMINANT/MAIN person for split-screen cropping
    5. Provides smoothed bounding boxes for stable rendering
    
    Algorithm: Simplified SORT (Simple Online Realtime Tracking)
    - No Kalman filter (keeping it simple)
    - No deep appearance features (would need face embeddings)
    - Uses centroid distance + IoU for matching
    """
    
    def __init__(
        self,
        iou_threshold: float = IOU_THRESHOLD,
        max_disappeared: int = MAX_FRAMES_DISAPPEARED,
        min_track_length: int = MIN_TRACK_LENGTH,
    ):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.min_track_length = min_track_length
        
        # Active tracks (track_id -> FaceTrack)
        self._tracks: OrderedDict[int, FaceTrack] = OrderedDict()
        self._next_track_id = 0
        
        # Frame dimensions (set on first frame)
        self._frame_width = 0
        self._frame_height = 0
        
        # Tracking state
        self._current_frame_idx = 0
        self._frame_results: list[TrackedFrame] = []
    
    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._tracks.clear()
        self._next_track_id = 0
        self._current_frame_idx = 0
        self._frame_results.clear()
    
    def process_frame(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame_idx: int,
        timestamp_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> TrackedFrame:
        """
        Process a single frame's face detections and update tracks.
        
        Args:
            detections: List of (x, y, w, h, confidence) tuples
            frame_idx: Frame index in the video
            timestamp_ms: Timestamp in milliseconds
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            TrackedFrame with tracking results for this frame
        """
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._current_frame_idx = frame_idx
        
        # Step 1: Match existing tracks to new detections
        if not detections:
            # No detections - mark all tracks as disappeared
            self._mark_all_disappeared()
        elif not self._tracks:
            # No existing tracks - create new tracks for all detections
            for det in detections:
                self._create_track(det, frame_idx)
        else:
            # Match detections to existing tracks
            self._match_and_update(detections, frame_idx)
        
        # Step 2: Update metrics for all tracks
        total_frames = frame_idx + 1
        for track in self._tracks.values():
            track.update_metrics(frame_width, frame_height, total_frames)
        
        # Step 3: Remove stale tracks
        self._remove_stale_tracks()
        
        # Step 4: Find dominant face
        dominant_track = self._find_dominant_track()
        
        # Step 5: Build frame result
        active_tracks = [t for t in self._tracks.values() if t.frames_disappeared == 0]
        
        frame_result = TrackedFrame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            tracked_faces=active_tracks,
            dominant_face=dominant_track,
            dominant_face_bbox=dominant_track.smoothed_bbox if dominant_track else None,
            num_detections=len(detections),
            num_active_tracks=len(active_tracks),
        )
        
        self._frame_results.append(frame_result)
        
        return frame_result
    
    def _create_track(self, detection: tuple[int, int, int, int, float], frame_idx: int) -> FaceTrack:
        """Create a new face track from a detection."""
        x, y, w, h, confidence = detection
        
        track = FaceTrack(
            track_id=self._next_track_id,
            bbox_history=[(frame_idx, x, y, w, h, confidence)],
            smoothed_bbox=(x, y, w, h),
            frames_visible=1,
            frames_disappeared=0,
            last_seen_frame=frame_idx,
            first_seen_frame=frame_idx,
        )
        
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1
        
        return track
    
    def _match_and_update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame_idx: int,
    ) -> None:
        """Match new detections to existing tracks and update."""
        
        track_ids = list(self._tracks.keys())
        track_bboxes = [
            self._tracks[tid].smoothed_bbox or self._tracks[tid].bbox_history[-1][1:5]
            for tid in track_ids
        ]
        
        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_bbox in enumerate(track_bboxes):
            for j, det_bbox in enumerate(det_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Greedy matching by highest IoU
        matched_tracks = set()
        matched_detections = set()
        
        while True:
            # Find highest IoU
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            # Match this track to this detection
            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            
            self._update_track(track_id, detection, frame_idx)
            
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            
            # Zero out this row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        # Mark unmatched tracks as disappeared
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[track_id].frames_disappeared += 1
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self._create_track(detection, frame_idx)
    
    def _update_track(
        self,
        track_id: int,
        detection: tuple[int, int, int, int, float],
        frame_idx: int,
    ) -> None:
        """Update an existing track with a new detection."""
        track = self._tracks[track_id]
        x, y, w, h, confidence = detection
        
        track.bbox_history.append((frame_idx, x, y, w, h, confidence))
        track.frames_visible += 1
        track.frames_disappeared = 0
        track.last_seen_frame = frame_idx
        
        # Update smoothed bbox
        track.update_smoothed_bbox((x, y, w, h))
    
    def _mark_all_disappeared(self) -> None:
        """Mark all tracks as disappeared for this frame."""
        for track in self._tracks.values():
            track.frames_disappeared += 1
    
    def _remove_stale_tracks(self) -> None:
        """Remove tracks that have been missing too long."""
        to_remove = []
        for track_id, track in self._tracks.items():
            if track.frames_disappeared > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracks[track_id]
    
    def _find_dominant_track(self) -> Optional[FaceTrack]:
        """
        Find the dominant face track (the MAIN person).
        
        The dominant track is selected based on:
        1. Persistence (how long the face has been visible)
        2. Size (larger faces are more important)
        3. Position (center and bottom are preferred)
        4. Must have enough frames to be valid
        """
        valid_tracks = [
            t for t in self._tracks.values()
            if t.frames_visible >= self.min_track_length
            and t.frames_disappeared == 0  # Must be currently visible
        ]
        
        if not valid_tracks:
            return None
        
        # Return track with highest dominance score
        return max(valid_tracks, key=lambda t: t.dominance_score)
    
    def _calculate_iou(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
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
    
    def get_result(self, preferred_position: Optional[str] = None) -> TrackingResult:
        """
        Get the complete tracking result after processing all frames.
        
        Args:
            preferred_position: Optional preferred position ("bottom-right", etc.)
            
        Returns:
            TrackingResult with all tracks and dominant face information
        """
        # Get all tracks that have enough frames
        valid_tracks = [
            t for t in self._tracks.values()
            if t.frames_visible >= self.min_track_length
        ]
        
        # Recalculate dominance scores with preferred position if provided
        if preferred_position:
            for track in valid_tracks:
                track.update_metrics(self._frame_width, self._frame_height, len(self._frame_results), preferred_position)
                # Log position filtering info
                logger.debug(
                    f"Track {track.track_id}: pos=({track.avg_center_x:.2f}, {track.avg_center_y:.2f}), "
                    f"score={track.dominance_score:.3f}, preferred={preferred_position}"
                )
        
        # Find the overall dominant track (highest score, even if negative it's our best option)
        dominant_track = None
        if valid_tracks:
            dominant_track = max(valid_tracks, key=lambda t: t.dominance_score)
            if dominant_track.dominance_score < 0 and preferred_position:
                logger.warning(
                    f"No face found in {preferred_position} quadrant. Best match: "
                    f"track {dominant_track.track_id} at ({dominant_track.avg_center_x:.2f}, {dominant_track.avg_center_y:.2f}) "
                    f"with score {dominant_track.dominance_score:.3f}"
                )
        
        # Calculate statistics
        total_faces = sum(fr.num_detections for fr in self._frame_results)
        total_frames = len(self._frame_results)
        avg_faces = total_faces / total_frames if total_frames > 0 else 0
        
        # Calculate dominant face visibility
        dominant_visibility = 0.0
        if dominant_track:
            frames_with_dominant = sum(
                1 for fr in self._frame_results
                if fr.dominant_face and fr.dominant_face.track_id == dominant_track.track_id
            )
            dominant_visibility = frames_with_dominant / total_frames if total_frames > 0 else 0
        
        return TrackingResult(
            tracks=valid_tracks,
            dominant_track=dominant_track,
            dominant_track_id=dominant_track.track_id if dominant_track else None,
            frame_results=self._frame_results,
            total_frames=total_frames,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            avg_faces_per_frame=avg_faces,
            dominant_face_visibility=dominant_visibility,
        )


async def track_faces_in_video(
    video_path: str,
    start_ms: int = 0,
    end_ms: Optional[int] = None,
    sample_fps: float = 10.0,
    face_detector=None,
    preferred_position: Optional[str] = None,
    max_frames: Optional[int] = None,
    max_duration_ms: Optional[int] = None,
) -> TrackingResult:
    """
    Track faces throughout a video clip.
    
    This is the main entry point for face tracking. It:
    1. Samples frames at the specified FPS
    2. Detects faces in each frame
    3. Tracks faces across frames
    4. Identifies the dominant/main person
    
    Args:
        video_path: Path to the video file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds (None = end of video)
        sample_fps: Frames per second to sample
        face_detector: FaceDetector instance (will create one if not provided)
        preferred_position: Optional preferred position ("bottom-right", etc.)
        max_frames: Optional max sampled frames to process
        max_duration_ms: Optional maximum duration to analyze (from start_ms)
        
    Returns:
        TrackingResult with all tracking information
    """
    logger.info(f"Tracking faces in {video_path}: {start_ms}ms - {end_ms}ms at {sample_fps} FPS (preferred_pos={preferred_position})")
    
    # Initialize face detector if not provided
    if face_detector is None:
        from app.services.face_detector import FaceDetector
        face_detector = FaceDetector(confidence_threshold=0.4)  # Lower threshold for tracking
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return TrackingResult()
    
    try:
        tracking_start = time.monotonic()
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int(total_frames / video_fps * 1000) if video_fps > 0 else 0
        
        if end_ms is None or end_ms > duration_ms:
            end_ms = duration_ms

        if max_duration_ms is not None and max_duration_ms > 0:
            end_ms = min(end_ms, start_ms + int(max_duration_ms))
        
        logger.info(f"Video: {frame_width}x{frame_height}, {video_fps:.1f}fps, duration={duration_ms}ms")
        
        # Initialize tracker
        tracker = FaceTracker()

        if sample_fps <= 0:
            sample_fps = 1.0

        frame_step = max(1, int(round(video_fps / sample_fps))) if video_fps > 0 else 1
        start_frame = int((start_ms / 1000.0) * video_fps) if video_fps > 0 else 0
        end_frame = int((end_ms / 1000.0) * video_fps) if video_fps > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        next_sample_frame = start_frame
        processed_frames = 0

        while current_frame < end_frame:
            if max_frames is not None and processed_frames >= max_frames:
                break

            if current_frame == next_sample_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int((current_frame / video_fps) * 1000) if video_fps > 0 else 0

                detection_result = face_detector.detect_faces(
                    frame,
                    frame_index=current_frame,
                    timestamp_ms=timestamp_ms,
                )

                detections = [
                    (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence)
                    for d in detection_result.detections
                ]

                tracker.process_frame(
                    detections=detections,
                    frame_idx=processed_frames,
                    timestamp_ms=timestamp_ms,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                processed_frames += 1
                next_sample_frame += frame_step
                current_frame += 1
            else:
                # Fast-skip frames without decoding.
                if not cap.grab():
                    break
                current_frame += 1

            if processed_frames > 0 and processed_frames % 100 == 0:
                progress = (current_frame - start_frame) / max(1, (end_frame - start_frame)) * 100
                logger.debug(f"Tracking progress: {progress:.1f}% ({processed_frames} frames)")
        
        # Get final result
        result = tracker.get_result(preferred_position=preferred_position)
        
        logger.info(
            f"Tracking complete: {processed_frames} frames, "
            f"{len(result.tracks)} tracks, "
            f"dominant_track={result.dominant_track_id}, "
            f"visibility={result.dominant_face_visibility:.1%}, "
            f"time={time.monotonic() - tracking_start:.1f}s"
        )
        
        return result
        
    finally:
        cap.release()


def get_dominant_face_crop_region(
    tracking_result: TrackingResult,
    timestamp_ms: int,
    target_aspect_ratio: float = 9/16,  # Vertical video
    padding_ratio: float = 0.3,  # Add 30% padding around face
    min_size_ratio: float = 0.15,  # Crop at least 15% of frame
) -> Optional[tuple[int, int, int, int]]:
    """
    Get the optimal crop region for the dominant face at a given timestamp.
    
    This is used for split-screen rendering where we need to:
    1. Find the dominant face at the given time
    2. Create a crop region that includes the face with padding
    3. Maintain the target aspect ratio
    
    Args:
        tracking_result: TrackingResult from track_faces_in_video
        timestamp_ms: Timestamp in milliseconds
        target_aspect_ratio: Target width/height ratio for the crop
        padding_ratio: Extra padding around face (0.3 = 30%)
        min_size_ratio: Minimum crop size as ratio of frame
        
    Returns:
        Crop region as (x, y, width, height) or None if no face
    """
    if not tracking_result.frame_results:
        return None
    
    frame_width = tracking_result.frame_width
    frame_height = tracking_result.frame_height
    
    # Find the frame closest to the timestamp
    # Assuming ~10 FPS sampling
    fps_estimate = len(tracking_result.frame_results) / (
        (tracking_result.frame_results[-1].timestamp_ms - tracking_result.frame_results[0].timestamp_ms) / 1000
    ) if len(tracking_result.frame_results) > 1 else 10
    
    target_frame_idx = int((timestamp_ms - tracking_result.frame_results[0].timestamp_ms) * fps_estimate / 1000)
    target_frame_idx = max(0, min(target_frame_idx, len(tracking_result.frame_results) - 1))
    
    frame_result = tracking_result.frame_results[target_frame_idx]
    
    if not frame_result.dominant_face_bbox:
        return None
    
    face_x, face_y, face_w, face_h = frame_result.dominant_face_bbox
    
    # Add padding
    padded_w = int(face_w * (1 + padding_ratio))
    padded_h = int(face_h * (1 + padding_ratio))
    
    # Ensure minimum size
    min_size = int(min(frame_width, frame_height) * min_size_ratio)
    padded_w = max(padded_w, min_size)
    padded_h = max(padded_h, min_size)
    
    # Calculate crop region maintaining aspect ratio
    if target_aspect_ratio > 0:
        # Adjust dimensions to match aspect ratio
        current_ratio = padded_w / padded_h if padded_h > 0 else 1
        
        if current_ratio > target_aspect_ratio:
            # Too wide, increase height
            padded_h = int(padded_w / target_aspect_ratio)
        else:
            # Too tall, increase width
            padded_w = int(padded_h * target_aspect_ratio)
    
    # Center crop on face
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2
    
    crop_x = face_center_x - padded_w // 2
    crop_y = face_center_y - padded_h // 2
    
    # Clamp to frame bounds
    crop_x = max(0, min(crop_x, frame_width - padded_w))
    crop_y = max(0, min(crop_y, frame_height - padded_h))
    
    # Ensure crop doesn't exceed frame
    if crop_x + padded_w > frame_width:
        padded_w = frame_width - crop_x
    if crop_y + padded_h > frame_height:
        padded_h = frame_height - crop_y
    
    return (crop_x, crop_y, padded_w, padded_h)
