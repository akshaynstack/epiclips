"""
Smart Layout Detector - Frame-by-frame layout analysis for dynamic layout switching.

This service analyzes video clips to detect layout types and transitions:
1. Samples frames throughout the clip
2. Classifies each frame as talking_head or screen_share
3. Detects transitions between layouts
4. Merges short segments to avoid flickering
5. Returns a timeline with layout segments for segment-based rendering

Similar to epiriumaiclips' smart_layout.py functionality.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Minimum segment duration in milliseconds to avoid flickering
MIN_SEGMENT_DURATION_MS = 2000  # 2 seconds

# Layout classification thresholds
FACE_AREA_THRESHOLD_TALKING_HEAD = 0.12  # Face > 12% of frame = talking head
FACE_AREA_THRESHOLD_SCREEN_SHARE = 0.08  # Face < 8% of frame = likely screen share
EDGE_DENSITY_THRESHOLD = 0.06  # High edge density suggests UI/text (screen share)
CORNER_THRESHOLD = 0.35  # Face in outer 35% considered "corner"


@dataclass
class LayoutSegment:
    """A segment of video with a consistent layout type."""

    start_ms: int
    end_ms: int
    layout_type: str  # "talking_head" or "screen_share"
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class FrameLayoutInfo:
    """Layout classification for a single frame."""

    timestamp_ms: int
    layout_type: str
    confidence: float
    face_area_ratio: float = 0.0
    edge_density: float = 0.0
    face_in_corner: bool = False


@dataclass
class LayoutAnalysis:
    """Complete layout analysis for a video clip."""

    has_transitions: bool
    dominant_layout: str
    layout_segments: list[LayoutSegment] = field(default_factory=list)
    transition_timestamps: list[int] = field(default_factory=list)
    frame_analyses: list[FrameLayoutInfo] = field(default_factory=list)

    @property
    def segment_count(self) -> int:
        return len(self.layout_segments)


class SmartLayoutDetector:
    """
    Detects layout types and transitions within video clips.

    Enables dynamic layout switching by:
    1. Analyzing frames at regular intervals
    2. Classifying each frame as talking_head or screen_share
    3. Detecting transition points where layout changes
    4. Generating layout segments for per-segment rendering
    """

    def __init__(self):
        self._sample_fps = 3.0  # Sample at 3 FPS for layout analysis

    async def analyze_clip_layout(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        face_detections: Optional[list[dict]] = None,
        sample_fps: float = 3.0,
    ) -> LayoutAnalysis:
        """
        Analyze video clip to detect layout types and transitions.

        Args:
            video_path: Path to the video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            face_detections: Optional pre-computed face detections from detection pipeline
            sample_fps: Frames per second to sample for analysis

        Returns:
            LayoutAnalysis with segments and transition info
        """
        self._sample_fps = sample_fps

        logger.info(
            f"Analyzing layout for clip: {start_ms}ms - {end_ms}ms "
            f"(duration: {(end_ms - start_ms) / 1000:.1f}s)"
        )

        # Run frame-by-frame analysis
        frame_analyses = await self._analyze_frames(
            video_path, start_ms, end_ms, face_detections
        )

        if not frame_analyses:
            # Fallback to single talking_head segment
            logger.warning("No frame analyses available, defaulting to talking_head")
            return LayoutAnalysis(
                has_transitions=False,
                dominant_layout="talking_head",
                layout_segments=[
                    LayoutSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        layout_type="talking_head",
                        confidence=0.5,
                    )
                ],
            )

        # Detect transitions and create segments
        raw_segments = self._create_segments_from_analyses(frame_analyses, start_ms, end_ms)

        # Merge short segments to avoid flickering
        merged_segments = self._merge_short_segments(raw_segments)

        # Determine dominant layout
        dominant_layout = self._calculate_dominant_layout(merged_segments)

        # Extract transition timestamps
        transitions = []
        for i in range(1, len(merged_segments)):
            transitions.append(merged_segments[i].start_ms)

        has_transitions = len(merged_segments) > 1

        logger.info(
            f"Layout analysis complete: {len(merged_segments)} segments, "
            f"transitions={has_transitions}, dominant={dominant_layout}"
        )

        for seg in merged_segments:
            logger.debug(
                f"  Segment: {seg.start_ms}ms - {seg.end_ms}ms "
                f"({seg.duration_ms}ms) = {seg.layout_type}"
            )

        return LayoutAnalysis(
            has_transitions=has_transitions,
            dominant_layout=dominant_layout,
            layout_segments=merged_segments,
            transition_timestamps=transitions,
            frame_analyses=frame_analyses,
        )

    async def _analyze_frames(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        face_detections: Optional[list[dict]] = None,
    ) -> list[FrameLayoutInfo]:
        """
        Analyze frames throughout the clip for layout classification.

        Uses multiple signals:
        - Face detection size and position
        - Edge density (high for UI/text, low for webcam)
        - Color variance
        """
        frame_analyses = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_area = frame_width * frame_height

            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            sample_interval = 1.0 / self._sample_fps

            current_time = start_sec

            # Build face detection lookup by timestamp if provided
            face_lookup = {}
            if face_detections:
                for frame_data in face_detections:
                    ts = int(frame_data.get("timestamp_sec", 0) * 1000)
                    face_lookup[ts] = frame_data.get("detections", [])

            while current_time < end_sec:
                timestamp_ms = int(current_time * 1000)
                frame_pos = int(current_time * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if not ret:
                    current_time += sample_interval
                    continue

                # Get face detections for this frame
                frame_faces = self._get_faces_for_timestamp(
                    timestamp_ms, face_lookup, face_detections
                )

                # Analyze frame
                layout_info = self._classify_frame(
                    frame=frame,
                    timestamp_ms=timestamp_ms,
                    face_detections=frame_faces,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    frame_area=frame_area,
                )

                frame_analyses.append(layout_info)
                current_time += sample_interval

            cap.release()

            logger.debug(f"Analyzed {len(frame_analyses)} frames for layout")
            return frame_analyses

        except Exception as e:
            logger.error(f"Frame analysis failed: {e}", exc_info=True)
            return []

    def _get_faces_for_timestamp(
        self,
        timestamp_ms: int,
        face_lookup: dict,
        face_detections: Optional[list[dict]],
    ) -> list[dict]:
        """Get face detections closest to the given timestamp."""
        if not face_detections:
            return []

        # Try exact match first
        if timestamp_ms in face_lookup:
            return face_lookup[timestamp_ms]

        # Find closest timestamp within 500ms
        closest_ts = None
        closest_diff = float('inf')

        for ts in face_lookup.keys():
            diff = abs(ts - timestamp_ms)
            if diff < closest_diff and diff < 500:
                closest_diff = diff
                closest_ts = ts

        if closest_ts is not None:
            return face_lookup[closest_ts]

        return []

    def _classify_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        face_detections: list[dict],
        frame_width: int,
        frame_height: int,
        frame_area: int,
    ) -> FrameLayoutInfo:
        """
        Classify a single frame as talking_head or screen_share.

        Classification logic:
        - talking_head: Large face (>12% of frame), usually centered
        - screen_share: Small face in corner (<8%), high edge density, or no face
        """
        # Calculate edge density
        edge_density = self._calculate_edge_density(frame)

        # Analyze face detections
        face_area_ratio = 0.0
        face_in_corner = False
        largest_face_area = 0

        if face_detections:
            for face in face_detections:
                bbox = face.get("bbox", {})
                face_w = bbox.get("width", 0)
                face_h = bbox.get("height", 0)
                face_area_px = face_w * face_h

                if face_area_px > largest_face_area:
                    largest_face_area = face_area_px
                    face_area_ratio = face_area_px / frame_area if frame_area > 0 else 0

                    # Check if face is in corner
                    center_x = bbox.get("x", 0) + face_w / 2
                    center_y = bbox.get("y", 0) + face_h / 2
                    rel_x = center_x / frame_width
                    rel_y = center_y / frame_height

                    # Corner = outer 35% of frame
                    in_x_corner = rel_x < CORNER_THRESHOLD or rel_x > (1 - CORNER_THRESHOLD)
                    in_y_corner = rel_y < CORNER_THRESHOLD or rel_y > (1 - CORNER_THRESHOLD)
                    face_in_corner = in_x_corner and in_y_corner

        # Classification logic
        layout_type, confidence = self._determine_layout_type(
            face_area_ratio=face_area_ratio,
            face_in_corner=face_in_corner,
            edge_density=edge_density,
            has_face=len(face_detections) > 0,
        )

        return FrameLayoutInfo(
            timestamp_ms=timestamp_ms,
            layout_type=layout_type,
            confidence=confidence,
            face_area_ratio=face_area_ratio,
            edge_density=edge_density,
            face_in_corner=face_in_corner,
        )

    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density of frame (high = UI/text content)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            height, width = frame.shape[:2]
            edge_density = np.sum(edges > 0) / (width * height)
            return edge_density
        except Exception:
            return 0.0

    def _determine_layout_type(
        self,
        face_area_ratio: float,
        face_in_corner: bool,
        edge_density: float,
        has_face: bool,
    ) -> tuple[str, float]:
        """
        Determine layout type based on analysis signals.

        Returns:
            Tuple of (layout_type, confidence)
        """
        # Strong talking head indicators
        if face_area_ratio > FACE_AREA_THRESHOLD_TALKING_HEAD and not face_in_corner:
            return ("talking_head", 0.95)

        # Strong screen share indicators
        if face_in_corner and face_area_ratio < FACE_AREA_THRESHOLD_SCREEN_SHARE:
            return ("screen_share", 0.90)

        if not has_face and edge_density > EDGE_DENSITY_THRESHOLD:
            # No face + high edge density = pure screen share
            return ("screen_share", 0.85)

        # Medium face, not in corner - likely talking head with some screen content
        if face_area_ratio > FACE_AREA_THRESHOLD_SCREEN_SHARE and not face_in_corner:
            return ("talking_head", 0.70)

        # Small face, not clearly in corner, with high edge density
        if edge_density > EDGE_DENSITY_THRESHOLD and face_area_ratio < FACE_AREA_THRESHOLD_TALKING_HEAD:
            return ("screen_share", 0.65)

        # Default to talking head with low confidence
        return ("talking_head", 0.50)

    def _create_segments_from_analyses(
        self,
        frame_analyses: list[FrameLayoutInfo],
        start_ms: int,
        end_ms: int,
    ) -> list[LayoutSegment]:
        """Create raw segments from frame analyses."""
        if not frame_analyses:
            return [
                LayoutSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    layout_type="talking_head",
                    confidence=0.5,
                )
            ]

        segments = []
        current_layout = frame_analyses[0].layout_type
        segment_start = start_ms
        confidences = [frame_analyses[0].confidence]

        for i in range(1, len(frame_analyses)):
            frame_info = frame_analyses[i]

            if frame_info.layout_type != current_layout:
                # Layout changed - create segment
                segment_end = frame_info.timestamp_ms
                avg_confidence = sum(confidences) / len(confidences)

                segments.append(LayoutSegment(
                    start_ms=segment_start,
                    end_ms=segment_end,
                    layout_type=current_layout,
                    confidence=avg_confidence,
                ))

                # Start new segment
                current_layout = frame_info.layout_type
                segment_start = segment_end
                confidences = [frame_info.confidence]
            else:
                confidences.append(frame_info.confidence)

        # Add final segment
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        segments.append(LayoutSegment(
            start_ms=segment_start,
            end_ms=end_ms,
            layout_type=current_layout,
            confidence=avg_confidence,
        ))

        return segments

    def _merge_short_segments(
        self,
        segments: list[LayoutSegment],
    ) -> list[LayoutSegment]:
        """
        Merge segments shorter than MIN_SEGMENT_DURATION_MS to avoid flickering.

        Short segments are merged into adjacent segments based on confidence.
        """
        if len(segments) <= 1:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]

            if current.duration_ms < MIN_SEGMENT_DURATION_MS:
                # This segment is too short - merge with neighbor
                if i == 0 and len(segments) > 1:
                    # Merge with next segment
                    next_seg = segments[i + 1]
                    merged_seg = LayoutSegment(
                        start_ms=current.start_ms,
                        end_ms=next_seg.end_ms,
                        layout_type=next_seg.layout_type,  # Use next segment's type
                        confidence=next_seg.confidence,
                    )
                    merged.append(merged_seg)
                    i += 2
                elif merged:
                    # Merge with previous segment
                    prev_seg = merged[-1]
                    merged[-1] = LayoutSegment(
                        start_ms=prev_seg.start_ms,
                        end_ms=current.end_ms,
                        layout_type=prev_seg.layout_type,
                        confidence=prev_seg.confidence,
                    )
                    i += 1
                else:
                    # Keep it if we have no choice
                    merged.append(current)
                    i += 1
            else:
                # Segment is long enough
                if merged and merged[-1].layout_type == current.layout_type:
                    # Merge with previous if same type
                    prev_seg = merged[-1]
                    merged[-1] = LayoutSegment(
                        start_ms=prev_seg.start_ms,
                        end_ms=current.end_ms,
                        layout_type=current.layout_type,
                        confidence=(prev_seg.confidence + current.confidence) / 2,
                    )
                else:
                    merged.append(current)
                i += 1

        # Second pass: merge consecutive segments of same type
        final = []
        for seg in merged:
            if final and final[-1].layout_type == seg.layout_type:
                prev = final[-1]
                final[-1] = LayoutSegment(
                    start_ms=prev.start_ms,
                    end_ms=seg.end_ms,
                    layout_type=seg.layout_type,
                    confidence=(prev.confidence + seg.confidence) / 2,
                )
            else:
                final.append(seg)

        return final if final else segments

    def _calculate_dominant_layout(
        self,
        segments: list[LayoutSegment],
    ) -> str:
        """Calculate the dominant layout type by total duration."""
        if not segments:
            return "talking_head"

        talking_head_duration = 0
        screen_share_duration = 0

        for seg in segments:
            if seg.layout_type == "talking_head":
                talking_head_duration += seg.duration_ms
            else:
                screen_share_duration += seg.duration_ms

        return "talking_head" if talking_head_duration >= screen_share_duration else "screen_share"

    def analyze_clip_layout_sync(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        face_detections: Optional[list[dict]] = None,
        sample_fps: float = 3.0,
    ) -> LayoutAnalysis:
        """Synchronous version for non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.analyze_clip_layout(
                video_path, start_ms, end_ms, face_detections, sample_fps
            )
        )
