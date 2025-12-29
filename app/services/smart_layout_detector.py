"""
Smart Layout Detector - Frame-by-frame layout analysis for dynamic layout switching.

This service analyzes video clips to detect layout types and transitions:
1. Samples frames throughout the clip
2. Classifies each frame as talking_head or screen_share
3. Detects transitions between layouts
4. Merges short segments to avoid flickering
5. Returns a timeline with layout segments for segment-based rendering
6. NEW: Uses FaceTracker for robust dominant face tracking across frames

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
# Reduced from 2s to 1.5s to allow faster layout transitions
MIN_SEGMENT_DURATION_MS = 1500  # 1.5 seconds

# Layout classification thresholds (simplified for robustness)
# Key insight: If face exists AND edge density is high -> screen_share (split screen)
# If face is large and centered with low edge density -> talking_head
FACE_AREA_THRESHOLD_TALKING_HEAD = 0.15  # Face > 15% of frame = talking head
EDGE_DENSITY_THRESHOLD = 0.04  # Lower threshold - any UI content triggers screen_share
CORNER_THRESHOLD = 0.30  # Face in outer 30% considered "corner"


@dataclass
class LayoutSegment:
    """A segment of video with a consistent layout type."""

    start_ms: int
    end_ms: int
    layout_type: str  # "talking_head" or "screen_share"
    confidence: float = 1.0
    # Corner facecam bounding box (x, y, width, height) - only set for screen_share
    corner_facecam_bbox: Optional[tuple[int, int, int, int]] = None

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
    has_embedded_facecam: bool = False  # True if face is already embedded in screen content
    
    # Corner facecam position (x, y, width, height) - average across frames
    # Used for split screen rendering to crop the corner webcam, not the main face
    corner_facecam_bbox: Optional[tuple[int, int, int, int]] = None
    
    # VLM-detected webcam position (e.g., "bottom-right")
    webcam_position: Optional[str] = None
    
    # Speaker-driven layout data (new)
    speaker_analysis: Optional[any] = None  # SpeakerAnalysis from speaker_tracker
    active_speaker_count: int = 0  # Number of active (non-background) speakers
    layout_driven_by_speakers: bool = False  # True if layout was determined by speaker analysis
    
    # For podcast layout: list of 2 face bboxes for side-by-side rendering
    podcast_face_bboxes: Optional[list[tuple[int, int, int, int]]] = None

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
    
    NEW: Speaker-driven layout mode
    - Uses active speaker detection to determine layout
    - SINGLE layout when only one speaker is active
    - SPLIT layout only when multiple speakers are active
    - Background faces (< 2% speaking time) don't affect layout
    
    NEW: VLM-first detection mode (recommended)
    - Uses Gemini Flash Vision as PRIMARY detection method
    - Much more accurate at detecting webcam overlays vs main subjects
    - Falls back to MediaPipe/Haar if VLM is unavailable
    """

    def __init__(self, use_speaker_driven_layout: bool = True, use_vlm_detection: bool = True):
        self._sample_fps = 3.0  # Sample at 3 FPS for layout analysis
        self._use_speaker_driven_layout = use_speaker_driven_layout
        self._use_vlm_detection = use_vlm_detection
        self._speaker_tracker = None
        self._vlm_detector = None
        
        if use_speaker_driven_layout:
            self._init_speaker_tracker()
        
        if use_vlm_detection:
            self._init_vlm_detector()
    
    def _init_vlm_detector(self):
        """Initialize VLM layout detector for accurate webcam overlay detection."""
        try:
            from app.services.vlm_layout_detector import get_vlm_detector
            self._vlm_detector = get_vlm_detector()
            logger.info("VLM layout detector initialized (Gemini Flash Vision)")
        except ImportError as e:
            logger.warning(f"VLM layout detector not available: {e}")
            self._vlm_detector = None
        except Exception as e:
            logger.warning(f"Failed to initialize VLM layout detector: {e}")
            self._vlm_detector = None
    
    def _init_speaker_tracker(self):
        """Initialize speaker tracker for speaker-driven layout."""
        try:
            from app.services.speaker_tracker import SpeakerTracker
            self._speaker_tracker = SpeakerTracker()
            logger.info("Speaker tracker initialized for speaker-driven layout")
        except ImportError as e:
            logger.warning(f"Speaker tracker not available: {e}")
            self._speaker_tracker = None
        except Exception as e:
            logger.warning(f"Failed to initialize speaker tracker: {e}")
            self._speaker_tracker = None

    async def analyze_clip_layout_with_speakers(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        face_detections: Optional[list[dict]] = None,
        transcript_segments: Optional[list] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        sample_fps: float = 3.0,
        debug_output_path: Optional[str] = None,
    ) -> LayoutAnalysis:
        """
        Analyze video clip for dynamic layout detection with transitions.
        
        DETECTION PRIORITY:
        1. VLM (Gemini Flash Vision) - Most accurate, analyzes 1-3 frames
        2. MediaPipe face detection - Fallback if VLM unavailable
        3. Haar cascade - Final fallback
        
        This method performs FRAME-BY-FRAME analysis to detect layout transitions
        within a clip. It identifies when the video switches between:
        - screen_share: Screen content visible (high edge density in top region)
        - talking_head: Full-screen person talking (low edge density, face visible)
        
        Key behavior:
        - Analyzes each frame for screen content presence
        - Creates segments for each layout type
        - Enables segment-based rendering with stitching
        
        Args:
            video_path: Path to the video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            face_detections: Pre-computed face detections
            transcript_segments: Transcript segments with timing
            frame_width: Video frame width
            frame_height: Video frame height
            sample_fps: Frames per second to sample
            debug_output_path: Optional path to save debug JSON
            
        Returns:
            LayoutAnalysis with layout segments and transitions
        """
        logger.info(
            f"Analyzing layout: {start_ms}ms - {end_ms}ms "
            f"(duration: {(end_ms - start_ms) / 1000:.1f}s), VLM={self._vlm_detector is not None}"
        )
        
        # ============================================================
        # STEP 1: Try VLM-based detection FIRST (most accurate)
        # ============================================================
        if self._vlm_detector is not None:
            try:
                vlm_result = await self._vlm_detector.analyze_clip_layout(
                    video_path=video_path,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    sample_count=None,  # Auto-calculate based on duration
                )
                
                # If VLM detection is confident (>0.7), use it directly
                if vlm_result.confidence >= 0.7:
                    logger.info(
                        f"VLM detection confident: layout={vlm_result.layout_type}, "
                        f"has_webcam={vlm_result.has_corner_webcam}, "
                        f"position={vlm_result.webcam_position}, "
                        f"confidence={vlm_result.confidence:.2f}, "
                        f"has_transitions={vlm_result.has_transitions}"
                    )
                    
                    # CHECK FOR TRANSITIONS: If VLM detected mixed layouts, create segments
                    if vlm_result.has_transitions and vlm_result.frame_results:
                        segments = self._create_segments_from_vlm_frames(
                            vlm_result.frame_results,
                            start_ms,
                            end_ms,
                            vlm_result.webcam_bbox_estimate,
                            frame_width,
                            frame_height,
                        )
                        
                        if len(segments) > 1:
                            # Merge short segments (< 2 seconds)
                            merged_segments = self._merge_short_segments(segments)
                            has_transitions = len(merged_segments) > 1
                            
                            logger.info(
                                f"VLM detected {len(merged_segments)} layout segments with transitions"
                            )
                            for i, seg in enumerate(merged_segments):
                                logger.info(
                                    f"  Segment {i+1}: {seg.start_ms}ms - {seg.end_ms}ms = {seg.layout_type}"
                                )
                            
                            return LayoutAnalysis(
                                has_transitions=has_transitions,
                                dominant_layout=vlm_result.layout_type,
                                layout_segments=merged_segments,
                                transition_timestamps=[seg.start_ms for seg in merged_segments[1:]],
                                frame_analyses=[],
                                has_embedded_facecam=vlm_result.has_corner_webcam,
                                corner_facecam_bbox=vlm_result.webcam_bbox_estimate,
                                webcam_position=vlm_result.webcam_position,
                                speaker_analysis=None,
                                active_speaker_count=1,
                                layout_driven_by_speakers=False,
                            )
                    
                    # No transitions - create single segment from VLM result
                    segment = LayoutSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        layout_type=vlm_result.layout_type,
                        confidence=vlm_result.confidence,
                        corner_facecam_bbox=vlm_result.webcam_bbox_estimate,
                    )
                    
                    return LayoutAnalysis(
                        has_transitions=False,
                        dominant_layout=vlm_result.layout_type,
                        layout_segments=[segment],
                        transition_timestamps=[],
                        frame_analyses=[],
                        has_embedded_facecam=vlm_result.has_corner_webcam,
                        corner_facecam_bbox=vlm_result.webcam_bbox_estimate,
                        webcam_position=vlm_result.webcam_position,
                        speaker_analysis=None,
                        active_speaker_count=1,
                        layout_driven_by_speakers=False,
                    )
                else:
                    logger.info(
                        f"VLM detection low confidence ({vlm_result.confidence:.2f}), "
                        f"falling back to MediaPipe"
                    )
            except Exception as e:
                logger.warning(f"VLM detection failed: {e}, falling back to MediaPipe")
        
        # ============================================================
        # STEP 2: Fall back to MediaPipe/Haar-based frame analysis
        # ============================================================
        # Perform frame-by-frame analysis to detect layout per frame
        # Pass face_detections from detection pipeline for better accuracy
        frame_layouts = await self._analyze_frames_for_layout(
            video_path, start_ms, end_ms, sample_fps, face_detections, frame_width, frame_height
        )
        
        if not frame_layouts:
            # Fallback: check if screen content exists at all
            has_screen_content = await self._detect_screen_content(
                video_path, start_ms, end_ms
            )
            layout_type = "screen_share" if has_screen_content else "talking_head"
            logger.warning(f"No frame analysis available, defaulting to {layout_type}")
            return LayoutAnalysis(
                has_transitions=False,
                dominant_layout=layout_type,
                layout_segments=[
                    LayoutSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        layout_type=layout_type,
                        confidence=0.6,
                    )
                ],
                transition_timestamps=[],
                frame_analyses=[],
                has_embedded_facecam=False,
                speaker_analysis=None,
                active_speaker_count=1,
                layout_driven_by_speakers=False,
            )
        
        # Create segments from frame-by-frame analysis
        raw_segments = self._create_layout_segments(frame_layouts, start_ms, end_ms)
        
        # Merge short segments to avoid flickering (min 2 seconds)
        merged_segments = self._merge_short_segments(raw_segments)
        
        # Calculate dominant layout
        dominant_layout = self._calculate_dominant_layout(merged_segments)
        
        # Get corner facecam bbox from the analysis (stored in instance variable)
        corner_facecam_bbox = getattr(self, '_last_corner_facecam_bbox', None)
        
        # Add corner facecam bbox to ALL screen_share segments for rendering
        # If corner_facecam_bbox exists, it means we detected a corner webcam
        if corner_facecam_bbox:
            for seg in merged_segments:
                if seg.layout_type == "screen_share":
                    seg.corner_facecam_bbox = corner_facecam_bbox
        
        # If all segments are screen_share and we have corner_facecam_bbox,
        # ensure it's set on all of them (for majority voting case)
        all_screen_share = all(seg.layout_type == "screen_share" for seg in merged_segments)
        if all_screen_share and corner_facecam_bbox:
            for seg in merged_segments:
                seg.corner_facecam_bbox = corner_facecam_bbox
        
        has_transitions = len(merged_segments) > 1
        transitions = [seg.start_ms for seg in merged_segments[1:]]
        
        # Log segment details
        logger.info(
            f"Layout analysis complete: {len(merged_segments)} segments, "
            f"transitions={has_transitions}, dominant={dominant_layout}, "
            f"corner_facecam_bbox={corner_facecam_bbox}"
        )
        for i, seg in enumerate(merged_segments):
            logger.info(
                f"  Segment {i+1}: {seg.start_ms}ms - {seg.end_ms}ms "
                f"({seg.duration_ms}ms) = {seg.layout_type}"
            )
        
        return LayoutAnalysis(
            has_transitions=has_transitions,
            dominant_layout=dominant_layout,
            layout_segments=merged_segments,
            transition_timestamps=transitions,
            frame_analyses=[],
            has_embedded_facecam=False,
            corner_facecam_bbox=corner_facecam_bbox,
            speaker_analysis=None,
            active_speaker_count=1,
            layout_driven_by_speakers=False,
        )

    def _detect_face_in_bottom_region(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, bool]:
        """
        Detect face in the bottom/camera region of the frame using Haar cascade.
        
        Returns:
            Tuple of (face_area_ratio, is_in_corner):
            - face_area_ratio: 0.0 to 1.0 - area of largest face / frame area
            - is_in_corner: True if face is in bottom corner (small facecam)
        """
        # Delegate to the new corner detection method
        result = self._detect_corner_facecam(frame, frame_width, frame_height)
        return (result[0], result[1])  # Return only first two values for backward compatibility

    def _detect_corner_facecam(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, bool, Optional[tuple[int, int, int, int]]]:
        """
        Detect if there's a SMALL webcam overlay in the corners/edges of the frame.
        
        This detects OVERLAY FACECAMS only, not main subjects:
        - Corners: top-left, top-right, bottom-left, bottom-right (outer 25%)
        - Edges: middle-left, middle-right (outer 20% horizontally)
        
        Key distinction:
        - CORNER FACECAM: Small (1-8% of frame), positioned in corners/edges
        - MAIN SUBJECT: Large face in center = NOT a corner facecam
        
        Returns:
            Tuple of (face_area_ratio, is_in_corner, bbox):
            - face_area_ratio: 0.0 to 1.0 - area of largest face / frame area
            - is_in_corner: True if SMALL face is in corner/edge region
            - bbox: (x, y, w, h) of the corner facecam, or None if not found
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use Haar cascade for quick face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Increase minNeighbors to reduce false positives on graphics
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6,
                minSize=(50, 50)  # Small enough to catch webcam overlays
            )
            
            if len(faces) == 0:
                return (0.0, False, None)
            
            # Define CORNER regions (outer 25% on BOTH axes for true corners)
            CORNER_THRESHOLD = 0.25  # Stricter - only outer 25%
            left_edge = frame_width * CORNER_THRESHOLD
            right_edge = frame_width * (1 - CORNER_THRESHOLD)
            top_edge = frame_height * CORNER_THRESHOLD
            bottom_edge = frame_height * (1 - CORNER_THRESHOLD)
            
            # Define EDGE regions (for middle-left, middle-right facecams)
            # These are faces on the far left/right but in middle height
            EDGE_THRESHOLD = 0.20  # Outer 20% horizontally
            edge_left = frame_width * EDGE_THRESHOLD
            edge_right = frame_width * (1 - EDGE_THRESHOLD)
            
            frame_area = frame_width * frame_height
            
            # Maximum face size for a corner/edge facecam
            # Webcam overlays are typically 0.3-8% of frame area
            # Larger faces are main subjects, not overlays
            MAX_CORNER_FACE_AREA_RATIO = 0.08  # 8% max
            MIN_CORNER_FACE_AREA_RATIO = 0.003  # 0.3% min (lowered from 0.5% to catch smaller webcams)
            
            # Find the largest VALID face that's in a corner/edge
            largest_corner_face_area = 0
            largest_corner_face_bbox: Optional[tuple[int, int, int, int]] = None
            has_corner_face = False
            
            for (x, y, w, h) in faces:
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                face_area = w * h
                face_area_ratio = face_area / frame_area if frame_area > 0 else 0
                
                # FIRST CHECK: Is the face small enough to be a webcam overlay?
                # Large faces (>8% of frame) are main subjects, not overlays
                if face_area_ratio > MAX_CORNER_FACE_AREA_RATIO:
                    logger.debug(f"Rejected face: too large ({face_area_ratio:.1%} > {MAX_CORNER_FACE_AREA_RATIO:.1%}) - likely main subject")
                    continue
                if face_area_ratio < MIN_CORNER_FACE_AREA_RATIO:
                    logger.debug(f"Rejected face: too small ({face_area_ratio:.2%} < {MIN_CORNER_FACE_AREA_RATIO:.2%})")
                    continue
                
                # Check if face is in CORNER (outer 25% on both axes)
                in_left = face_center_x < left_edge
                in_right = face_center_x > right_edge
                in_top = face_center_y < top_edge
                in_bottom = face_center_y > bottom_edge
                
                # True corner = (left OR right) AND (top OR bottom)
                is_in_true_corner = (in_left or in_right) and (in_top or in_bottom)
                
                # Check if face is on far left/right EDGE (middle height)
                # This catches facecams positioned on the sides but not in corners
                in_far_left = face_center_x < edge_left
                in_far_right = face_center_x > edge_right
                is_on_edge = in_far_left or in_far_right
                
                # Face must be in corner OR on edge (and be small enough)
                is_in_corner = is_in_true_corner or is_on_edge
                
                if not is_in_corner:
                    continue
                
                # VALIDATION: Check for natural skin tones to reject graphics/icons
                # Extract the face region
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                
                # Convert to HSV for skin detection
                hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                
                # Skin tone range in HSV (covers various skin colors)
                # H: 0-25 (red/orange/yellow hues)
                # S: 20-255 (some saturation, not too gray)
                # V: 50-255 (not too dark)
                lower_skin = np.array([0, 20, 50], dtype=np.uint8)
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                
                # Alternative skin range for darker skin tones
                lower_skin2 = np.array([0, 10, 40], dtype=np.uint8)
                upper_skin2 = np.array([20, 150, 255], dtype=np.uint8)
                
                # Create skin masks
                mask1 = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                mask2 = cv2.inRange(hsv_roi, lower_skin2, upper_skin2)
                skin_mask = cv2.bitwise_or(mask1, mask2)
                
                # Calculate skin percentage
                skin_pixels = cv2.countNonZero(skin_mask)
                total_pixels = face_roi.shape[0] * face_roi.shape[1]
                skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
                
                # Real faces typically have 20-80% skin pixels visible
                # Graphics/icons usually have < 10% or > 90% (solid colors)
                MIN_SKIN_RATIO = 0.15  # At least 15% skin-like pixels
                MAX_SKIN_RATIO = 0.85  # Not more than 85% (avoid solid skin-colored graphics)
                
                if skin_ratio < MIN_SKIN_RATIO or skin_ratio > MAX_SKIN_RATIO:
                    # Likely a graphic, not a real face
                    continue
                
                # Additional check: face aspect ratio should be reasonable (0.7 - 1.3)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                    continue
                
                # Additional check: Real webcams have natural color variance
                # Graphics/mockups tend to have very uniform or very high contrast colors
                # Check standard deviation of color channels
                b, g, r = cv2.split(face_roi)
                color_std = (np.std(b) + np.std(g) + np.std(r)) / 3
                
                # Real faces have moderate color variance (std dev 20-80)
                # Flat graphics have very low variance, high contrast graphics have very high
                MIN_COLOR_STD = 15
                MAX_COLOR_STD = 100
                
                if color_std < MIN_COLOR_STD or color_std > MAX_COLOR_STD:
                    # Likely a graphic with unusual color distribution
                    logger.debug(f"Rejected face: color_std={color_std:.1f} out of range [{MIN_COLOR_STD}, {MAX_COLOR_STD}]")
                    continue
                
                # Check for sharp edges within the face region (real faces are smooth)
                # Graphics/mockups often have sharp internal edges
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                edges_roi = cv2.Canny(gray_roi, 100, 200)
                edge_ratio = cv2.countNonZero(edges_roi) / total_pixels if total_pixels > 0 else 0
                
                # Real faces have low edge density inside the face region
                # Graphics with internal details have high edge density
                MAX_INTERNAL_EDGE_RATIO = 0.15  # Less than 15% edges inside face
                
                if edge_ratio > MAX_INTERNAL_EDGE_RATIO:
                    logger.debug(f"Rejected face: internal edge_ratio={edge_ratio:.3f} > {MAX_INTERNAL_EDGE_RATIO}")
                    continue
                
                if face_area > largest_corner_face_area:
                    largest_corner_face_area = face_area
                    largest_corner_face_bbox = (x, y, w, h)
                    has_corner_face = True
            
            face_area_ratio = largest_corner_face_area / frame_area if frame_area > 0 else 0.0
            
            return (face_area_ratio, has_corner_face, largest_corner_face_bbox if has_corner_face else None)
            
        except Exception as e:
            logger.debug(f"Corner facecam detection error: {e}")
            return (0.0, False, None)

    def _detect_split_line(self, frame: np.ndarray) -> bool:
        """
        Detect if there is a strong horizontal or vertical split line in the frame.
        Uses Hough Transform to find perfectly straight lines, avoiding hand-drawn lines.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h, w = frame.shape[:2]
            
            # Use Probabilistic Hough Transform
            # minLineLength: 30% of the smaller dimension
            min_line_len = int(min(h, w) * 0.3)
            # maxLineGap: Allow small gaps (e.g. text crossing the line)
            max_line_gap = int(min(h, w) * 0.05)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=min_line_len, maxLineGap=max_line_gap)
            
            if lines is None:
                return False
                
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Check for horizontal (near 0 or 180)
                is_horizontal = abs(angle) < 2 or abs(abs(angle) - 180) < 2
                
                # Check for vertical (near 90)
                is_vertical = abs(abs(angle) - 90) < 2
                
                if is_horizontal:
                    # Check if in central region (20-80% height)
                    y_avg = (y1 + y2) / 2
                    if 0.2 * h < y_avg < 0.8 * h:
                        return True
                        
                if is_vertical:
                    # Check if in central region (20-80% width)
                    x_avg = (x1 + x2) / 2
                    if 0.2 * w < x_avg < 0.8 * w:
                        return True
                        
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting split line: {e}")
            return False

    def _detect_dark_ui_content(self, frame: np.ndarray) -> bool:
        """
        Detect dark UI content (dark mode interfaces, dark code editors, tables on dark bg).
        
        Dark UI often has:
        - Low average brightness in top region
        - High contrast between text/elements and background
        - Structured patterns (horizontal lines from tables, vertical from menus)
        
        Returns True if dark UI content detected in top region.
        """
        try:
            height = frame.shape[0]
            # Analyze top 60% for dark UI
            top_region = frame[:int(height * 0.6), :]
            
            # Convert to grayscale
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            
            # Check average brightness - dark UI has low brightness
            avg_brightness = np.mean(gray)
            
            # Check contrast (std deviation) - dark UI has high contrast
            brightness_std = np.std(gray)
            
            # Dark UI characteristics:
            # - Average brightness < 80 (dark background)
            # - High standard deviation > 40 (light text on dark bg)
            is_dark_ui = avg_brightness < 80 and brightness_std > 40
            
            # Additional check: detect horizontal lines (table rows)
            if is_dark_ui:
                # Look for horizontal structure patterns
                edges = cv2.Canny(gray, 30, 100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                       minLineLength=int(top_region.shape[1] * 0.2),
                                       maxLineGap=20)
                if lines is not None and len(lines) > 3:
                    # Multiple horizontal lines = table structure
                    horizontal_count = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                        if angle < 5 or angle > 175:
                            horizontal_count += 1
                    if horizontal_count >= 3:
                        return True
            
            return is_dark_ui
            
        except Exception:
            return False

    def _detect_structured_content(self, frame: np.ndarray) -> bool:
        """
        Detect structured content like tables, code, lists, grids.
        
        Structured content has:
        - Regular horizontal lines (table rows, code lines)
        - Regular vertical alignment (columns)
        - Repeating patterns
        
        Returns True if structured content detected in top region.
        """
        try:
            height, width = frame.shape[:2]
            # Analyze top 50% for structured content
            top_region = frame[:int(height * 0.5), :]
            
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal lines (rows in tables, code lines)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                   minLineLength=int(width * 0.15),
                                   maxLineGap=10)
            
            if lines is None:
                return False
            
            # Count horizontal and vertical lines
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:  # Horizontal
                    horizontal_lines += 1
                elif 80 < angle < 100:  # Vertical
                    vertical_lines += 1
            
            # Structured content: multiple horizontal lines (table/code)
            # OR combination of horizontal + vertical (grid/UI)
            has_structure = horizontal_lines >= 5 or (horizontal_lines >= 2 and vertical_lines >= 2)
            
            return has_structure
            
        except Exception:
            return False

    async def _analyze_frames_for_layout(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        sample_fps: float = 2.0,
        face_detections: Optional[list[dict]] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> list[tuple[int, str, float]]:
        """
        Analyze frames to determine layout type per frame.
        
        SIMPLIFIED LOGIC:
        - If CORNER FACECAM detected (small face in corner) -> SPLIT SCREEN
        - If NO corner facecam -> TALKING HEAD (9:16 full frame)
        
        PRIORITY: Uses pre-computed face_detections from MediaPipe when available.
        Falls back to Haar cascade if no face_detections provided.
        
        KEY CONSTRAINT: Corner facecam detection requires:
        1. Face must be VERY small (0.2%-4% of frame - real webcam overlays)
        2. Face must be in TRUE corner (both X and Y edge, not just edge)
        3. Face position must be CONSISTENT across frames (overlay doesn't move)
        """
        frame_layouts = []
        
        # Thresholds for corner facecam detection - TIGHTENED to avoid false positives
        # Real webcam overlays are typically 0.03%-3% of frame area (very small corner webcams)
        CORNER_FACECAM_MAX_AREA = 0.04    # Corner facecam is < 4% of frame (reduced from 8%)
        CORNER_FACECAM_MIN_AREA = 0.0003  # Minimum 0.03% to catch very small webcams (matches face_detector)
        CORNER_THRESHOLD = 0.25  # Outer 25% of frame is "corner" (tightened from 30%)
        # Note: Removed EDGE_THRESHOLD - now require TRUE corner (both X and Y), not just edge
        
        frame_area = frame_width * frame_height
        
        # Corner boundaries - must be in BOTH X and Y corner regions
        left_edge = frame_width * CORNER_THRESHOLD
        right_edge = frame_width * (1 - CORNER_THRESHOLD)
        top_edge = frame_height * CORNER_THRESHOLD
        bottom_edge = frame_height * (1 - CORNER_THRESHOLD)
        
        logger.info(
            f"Corner detection thresholds: frame={frame_width}x{frame_height}, "
            f"corner_x=[0-{left_edge:.0f}|{right_edge:.0f}-{frame_width}], "
            f"corner_y=[0-{top_edge:.0f}|{bottom_edge:.0f}-{frame_height}], "
            f"area_ratio=[{CORNER_FACECAM_MIN_AREA}-{CORNER_FACECAM_MAX_AREA}] (TRUE CORNER REQUIRED)"
        )
        
        # Track corner facecam detections
        corner_facecam_frames = 0
        corner_facecam_bboxes: list[tuple[int, int, int, int]] = []
        total_frames = 0
        debug_signals = []
        
        # USE PRE-COMPUTED FACE DETECTIONS if available (from MediaPipe)
        if face_detections:
            logger.info(f"Using {len(face_detections)} pre-computed face detection frames")
            
            for frame_data in face_detections:
                timestamp_sec = frame_data.get("timestamp_sec", 0)
                timestamp_ms = int(timestamp_sec * 1000)
                
                # Skip frames outside our range
                if timestamp_ms < start_ms or timestamp_ms > end_ms:
                    continue
                
                total_frames += 1
                faces = frame_data.get("detections", [])
                
                # Find corner facecam among detected faces
                corner_bbox = None
                face_area_ratio = 0.0
                has_corner_facecam = False
                rejected_reasons = []  # Track why faces were rejected
                
                for face in faces:
                    bbox = face.get("bbox", {})
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)
                    
                    # Allow smaller faces for corner webcams (reduced from 30x30 to 20x20)
                    if w < 20 or h < 20:
                        rejected_reasons.append(f"too_small({w}x{h})")
                        continue
                    
                    face_center_x = x + w / 2
                    face_center_y = y + h / 2
                    area = w * h
                    area_ratio = area / frame_area if frame_area > 0 else 0
                    
                    # Check size constraints for corner facecam
                    if area_ratio < CORNER_FACECAM_MIN_AREA:
                        rejected_reasons.append(f"area_too_small({area_ratio:.4f}<{CORNER_FACECAM_MIN_AREA})")
                        continue
                    if area_ratio > CORNER_FACECAM_MAX_AREA:
                        rejected_reasons.append(f"area_too_large({area_ratio:.4f}>{CORNER_FACECAM_MAX_AREA})")
                        continue
                    
                    # Check if in TRUE corner (BOTH X and Y must be in corner region)
                    # This prevents false positives from people standing on the side of frame
                    in_left = face_center_x < left_edge
                    in_right = face_center_x > right_edge
                    in_top = face_center_y < top_edge
                    in_bottom = face_center_y > bottom_edge
                    
                    # STRICT: Require TRUE corner - both horizontal AND vertical edge
                    # A webcam overlay is always in a corner (top-left, top-right, bottom-left, bottom-right)
                    # NOT just on the side of the frame
                    is_in_true_corner = (in_left or in_right) and (in_top or in_bottom)
                    
                    # Removed is_on_edge check - this was causing false positives
                    # when a person stands on the right side of frame
                    
                    if is_in_true_corner:
                        has_corner_facecam = True
                        corner_bbox = (x, y, w, h)
                        face_area_ratio = area_ratio
                        break  # Found a corner facecam
                    else:
                        rejected_reasons.append(f"not_in_true_corner(x={face_center_x:.0f},y={face_center_y:.0f},in_corner_x={in_left or in_right},in_corner_y={in_top or in_bottom})")
                
                # Log detailed info for first few frames with faces but no corner detection
                if not has_corner_facecam and faces and total_frames <= 5:
                    logger.info(
                        f"Frame {timestamp_ms}ms: {len(faces)} faces detected but no corner facecam. "
                        f"Rejected: {rejected_reasons[:3]}"
                    )
                
                # Log successful corner facecam detections (first 3)
                if has_corner_facecam and corner_facecam_frames <= 3:
                    logger.info(
                        f"Corner facecam FOUND at {timestamp_ms}ms: bbox=({corner_bbox[0]},{corner_bbox[1]},{corner_bbox[2]},{corner_bbox[3]}), "
                        f"area_ratio={face_area_ratio:.4f}"
                    )
                
                if has_corner_facecam:
                    corner_facecam_frames += 1
                    if corner_bbox:
                        corner_facecam_bboxes.append(corner_bbox)
                    layout_type = "screen_share"
                    reason = "corner_facecam_detected"
                else:
                    # No corner facecam found from MediaPipe detections
                    # Check if there's a LARGE center face AND also a SMALL corner face
                    # that might have been missed due to size thresholds
                    has_large_center_face = False
                    has_any_corner_face = False
                    corner_face_bbox = None
                    
                    for face in faces:
                        bbox = face.get("bbox", {})
                        x = bbox.get("x", 0)
                        y = bbox.get("y", 0)
                        w = bbox.get("width", 0)
                        h = bbox.get("height", 0)
                        
                        area = w * h
                        area_ratio = area / frame_area if frame_area > 0 else 0
                        face_center_x = x + w / 2
                        face_center_y = y + h / 2
                        
                        # Check if face is in corner region (relaxed - any size)
                        in_left = face_center_x < left_edge
                        in_right = face_center_x > right_edge
                        in_top = face_center_y < top_edge
                        in_bottom = face_center_y > bottom_edge
                        is_in_corner = (in_left or in_right) and (in_top or in_bottom)
                        
                        # Large face (>8% of frame) in center region
                        in_center_x = left_edge < face_center_x < right_edge
                        is_large = area_ratio > 0.08
                        
                        if is_large and in_center_x:
                            has_large_center_face = True
                        
                        # Any face in corner region (even if below min size threshold)
                        # This catches webcams that MediaPipe detected but we rejected
                        if is_in_corner and area_ratio > 0.001:  # Very low threshold
                            has_any_corner_face = True
                            corner_face_bbox = (x, y, w, h)
                    
                    # DECISION LOGIC:
                    # If we have BOTH a large center face AND a corner face -> screen_share
                    # This catches cases where corner webcam was detected but rejected by size
                    if has_large_center_face and has_any_corner_face:
                        layout_type = "screen_share"
                        reason = "large_center_face_with_corner_face"
                        # Use the corner face as the webcam
                        corner_bbox = corner_face_bbox
                        has_corner_facecam = True
                        corner_facecam_frames += 1
                        if corner_bbox:
                            corner_facecam_bboxes.append(corner_bbox)
                    else:
                        layout_type = "talking_head"
                        if has_large_center_face:
                            reason = "large_center_face_only"
                        else:
                            reason = "no_corner_facecam"
                
                # Include corner_bbox as 4th element for per-frame tracking
                frame_layouts.append((timestamp_ms, layout_type, face_area_ratio, corner_bbox))
                debug_signals.append({
                    "ts": timestamp_ms,
                    "layout": layout_type,
                    "reason": reason,
                    "face_area": f"{face_area_ratio:.4f}",
                    "corner": has_corner_facecam,
                    "bbox": corner_bbox,
                })
        else:
            # FALLBACK: Read video and use Haar cascade
            logger.warning("No pre-computed face detections, falling back to Haar cascade")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Failed to open video: {video_path}")
                    return []
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_area = frame_width * frame_height
                sample_interval = 1.0 / sample_fps
                
                start_sec = start_ms / 1000
                end_sec = end_ms / 1000
                current_time = start_sec
                
                while current_time < end_sec:
                    timestamp_ms = int(current_time * 1000)
                    frame_pos = int(current_time * fps)
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    
                    if not ret:
                        current_time += sample_interval
                        continue
                    
                    total_frames += 1
                    
                    # Use Haar cascade for face detection
                    detection_result = self._detect_corner_facecam(frame, frame_width, frame_height)
                    face_area_ratio = detection_result[0]
                    face_in_corner = detection_result[1]
                    corner_bbox = detection_result[2] if len(detection_result) > 2 else None
                    
                    has_corner_facecam = (
                        face_in_corner and 
                        CORNER_FACECAM_MIN_AREA < face_area_ratio < CORNER_FACECAM_MAX_AREA
                    )
                    
                    if has_corner_facecam:
                        corner_facecam_frames += 1
                        if corner_bbox:
                            corner_facecam_bboxes.append(corner_bbox)
                        layout_type = "screen_share"
                        reason = "corner_facecam_detected"
                    else:
                        layout_type = "talking_head"
                        reason = "no_corner_facecam"
                    
                    # Include corner_bbox as 4th element for per-frame tracking
                    frame_layouts.append((timestamp_ms, layout_type, face_area_ratio, corner_bbox))
                    debug_signals.append({
                        "ts": timestamp_ms,
                        "layout": layout_type,
                        "reason": reason,
                        "face_area": f"{face_area_ratio:.4f}",
                        "corner": face_in_corner,
                        "bbox": corner_bbox,
                    })
                    current_time += sample_interval
                
                cap.release()
            except Exception as e:
                logger.error(f"Frame analysis failed: {e}", exc_info=True)
                return []
        
        # Calculate average corner facecam bbox for rendering (only from frames with corner facecam)
        avg_corner_bbox = None
        if corner_facecam_bboxes:
            avg_x = int(sum(b[0] for b in corner_facecam_bboxes) / len(corner_facecam_bboxes))
            avg_y = int(sum(b[1] for b in corner_facecam_bboxes) / len(corner_facecam_bboxes))
            avg_w = int(sum(b[2] for b in corner_facecam_bboxes) / len(corner_facecam_bboxes))
            avg_h = int(sum(b[3] for b in corner_facecam_bboxes) / len(corner_facecam_bboxes))
            
            # CONSISTENCY CHECK: A real webcam overlay stays in a FIXED position
            # Calculate position variance - if face moves too much, it's not a webcam overlay
            if len(corner_facecam_bboxes) >= 3:
                x_positions = [b[0] for b in corner_facecam_bboxes]
                y_positions = [b[1] for b in corner_facecam_bboxes]
                x_variance = max(x_positions) - min(x_positions)
                y_variance = max(y_positions) - min(y_positions)
                
                # Webcam overlays move very little (maybe a few pixels due to detection jitter)
                # A real person moving around will have high variance (400+ pixels)
                MAX_POSITION_VARIANCE = 350  # pixels - increased to allow for webcam jitter and head movement
                
                if x_variance > MAX_POSITION_VARIANCE or y_variance > MAX_POSITION_VARIANCE:
                    logger.info(
                        f"Corner facecam REJECTED: Position too inconsistent (x_var={x_variance}, y_var={y_variance} > {MAX_POSITION_VARIANCE}px). "
                        f"This is likely a person moving around, not a fixed webcam overlay."
                    )
                    # Clear the corner facecam detections - this is not a real overlay
                    corner_facecam_frames = 0
                    corner_facecam_bboxes = []
                    avg_corner_bbox = None
                    # Update frame_layouts to all be talking_head
                    frame_layouts = [(ts, "talking_head", area, None) for ts, _, area, _ in frame_layouts]
                else:
                    avg_corner_bbox = (avg_x, avg_y, avg_w, avg_h)
                    logger.info(f"Average corner facecam bbox: x={avg_x}, y={avg_y}, w={avg_w}, h={avg_h} (position variance: x={x_variance}, y={y_variance})")
            else:
                avg_corner_bbox = (avg_x, avg_y, avg_w, avg_h)
                logger.info(f"Average corner facecam bbox: x={avg_x}, y={avg_y}, w={avg_w}, h={avg_h}")
        
        # Calculate corner facecam ratio for logging only
        corner_facecam_ratio = corner_facecam_frames / total_frames if total_frames > 0 else 0
        
        # NO MAJORITY VOTING: Keep per-frame layout decisions
        # This allows proper segment creation when video switches between layouts
        # (e.g., screen share with webcam -> talking head whiteboard -> screen share with webcam)
        # The segment merging will handle short segments to avoid flickering
        logger.info(
            f"Per-frame layout analysis: corner_facecam_ratio={corner_facecam_ratio:.1%} "
            f"({corner_facecam_frames}/{total_frames} frames). Keeping per-frame decisions."
        )
        
        # Log summary
        if frame_layouts:
            screen_share_count = sum(1 for fl in frame_layouts if fl[1] == "screen_share")
            talking_head_count = len(frame_layouts) - screen_share_count
            logger.info(
                f"Frame-by-frame analysis: {len(frame_layouts)} frames, "
                f"corner_facecam={corner_facecam_frames}/{total_frames} ({corner_facecam_ratio:.1%}), "
                f"screen_share={screen_share_count}, talking_head={talking_head_count}"
            )
            # Log sample frame signals for debugging
            if debug_signals:
                sample_frames = debug_signals[:3] + (debug_signals[-2:] if len(debug_signals) > 5 else [])
                for sig in sample_frames:
                    logger.info(f"  Frame {sig['ts']}ms: {sig['layout']} ({sig['reason']}) - face={sig['face_area']}, corner={sig['corner']}")
        
        # Store the average corner bbox for later use
        self._last_corner_facecam_bbox = avg_corner_bbox
        
        return frame_layouts

    def _create_layout_segments(
        self,
        frame_layouts: list[tuple],  # (timestamp, layout_type, face_area, corner_bbox)
        start_ms: int,
        end_ms: int,
    ) -> list[LayoutSegment]:
        """
        Create layout segments from frame-by-frame analysis.
        
        Groups consecutive frames with same layout into segments.
        Each segment gets the average corner_facecam_bbox from its frames.
        """
        if not frame_layouts:
            return [LayoutSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                layout_type="talking_head",
                confidence=0.5,
            )]
        
        segments = []
        current_layout = frame_layouts[0][1]
        segment_start = start_ms
        segment_start_idx = 0
        
        def get_segment_bbox(start_idx: int, end_idx: int) -> Optional[tuple]:
            """Calculate average bbox for frames in segment range."""
            bboxes = []
            for idx in range(start_idx, end_idx + 1):
                if idx < len(frame_layouts) and len(frame_layouts[idx]) > 3:
                    bbox = frame_layouts[idx][3]
                    if bbox:
                        bboxes.append(bbox)
            if not bboxes:
                return None
            avg_x = int(sum(b[0] for b in bboxes) / len(bboxes))
            avg_y = int(sum(b[1] for b in bboxes) / len(bboxes))
            avg_w = int(sum(b[2] for b in bboxes) / len(bboxes))
            avg_h = int(sum(b[3] for b in bboxes) / len(bboxes))
            return (avg_x, avg_y, avg_w, avg_h)
        
        for i in range(1, len(frame_layouts)):
            timestamp_ms = frame_layouts[i][0]
            layout_type = frame_layouts[i][1]
            
            if layout_type != current_layout:
                # Layout changed - create segment with bbox from its frames
                segment_bbox = get_segment_bbox(segment_start_idx, i - 1) if current_layout == "screen_share" else None
                segments.append(LayoutSegment(
                    start_ms=segment_start,
                    end_ms=timestamp_ms,
                    layout_type=current_layout,
                    confidence=0.9,
                    corner_facecam_bbox=segment_bbox,
                ))
                current_layout = layout_type
                segment_start = timestamp_ms
                segment_start_idx = i
        
        # Add final segment with its bbox
        final_bbox = get_segment_bbox(segment_start_idx, len(frame_layouts) - 1) if current_layout == "screen_share" else None
        segments.append(LayoutSegment(
            start_ms=segment_start,
            end_ms=end_ms,
            layout_type=current_layout,
            confidence=0.9,
            corner_facecam_bbox=final_bbox,
        ))
        
        # MERGE SHORT SEGMENTS: Eliminate jarring transitions from detection gaps
        # Short segments (< 3 seconds) are usually detection failures, not real layout changes
        MIN_SEGMENT_DURATION_MS = 3000  # 3 seconds
        
        if len(segments) > 1:
            # Calculate total duration by layout type
            screen_share_duration = sum(
                seg.end_ms - seg.start_ms 
                for seg in segments 
                if seg.layout_type == "screen_share"
            )
            total_duration = end_ms - start_ms
            dominant_layout = "screen_share" if screen_share_duration > total_duration / 2 else "talking_head"
            
            # Get global average bbox from all screen_share segments
            all_bboxes = [seg.corner_facecam_bbox for seg in segments if seg.corner_facecam_bbox]
            global_avg_bbox = None
            if all_bboxes:
                avg_x = int(sum(b[0] for b in all_bboxes) / len(all_bboxes))
                avg_y = int(sum(b[1] for b in all_bboxes) / len(all_bboxes))
                avg_w = int(sum(b[2] for b in all_bboxes) / len(all_bboxes))
                avg_h = int(sum(b[3] for b in all_bboxes) / len(all_bboxes))
                global_avg_bbox = (avg_x, avg_y, avg_w, avg_h)
            
            # Merge short segments that differ from dominant layout
            merged_segments = []
            for seg in segments:
                seg_duration = seg.end_ms - seg.start_ms
                
                # If segment is short and differs from dominant, merge into neighbors
                if seg_duration < MIN_SEGMENT_DURATION_MS and seg.layout_type != dominant_layout:
                    if merged_segments:
                        # Extend previous segment
                        merged_segments[-1] = LayoutSegment(
                            start_ms=merged_segments[-1].start_ms,
                            end_ms=seg.end_ms,
                            layout_type=merged_segments[-1].layout_type,
                            confidence=merged_segments[-1].confidence,
                            corner_facecam_bbox=merged_segments[-1].corner_facecam_bbox or global_avg_bbox,
                        )
                    else:
                        # First segment is short - convert to dominant layout
                        merged_seg = LayoutSegment(
                            start_ms=seg.start_ms,
                            end_ms=seg.end_ms,
                            layout_type=dominant_layout,
                            confidence=seg.confidence,
                            corner_facecam_bbox=global_avg_bbox if dominant_layout == "screen_share" else None,
                        )
                        merged_segments.append(merged_seg)
                else:
                    # Merge with previous if same layout type
                    if merged_segments and merged_segments[-1].layout_type == seg.layout_type:
                        merged_segments[-1] = LayoutSegment(
                            start_ms=merged_segments[-1].start_ms,
                            end_ms=seg.end_ms,
                            layout_type=seg.layout_type,
                            confidence=max(merged_segments[-1].confidence, seg.confidence),
                            corner_facecam_bbox=merged_segments[-1].corner_facecam_bbox or seg.corner_facecam_bbox,
                        )
                    else:
                        merged_segments.append(seg)
            
            if len(merged_segments) < len(segments):
                logger.info(
                    f"Merged {len(segments)} raw segments into {len(merged_segments)} "
                    f"(eliminated {len(segments) - len(merged_segments)} short segments < {MIN_SEGMENT_DURATION_MS}ms)"
                )
                segments = merged_segments
        
        # Log segment creation
        for seg in segments:
            logger.info(
                f"Created segment: {seg.start_ms}ms-{seg.end_ms}ms = {seg.layout_type}, "
                f"bbox={seg.corner_facecam_bbox}"
            )
        
        return segments

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
        
        # Log frame analyses for debugging
        if frame_analyses:
            screen_share_count = sum(1 for f in frame_analyses if f.layout_type == "screen_share")
            talking_head_count = len(frame_analyses) - screen_share_count
            avg_edge = sum(f.edge_density for f in frame_analyses) / len(frame_analyses)
            logger.info(
                f"Frame analysis summary: {len(frame_analyses)} frames, "
                f"screen_share={screen_share_count}, talking_head={talking_head_count}, "
                f"avg_edge_density={avg_edge:.4f}, raw_segments={len(raw_segments)}"
            )

        # Merge short segments to avoid flickering
        merged_segments = self._merge_short_segments(raw_segments)

        # Determine dominant layout
        dominant_layout = self._calculate_dominant_layout(merged_segments)

        # Check if face is embedded in screen content (small face in corner with high edge density)
        # This indicates the video already has a picture-in-picture layout and we should NOT split
        has_embedded_facecam = self._detect_embedded_facecam(frame_analyses)

        # Extract transition timestamps
        transitions = []
        for i in range(1, len(merged_segments)):
            transitions.append(merged_segments[i].start_ms)

        has_transitions = len(merged_segments) > 1

        logger.info(
            f"Layout analysis complete: {len(merged_segments)} segments, "
            f"transitions={has_transitions}, dominant={dominant_layout}, "
            f"embedded_facecam={has_embedded_facecam}"
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
            has_embedded_facecam=has_embedded_facecam,
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

        Classification logic (IMPROVED):
        - talking_head: Large face in camera region, low edge density in TOP region
        - screen_share: High edge density in TOP region (screen content visible)
        
        Key insight: Layout is determined by screen content presence, NOT face count.
        We look at edge density in the TOP region specifically because:
        - Screen content (UI, browser, etc.) appears in top portion
        - Camera/webcam is in bottom portion
        """
        # Calculate overall edge density (for compatibility)
        edge_density = self._calculate_edge_density(frame)
        
        # NEW: Calculate edge density specifically in TOP region
        top_edge_density = self._calculate_top_region_edge_density(frame)

        # Analyze face detections - only consider faces in CAMERA region (bottom)
        face_area_ratio = 0.0
        face_in_corner = False
        largest_face_area = 0
        camera_region_min_y = 0.35  # Bottom 65% is camera region

        if face_detections:
            for face in face_detections:
                bbox = face.get("bbox", {})
                face_w = bbox.get("width", 0)
                face_h = bbox.get("height", 0)
                face_y = bbox.get("y", 0)
                face_area_px = face_w * face_h
                
                # Only consider faces in camera region (bottom portion)
                face_center_y = (face_y + face_h / 2) / frame_height
                if face_center_y < camera_region_min_y:
                    # Face is in top region (likely screen content) - skip
                    continue

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

        # IMPROVED: Classification using TOP region edge density
        # High edge density in TOP = screen content = screen_share
        # Low edge density in TOP = full-screen speaker = talking_head
        layout_type, confidence = self._determine_layout_type_improved(
            face_area_ratio=face_area_ratio,
            face_in_corner=face_in_corner,
            edge_density=edge_density,
            top_edge_density=top_edge_density,
            has_face=face_area_ratio > 0,  # Only count faces in camera region
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

    def _calculate_top_region_edge_density(self, frame: np.ndarray) -> float:
        """
        Calculate edge density specifically in the TOP region of the frame.
        
        This is used to detect screen content (browser, UI, code editor, etc.).
        
        Key insight: Screen content has VERY HIGH edge density (sharp UI elements,
        text, buttons, etc.) compared to a person talking (even with whiteboard).
        
        We analyze the top 60% of the frame because:
        - In split layouts, screen content is in the top portion
        - Camera/face is typically in the bottom portion
        - A person with whiteboard will have some edges but much less than UI
        """
        try:
            height, width = frame.shape[:2]
            # Analyze top 60% of frame for screen content (covers full split region)
            top_region = frame[:int(height * 0.6), :]
            
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            region_height, region_width = top_region.shape[:2]
            edge_density = np.sum(edges > 0) / (region_width * region_height)
            return edge_density
        except Exception:
            return 0.0

    async def _detect_screen_content(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        sample_count: int = 5,
    ) -> bool:
        """
        Detect if video clip contains screen content (UI, browser, code, etc.)
        by analyzing edge density in the TOP region of sampled frames.
        
        This is the PRIMARY signal for choosing screen_share vs talking_head layout.
        - Screen recordings have HIGH edge density (text, UI elements, graphics)
        - Talking head videos have LOW edge density (face, simple background)
        
        Args:
            video_path: Path to video file
            start_ms: Start time in ms
            end_ms: End time in ms
            sample_count: Number of frames to sample
            
        Returns:
            True if screen content detected, False otherwise
        """
        # RAISED threshold - screen content typically has > 5% edge density in top region
        # Whiteboard/plain backgrounds can have 3-4% edge density, so we need higher threshold
        SCREEN_CONTENT_THRESHOLD = 0.05  # 5% edge density in top region
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning("Could not open video for screen content detection")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration_ms = end_ms - start_ms
            sample_interval_ms = duration_ms / (sample_count + 1)
            
            edge_densities = []
            
            for i in range(1, sample_count + 1):
                timestamp_ms = start_ms + int(i * sample_interval_ms)
                frame_pos = int((timestamp_ms / 1000) * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Calculate edge density in TOP region only
                top_edge_density = self._calculate_top_region_edge_density(frame)
                edge_densities.append(top_edge_density)
                logger.debug(f"Frame {i} top edge density: {top_edge_density:.4f}")
            
            cap.release()
            
            if not edge_densities:
                logger.warning("No frames analyzed for screen content detection")
                return False
            
            # Use average edge density
            avg_edge_density = sum(edge_densities) / len(edge_densities)
            max_edge_density = max(edge_densities)
            
            # Check if majority of frames have screen content
            screen_frames = sum(1 for d in edge_densities if d > SCREEN_CONTENT_THRESHOLD)
            screen_ratio = screen_frames / len(edge_densities)
            
            # DECISION: Has screen content if:
            # - Average edge density above threshold, OR
            # - More than 50% of frames have screen content, OR
            # - Max edge density is very high (> 2x threshold)
            has_screen_content = (
                avg_edge_density > SCREEN_CONTENT_THRESHOLD or 
                screen_ratio > 0.5 or
                max_edge_density > SCREEN_CONTENT_THRESHOLD * 2
            )
            
            logger.info(
                f"Screen content detection: avg={avg_edge_density:.4f}, "
                f"max={max_edge_density:.4f}, screen_frames={screen_frames}/{len(edge_densities)}, "
                f"threshold={SCREEN_CONTENT_THRESHOLD}, HAS_SCREEN_CONTENT={has_screen_content}"
            )
            
            return has_screen_content
            
        except Exception as e:
            logger.error(f"Screen content detection failed: {e}", exc_info=True)
            return False

    def _calculate_bottom_face_presence(
        self,
        face_detections: list[dict],
        frame_height: int,
    ) -> tuple[bool, float]:
        """
        Check if there's a significant face in the bottom portion of the frame.
        
        Returns:
            Tuple of (has_camera_face, face_area_ratio)
        """
        camera_region_min_y = 0.35  # Bottom 65% is camera region
        
        for face in face_detections:
            bbox = face.get("bbox", {})
            face_y = bbox.get("y", 0)
            face_h = bbox.get("height", 0)
            face_w = bbox.get("width", 0)
            
            # Check if face center is in bottom region
            face_center_y = (face_y + face_h / 2) / frame_height
            if face_center_y >= camera_region_min_y:
                # Calculate face area ratio
                face_area = face_w * face_h
                # Assume 1920x1080 for normalization
                frame_area = 1920 * 1080
                face_area_ratio = face_area / frame_area
                return (True, face_area_ratio)
        
        return (False, 0.0)

    def _determine_layout_type_improved(
        self,
        face_area_ratio: float,
        face_in_corner: bool,
        edge_density: float,
        top_edge_density: float,
        has_face: bool,
    ) -> tuple[str, float]:
        """
        IMPROVED layout type detection using multiple signals.
        
        PRIORITY ORDER (most definitive first):
        1. Large face (>8% of frame in camera region) = TALKING HEAD
           - When person fills the frame, they are the content.
        2. Very high top edge density (>0.08) = SCREEN SHARE
           - Strong UI/browser/code content visible.
        3. Face in corner + moderate top edge = SCREEN SHARE  
           - Small webcam overlay with content.
        4. Low top edge density (<0.05) = TALKING HEAD
           - No significant screen content detected.
        
        Returns:
            Tuple of (layout_type, confidence)
        """
        # RULE 1 (HIGHEST PRIORITY): Large face = TALKING HEAD
        # When the face takes up >8% of the frame, the person IS the content.
        # This overrides edge density because a person talking with whiteboard
        # background can have moderate edge density.
        LARGE_FACE_THRESHOLD = 0.08  # 8% of frame area
        if face_area_ratio > LARGE_FACE_THRESHOLD:
            return ("talking_head", 0.98)
        
        # RULE 2: Very high top edge density = SCREEN SHARE
        # Strong UI elements, code, browser = definitely screen content.
        HIGH_EDGE_THRESHOLD = 0.08  # 8% edge density = heavy UI
        if top_edge_density > HIGH_EDGE_THRESHOLD:
            return ("screen_share", 0.95)
        
        # RULE 3: Face in corner + moderate edge = SCREEN SHARE
        # Small webcam overlay in corner while showing content.
        MODERATE_EDGE_THRESHOLD = 0.04  # 4% edge density
        if has_face and face_in_corner and top_edge_density > MODERATE_EDGE_THRESHOLD:
            return ("screen_share", 0.90)
        
        # RULE 4: Moderate-high edge density with small face = SCREEN SHARE
        # Screen content with small webcam (not in corner).
        SCREEN_CONTENT_THRESHOLD = 0.06  # 6% edge density
        if top_edge_density > SCREEN_CONTENT_THRESHOLD and face_area_ratio < 0.05:
            return ("screen_share", 0.85)
        
        # RULE 5: Low edge density = TALKING HEAD
        # No significant screen content visible.
        LOW_EDGE_THRESHOLD = 0.05  # Below 5% = minimal UI
        if top_edge_density < LOW_EDGE_THRESHOLD:
            return ("talking_head", 0.90)
        
        # RULE 6: Medium edge density + face present = TALKING HEAD
        # Ambiguous but favor talking head when face is present.
        if has_face:
            return ("talking_head", 0.75)
        
        # Default: No face + medium edge = SCREEN SHARE
        if top_edge_density > 0.04:
            return ("screen_share", 0.70)
        
        return ("talking_head", 0.60)

    def _determine_layout_type(
        self,
        face_area_ratio: float,
        face_in_corner: bool,
        edge_density: float,
        has_face: bool,
    ) -> tuple[str, float]:
        """
        Determine layout type using simplified binary logic.
        
        Key insight from OpusClip/Klap research:
        - Face in corner + screen content = screen_share (split screen)
        - Large/medium centered face = talking_head (9:16 full height)
        
        Returns:
            Tuple of (layout_type, confidence)
        """
        # Rule 1: Face in corner = screen_share (typical screen recording with facecam PiP)
        # This is the clearest signal for screen share
        if has_face and face_in_corner:
            return ("screen_share", 0.95)
        
        # Rule 2: Large face (>15%) centered = talking head
        # Person is the main content, no need for split screen
        if face_area_ratio > FACE_AREA_THRESHOLD_TALKING_HEAD and not face_in_corner:
            return ("talking_head", 0.95)
        
        # Rule 3: Medium face (8-15%) centered with low edge density = talking head
        # Person talking with simple background (office, room, etc.)
        if face_area_ratio > 0.08 and not face_in_corner and edge_density < 0.08:
            return ("talking_head", 0.85)
        
        # Rule 4: Small face + HIGH edge density = screen_share (screen recording with small facecam)
        if has_face and face_area_ratio < 0.08 and edge_density > 0.10:
            return ("screen_share", 0.90)
        
        # Rule 5: No face but high edge density = pure screen share
        if not has_face and edge_density > EDGE_DENSITY_THRESHOLD:
            return ("screen_share", 0.85)
        
        # Rule 6: Face exists, not in corner = default to talking head
        # When in doubt with a centered face, use talking head (9:16 focus)
        if has_face and not face_in_corner:
            return ("talking_head", 0.70)
        
        # Default: No face, low edge density - treat as talking head
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

    def _create_segments_from_vlm_frames(
        self,
        frame_results: list,
        start_ms: int,
        end_ms: int,
        default_webcam_bbox: Optional[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
    ) -> list[LayoutSegment]:
        """
        Create layout segments from VLM per-frame results.
        
        This method converts VLM frame-by-frame layout detection into
        contiguous segments for rendering. When layout changes between
        frames, a new segment is created.
        
        Args:
            frame_results: List of VLMFrameResult with per-frame layouts
            start_ms: Clip start time
            end_ms: Clip end time
            default_webcam_bbox: Default webcam bbox for screen_share segments
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            List of LayoutSegment representing layout timeline
        """
        if not frame_results:
            return [LayoutSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                layout_type="talking_head",
                confidence=0.5,
            )]
        
        segments = []
        current_layout = None
        segment_start = start_ms
        previous_timestamp_ms: Optional[int] = None
        current_webcam_pos = None
        confidences = []
        
        for i, frame_result in enumerate(frame_results):
            # Determine layout for this frame
            # If webcam detected, it's screen_share; otherwise use the frame's layout_type
            if frame_result.has_corner_webcam:
                frame_layout = "screen_share"
                frame_webcam_pos = frame_result.webcam_position
            else:
                frame_layout = frame_result.layout_type
                frame_webcam_pos = None
            
            if current_layout is None:
                # First frame
                current_layout = frame_layout
                current_webcam_pos = frame_webcam_pos
                confidences.append(frame_result.confidence)
            elif frame_layout != current_layout:
                # Layout changed - estimate boundary between the last sample of the
                # previous layout and the first sample of the new layout.
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
                
                prev_ts = previous_timestamp_ms if previous_timestamp_ms is not None else segment_start
                segment_end = int((prev_ts + frame_result.timestamp_ms) / 2)
                
                # Keep boundaries monotonic and within the clip range.
                # Avoid zero-length segments; short segments are merged later.
                min_gap_ms = 250
                segment_end = max(segment_start + min_gap_ms, min(segment_end, end_ms))
                
                logger.info(
                    f"Layout transition detected near {segment_end}ms: "
                    f"{current_layout} -> {frame_layout}"
                )
                
                # Create webcam bbox for screen_share segments
                webcam_bbox = None
                if current_layout == "screen_share":
                    if current_webcam_pos:
                        webcam_bbox = self._estimate_webcam_bbox(
                            current_webcam_pos, frame_width, frame_height
                        )
                    else:
                        webcam_bbox = default_webcam_bbox
                
                segments.append(LayoutSegment(
                    start_ms=segment_start,
                    end_ms=segment_end,
                    layout_type=current_layout,
                    confidence=avg_confidence,
                    corner_facecam_bbox=webcam_bbox,
                ))
                
                logger.info(
                    f"VLM segment: {segment_start}ms - {segment_end}ms = {current_layout}"
                )
                
                # Start new segment
                segment_start = segment_end
                current_layout = frame_layout
                current_webcam_pos = frame_webcam_pos
                confidences = [frame_result.confidence]
            else:
                # Same layout - accumulate confidence
                confidences.append(frame_result.confidence)
                if frame_webcam_pos:
                    current_webcam_pos = frame_webcam_pos

            previous_timestamp_ms = frame_result.timestamp_ms
        
        # Add final segment
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
        webcam_bbox = None
        if current_layout == "screen_share":
            if current_webcam_pos:
                webcam_bbox = self._estimate_webcam_bbox(
                    current_webcam_pos, frame_width, frame_height
                )
            else:
                webcam_bbox = default_webcam_bbox
        
        segments.append(LayoutSegment(
            start_ms=segment_start,
            end_ms=end_ms,
            layout_type=current_layout,
            confidence=avg_confidence,
            corner_facecam_bbox=webcam_bbox,
        ))
        
        logger.info(
            f"VLM final segment: {segment_start}ms - {end_ms}ms = {current_layout}"
        )
        
        return segments
    
    def _estimate_webcam_bbox(
        self,
        position: str,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Estimate webcam bounding box based on corner position.
        
        Returns typical webcam size/position for split-screen rendering.
        """
        # Typical webcam overlay is about 15-20% of frame width
        webcam_width = int(frame_width * 0.18)
        webcam_height = int(frame_height * 0.18)
        
        # Add small margin from edges
        margin = 20
        
        if position == "top-left":
            x = margin
            y = margin
        elif position == "top-right":
            x = frame_width - webcam_width - margin
            y = margin
        elif position == "bottom-left":
            x = margin
            y = frame_height - webcam_height - margin
        elif position == "bottom-right":
            x = frame_width - webcam_width - margin
            y = frame_height - webcam_height - margin
        else:
            # Default to bottom-right
            x = frame_width - webcam_width - margin
            y = frame_height - webcam_height - margin
        
        return (x, y, webcam_width, webcam_height)

    def _merge_short_segments(
        self,
        segments: list[LayoutSegment],
    ) -> list[LayoutSegment]:
        """
        Merge segments shorter than MIN_SEGMENT_DURATION_MS to avoid flickering.

        Short segments are merged into adjacent segments based on confidence.
        Preserves corner_facecam_bbox for screen_share segments.
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
                        corner_facecam_bbox=next_seg.corner_facecam_bbox or current.corner_facecam_bbox,
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
                        corner_facecam_bbox=prev_seg.corner_facecam_bbox or current.corner_facecam_bbox,
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
                        corner_facecam_bbox=prev_seg.corner_facecam_bbox or current.corner_facecam_bbox,
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
                    corner_facecam_bbox=prev.corner_facecam_bbox or seg.corner_facecam_bbox,
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

    def _detect_embedded_facecam(
        self,
        frame_analyses: list[FrameLayoutInfo],
    ) -> bool:
        """
        Detect if the video has an embedded facecam (picture-in-picture) or 
        side-by-side layout where the face is already part of the screen recording.

        When a video already has a facecam overlay/layout embedded in the screen recording,
        we should NOT do split screen as it would duplicate the face or mess up the layout.

        Indicators of embedded facecam/layout:
        1. Consistent small face in corner (face_in_corner=True) - classic PiP
        2. High edge density (UI/screen content with face overlay)
        3. Small face area ratio (< 15% of frame) with screen content
        4. Screen share layout detected consistently

        Returns:
            True if embedded facecam detected, False otherwise
        """
        if not frame_analyses:
            return False

        # Count frames with various indicators
        embedded_indicators = 0
        screen_share_with_face_in_corner = 0
        screen_share_with_small_face = 0
        screen_share_count = 0

        for frame in frame_analyses:
            # Track all screen_share frames
            if frame.layout_type == "screen_share":
                screen_share_count += 1
                
                # Small face with high edge density = embedded facecam
                if frame.face_area_ratio < 0.15 and frame.edge_density > 0.04:
                    screen_share_with_small_face += 1
                
                # Face in corner = classic PiP
                if frame.face_in_corner:
                    screen_share_with_face_in_corner += 1

            # Check for the classic embedded facecam pattern:
            # - Small face (< 15% of frame) in a corner or anywhere
            # - High edge density (screen content)
            if frame.face_area_ratio < 0.15 and frame.face_area_ratio > 0.001:
                if frame.edge_density > 0.04:  # Has screen content
                    embedded_indicators += 1

        # Calculate ratios
        total_frames = len(frame_analyses)
        embedded_ratio = embedded_indicators / total_frames if total_frames > 0 else 0
        corner_face_ratio = screen_share_with_face_in_corner / total_frames if total_frames > 0 else 0
        screen_share_ratio = screen_share_count / total_frames if total_frames > 0 else 0
        small_face_screen_ratio = screen_share_with_small_face / total_frames if total_frames > 0 else 0

        # Detect embedded facecam if:
        # 1. Majority of frames are screen_share with small face, OR
        # 2. High embedded indicator ratio, OR
        # 3. Consistent face in corner pattern
        has_embedded = (
            embedded_ratio >= 0.50 or 
            corner_face_ratio >= 0.60 or
            small_face_screen_ratio >= 0.50 or
            (screen_share_ratio >= 0.70 and embedded_ratio >= 0.30)
        )

        logger.info(
            f"Embedded facecam analysis: embedded_ratio={embedded_ratio:.2f}, "
            f"corner_face_ratio={corner_face_ratio:.2f}, screen_share_ratio={screen_share_ratio:.2f}, "
            f"small_face_screen_ratio={small_face_screen_ratio:.2f} -> has_embedded={has_embedded}"
        )

        return has_embedded

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

    async def analyze_with_face_tracking(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        frame_width: int = 1920,
        frame_height: int = 1080,
        sample_fps: float = 10.0,
    ) -> tuple[LayoutAnalysis, Optional["TrackingResult"]]:
        """
        Analyze layout with robust face tracking for dominant face detection.
        
        This method uses the FaceTracker service for frame-by-frame face tracking
        which provides:
        1. Consistent face tracking across frames (temporal smoothing)
        2. Dominant face identification (the MAIN person to show in split-screen)
        3. Smoothed bounding boxes for stable rendering
        
        Args:
            video_path: Path to the video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            frame_width: Video frame width
            frame_height: Video frame height
            sample_fps: FPS for face tracking (higher = more accurate, slower)
            
        Returns:
            Tuple of (LayoutAnalysis, TrackingResult or None)
            - LayoutAnalysis: Layout segments and transitions
            - TrackingResult: Face tracking data with dominant face info
        """
        try:
            from app.services.face_tracker import track_faces_in_video, TrackingResult
        except ImportError:
            logger.warning("FaceTracker not available, using standard analysis")
            layout = await self.analyze_clip_layout(
                video_path, start_ms, end_ms, sample_fps=sample_fps
            )
            return (layout, None)
        
        logger.info(
            f"Running face tracking analysis: {start_ms}ms - {end_ms}ms at {sample_fps} FPS"
        )
        
        try:
            # Run face tracking
            tracking_result = await track_faces_in_video(
                video_path=video_path,
                start_ms=start_ms,
                end_ms=end_ms,
                sample_fps=sample_fps,
            )
            
            if tracking_result.dominant_track:
                logger.info(
                    f"Face tracking complete: {len(tracking_result.tracks)} tracks, "
                    f"dominant_track_id={tracking_result.dominant_track_id}, "
                    f"dominant_visibility={tracking_result.dominant_face_visibility:.1%}, "
                    f"dominance_score={tracking_result.dominant_track.dominance_score:.3f}"
                )
                
                # If we found a dominant face, update the layout analysis with its bbox
                dominant_bbox = tracking_result.dominant_track.smoothed_bbox
                
                if dominant_bbox:
                    # Run standard layout analysis
                    layout = await self.analyze_clip_layout(
                        video_path, start_ms, end_ms, sample_fps=3.0
                    )
                    
                    # Override corner_facecam_bbox with tracked dominant face
                    # ONLY if the dominant face looks like a corner webcam (small, in corner)
                    x, y, w, h = dominant_bbox
                    face_area_ratio = (w * h) / (frame_width * frame_height)
                    
                    # Check if face is in corner region (outer 30%)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    is_in_corner = (
                        (center_x < frame_width * 0.30 or center_x > frame_width * 0.70) or
                        (center_y < frame_height * 0.30 or center_y > frame_height * 0.70)
                    )
                    
                    # If face is small (< 10% of frame) and in corner, it's likely a webcam overlay
                    # In that case, use it for split-screen rendering
                    if face_area_ratio < 0.10 and is_in_corner:
                        logger.info(
                            f"Dominant face is webcam overlay: {w}x{h} at ({x},{y}), "
                            f"area_ratio={face_area_ratio:.2%}, is_in_corner={is_in_corner}"
                        )
                        layout.corner_facecam_bbox = dominant_bbox
                        
                        # Update all screen_share segments with this bbox
                        for seg in layout.layout_segments:
                            if seg.layout_type == "screen_share":
                                seg.corner_facecam_bbox = dominant_bbox
                    else:
                        # Face is large/centered - likely the main subject in talking head
                        logger.info(
                            f"Dominant face is main subject (not webcam overlay): {w}x{h} at ({x},{y}), "
                            f"area_ratio={face_area_ratio:.2%}, is_in_corner={is_in_corner}"
                        )
                    
                    return (layout, tracking_result)
            else:
                logger.warning("No dominant face found in tracking")
                
        except Exception as e:
            logger.warning(f"Face tracking failed: {e}, falling back to standard analysis")
        
        # Fallback to standard analysis
        layout = await self.analyze_clip_layout(
            video_path, start_ms, end_ms, sample_fps=3.0
        )
        return (layout, None)


def get_dominant_face_for_clip(
    tracking_result: "TrackingResult",
    timestamp_ms: int,
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[int, int, int, int]]:
    """
    Get the dominant face bounding box at a specific timestamp.
    
    This is a helper function for rendering to get the smoothed,
    tracked face position at any point in time.
    
    Args:
        tracking_result: TrackingResult from analyze_with_face_tracking
        timestamp_ms: Timestamp in milliseconds
        frame_width: Video frame width
        frame_height: Video frame height
        
    Returns:
        (x, y, width, height) of the dominant face, or None
    """
    if not tracking_result or not tracking_result.frame_results:
        return None
    
    # Find the closest frame to the requested timestamp
    closest_frame = None
    min_diff = float('inf')
    
    for frame_result in tracking_result.frame_results:
        diff = abs(frame_result.timestamp_ms - timestamp_ms)
        if diff < min_diff:
            min_diff = diff
            closest_frame = frame_result
    
    if closest_frame and closest_frame.dominant_face_bbox:
        return closest_frame.dominant_face_bbox
    
    return None
