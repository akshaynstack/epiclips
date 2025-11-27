# Scene Switching: Dynamic Layout Detection & Mid-Clip Layout Changes

## Overview

This document outlines a plan to implement **dynamic scene switching** - the ability to automatically detect and switch between different layouts (screen share, talking head, full screen, etc.) within a single clip based on the video content at each moment.

### Current State
- Single layout per clip (hardcoded to `screen_share`)
- 50/50 split: screen content top, face bottom
- Layout determined once at clip creation, never changes

### Proposed State
- Multiple layouts within a single clip
- Auto-detection of optimal layout per time segment
- Smooth transitions between layout changes
- Example: 0-3s talking head → 3-7s screen share → 7-10s talking head

---

## Layout Types

### 1. `SCREEN_SHARE` (Current Default)
**When to use:** Speaker sharing screen with webcam overlay

```
┌────────────────────┐
│   SCREEN CONTENT   │  50%
│   (code, slides)   │
├────────────────────┤
│   SPEAKER FACE     │  50%
│   (scaled up)      │
└────────────────────┘
```

**Detection signals:**
- Small face detected in corner (< 12% of frame)
- High edge density (UI elements, text)
- Webcam overlay pattern detected

### 2. `TALKING_HEAD`
**When to use:** Speaker face dominates frame, minimal screen content

```
┌────────────────────┐
│                    │
│    SPEAKER FACE    │  100%
│    (9:16 crop)     │
│                    │
└────────────────────┘
```

**Detection signals:**
- Large face detected (> 15% of frame area)
- Face is centered (not in corner)
- Low edge density outside face region

### 3. `FULL_SCREEN`
**When to use:** No face visible, content-only (gameplay, B-roll, animations)

```
┌────────────────────┐
│                    │
│   FULL CONTENT     │  100%
│   (center crop)    │
│                    │
└────────────────────┘
```

**Detection signals:**
- No face detected OR face < 3% of frame
- Could be gameplay, presentation, or B-roll footage

### 4. `PICTURE_IN_PICTURE` (Future)
**When to use:** Large screen content with small speaker overlay

```
┌────────────────────┐
│   SCREEN CONTENT   │
│                    │
│              ┌────┐│
│              │FACE││  Small overlay
│              └────┘│
└────────────────────┘
```

**Detection signals:**
- Screen content detected as primary
- Small face in corner
- User preference for PiP style

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMIC LAYOUT PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              Input Video
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: FRAME SAMPLING (Existing)                                         │
│  ─────────────────────────────────────                                      │
│  Sample frames at 2 FPS (every 0.5s)                                        │
│  Already implemented in detection_pipeline.py                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: PER-FRAME ANALYSIS (Existing + Enhancement)                       │
│  ─────────────────────────────────────────────────────                      │
│                                                                             │
│  For each sampled frame, collect:                                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EXISTING DATA (from face_detector.py, content_region_detector.py)   │   │
│  │                                                                     │   │
│  │ • face_detections[]     - bbox, confidence, track_id                │   │
│  │ • face_area_ratio       - face area / frame area                    │   │
│  │ • face_position         - center coordinates                        │   │
│  │ • webcam_detected       - is face in corner?                        │   │
│  │ • is_screen_share       - heuristic from content_region_detector    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ NEW DATA (to be added)                                              │   │
│  │                                                                     │   │
│  │ • edge_density          - % of frame with edges (UI/text indicator) │   │
│  │ • face_is_centered      - is face in middle third of frame?         │   │
│  │ • recommended_layout    - LayoutType enum                           │   │
│  │ • layout_confidence     - 0.0 - 1.0                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: FrameLayoutAnalysis per frame                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LAYOUT CLASSIFICATION (New)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Decision tree per frame:                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  IF face_area_ratio > 0.15 AND face_is_centered:                    │   │
│  │      layout = TALKING_HEAD                                          │   │
│  │      confidence = 0.9                                               │   │
│  │                                                                     │   │
│  │  ELIF face_area_ratio < 0.03 OR no_face_detected:                   │   │
│  │      layout = FULL_SCREEN                                           │   │
│  │      confidence = 0.8                                               │   │
│  │                                                                     │   │
│  │  ELIF webcam_detected AND face_area_ratio < 0.12:                   │   │
│  │      layout = SCREEN_SHARE                                          │   │
│  │      confidence = 0.95                                              │   │
│  │                                                                     │   │
│  │  ELIF edge_density > 0.08 AND face_area_ratio < 0.10:               │   │
│  │      layout = SCREEN_SHARE                                          │   │
│  │      confidence = 0.7                                               │   │
│  │                                                                     │   │
│  │  ELSE:                                                              │   │
│  │      layout = SCREEN_SHARE  # Safe default                          │   │
│  │      confidence = 0.5                                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: (timestamp_ms, LayoutType, confidence) per frame                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: TEMPORAL SEGMENTATION (New)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Group consecutive frames with same layout into segments:                   │
│                                                                             │
│  Raw frame layouts:                                                         │
│  Frame:   0    1    2    3    4    5    6    7    8    9                    │
│  Time:   0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5                   │
│  Layout:  SS   SS   SS   TH   TH   TH   TH   SS   SS   SS                   │
│                                                                             │
│  After grouping:                                                            │
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────────┐          │
│  │  Segment 1   │  │     Segment 2        │  │    Segment 3     │          │
│  │  SCREEN_SHARE│  │    TALKING_HEAD      │  │   SCREEN_SHARE   │          │
│  │  0.0 - 1.5s  │  │    1.5 - 3.5s        │  │   3.5 - 5.0s     │          │
│  └──────────────┘  └──────────────────────┘  └──────────────────┘          │
│                                                                             │
│  Smoothing rules:                                                           │
│  • Minimum segment duration: 1.5 seconds                                    │
│  • Segments shorter than minimum are merged with neighbors                  │
│  • Prefer merging with higher-confidence neighbor                           │
│  • Hysteresis: require 3+ consecutive frames to trigger switch              │
│                                                                             │
│  Output: LayoutSegment[] with start/end times                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: SEGMENT RENDERING (Modified)                                      │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  For each LayoutSegment:                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  segment_1.mp4 ← render_with_layout(SCREEN_SHARE, 0.0s - 1.5s)      │   │
│  │  segment_2.mp4 ← render_with_layout(TALKING_HEAD, 1.5s - 3.5s)      │   │
│  │  segment_3.mp4 ← render_with_layout(SCREEN_SHARE, 3.5s - 5.0s)      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Each layout uses existing render methods:                                  │
│  • SCREEN_SHARE  → _render_opusclip_mode()                                  │
│  • TALKING_HEAD  → _render_focus_mode()                                     │
│  • FULL_SCREEN   → _render_static() with center crop                        │
│                                                                             │
│  Output: Temporary segment video files                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: SEGMENT CONCATENATION (New)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Merge segments using FFmpeg concat demuxer:                                │
│                                                                             │
│  concat_list.txt:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ file 'segment_1.mp4'                                                │   │
│  │ file 'segment_2.mp4'                                                │   │
│  │ file 'segment_3.mp4'                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FFmpeg command:                                                            │
│  ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp4            │
│                                                                             │
│  Transition options (future enhancement):                                   │
│  • Hard cut (default) - instant switch                                      │
│  • Crossfade (0.1-0.2s) - smooth blend                                      │
│  • Cut on scene change - detect natural transition points                   │
│                                                                             │
│  Output: Final rendered clip with dynamic layouts                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: CAPTION OVERLAY (Existing - Unchanged)                            │
│  ───────────────────────────────────────────────                            │
│                                                                             │
│  Captions are overlaid on the final concatenated video                      │
│  Position remains consistent regardless of layout changes                   │
│  Word-by-word highlighting works across all segments                        │
│                                                                             │
│  Output: Final clip with captions                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### New Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class LayoutType(str, Enum):
    """Available layout types for clip rendering."""
    SCREEN_SHARE = "screen_share"      # 50/50 split - screen top, face bottom
    TALKING_HEAD = "talking_head"      # Face fills frame (9:16 crop)
    FULL_SCREEN = "full_screen"        # Content only, no face region
    PICTURE_IN_PICTURE = "pip"         # Screen with small face overlay (future)


@dataclass
class FrameLayoutAnalysis:
    """Layout analysis for a single frame."""
    timestamp_ms: int

    # Face metrics
    face_detected: bool
    face_area_ratio: float          # 0.0 - 1.0 (face area / frame area)
    face_is_centered: bool          # True if face in middle third
    face_is_corner: bool            # True if face in corner (webcam pattern)
    face_center: Optional[tuple[int, int]]  # (x, y) if detected

    # Content metrics
    edge_density: float             # 0.0 - 1.0 (high = UI/text content)
    webcam_overlay_detected: bool

    # Classification result
    recommended_layout: LayoutType
    layout_confidence: float        # 0.0 - 1.0


@dataclass
class LayoutSegment:
    """A time segment with a specific layout."""
    start_time_ms: int
    end_time_ms: int
    layout_type: LayoutType
    confidence: float               # Average confidence for this segment

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms


@dataclass
class LayoutTimeline:
    """Complete layout timeline for a clip."""
    segments: list[LayoutSegment]
    total_duration_ms: int
    dominant_layout: LayoutType     # Most common layout by duration

    def get_layout_at(self, timestamp_ms: int) -> LayoutType:
        """Get the layout type at a specific timestamp."""
        for segment in self.segments:
            if segment.start_time_ms <= timestamp_ms < segment.end_time_ms:
                return segment.layout_type
        return self.dominant_layout
```

### Modified RenderRequest

```python
@dataclass
class RenderRequest:
    """Request for rendering a clip."""
    video_path: str
    output_path: str
    start_time_ms: int
    end_time_ms: int
    source_width: int
    source_height: int

    # NEW: Layout timeline for dynamic switching
    layout_timeline: Optional[LayoutTimeline] = None

    # DEPRECATED: Single layout (kept for backwards compatibility)
    layout_type: Literal["talking_head", "screen_share"] = "screen_share"

    # Crop timelines (used per-segment now)
    face_timeline: Optional[CropTimeline] = None
    screen_timeline: Optional[CropTimeline] = None

    # Captions
    transcript_segments: Optional[list[TranscriptSegment]] = None
    caption_style: Optional[CaptionStyle] = None
```

---

## Implementation Plan

### Phase 1: Layout Classifier (2 hours)

**File:** `app/services/layout_classifier.py` (new)

```python
class LayoutClassifier:
    """Classifies video frames into layout types."""

    def __init__(self):
        self.min_face_ratio_talking_head = 0.15
        self.max_face_ratio_screen_share = 0.12
        self.min_face_ratio_visible = 0.03
        self.min_edge_density_screen = 0.08

    def classify_frame(
        self,
        face_detections: list[dict],
        frame: Optional[np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> FrameLayoutAnalysis:
        """Classify a single frame's optimal layout."""
        # Calculate face metrics
        face_area_ratio = self._calculate_face_area_ratio(...)
        face_is_centered = self._is_face_centered(...)
        face_is_corner = self._is_face_in_corner(...)

        # Calculate content metrics
        edge_density = self._calculate_edge_density(frame) if frame else 0.0
        webcam_detected = face_is_corner and face_area_ratio < 0.12

        # Classification logic
        layout, confidence = self._classify(
            face_area_ratio,
            face_is_centered,
            face_is_corner,
            edge_density,
            webcam_detected,
        )

        return FrameLayoutAnalysis(...)

    def _classify(self, ...) -> tuple[LayoutType, float]:
        """Core classification decision tree."""
        if face_area_ratio > self.min_face_ratio_talking_head and face_is_centered:
            return LayoutType.TALKING_HEAD, 0.9

        if face_area_ratio < self.min_face_ratio_visible:
            return LayoutType.FULL_SCREEN, 0.8

        if webcam_detected:
            return LayoutType.SCREEN_SHARE, 0.95

        if edge_density > self.min_edge_density_screen:
            return LayoutType.SCREEN_SHARE, 0.7

        return LayoutType.SCREEN_SHARE, 0.5  # Safe default
```

### Phase 2: Temporal Segmentation (4 hours)

**File:** `app/services/layout_segmenter.py` (new)

```python
class LayoutSegmenter:
    """Groups frame layouts into temporal segments."""

    def __init__(self):
        self.min_segment_duration_ms = 1500  # 1.5 seconds
        self.hysteresis_frames = 3           # Frames needed to trigger switch

    def build_timeline(
        self,
        frame_analyses: list[FrameLayoutAnalysis],
    ) -> LayoutTimeline:
        """Build layout timeline from frame analyses."""
        # Step 1: Group consecutive same-layout frames
        raw_segments = self._group_consecutive(frame_analyses)

        # Step 2: Apply hysteresis (require N consecutive frames)
        stable_segments = self._apply_hysteresis(raw_segments)

        # Step 3: Merge short segments
        merged_segments = self._merge_short_segments(stable_segments)

        # Step 4: Calculate dominant layout
        dominant = self._find_dominant_layout(merged_segments)

        return LayoutTimeline(
            segments=merged_segments,
            total_duration_ms=frame_analyses[-1].timestamp_ms,
            dominant_layout=dominant,
        )

    def _merge_short_segments(
        self,
        segments: list[LayoutSegment],
    ) -> list[LayoutSegment]:
        """Merge segments shorter than minimum duration."""
        result = []
        for segment in segments:
            if segment.duration_ms < self.min_segment_duration_ms:
                # Merge with neighbor (prefer higher confidence)
                if result and result[-1].confidence >= segment.confidence:
                    result[-1].end_time_ms = segment.end_time_ms
                elif result:
                    result[-1].end_time_ms = segment.end_time_ms
                # else: keep as-is if first segment
            else:
                result.append(segment)
        return result
```

### Phase 3: Segment-Based Renderer (6 hours)

**File:** `app/services/rendering_service.py` (modified)

```python
async def render_clip(self, request: RenderRequest) -> RenderResult:
    """Render a clip with optional dynamic layout switching."""

    # NEW: Check for dynamic layout timeline
    if request.layout_timeline and len(request.layout_timeline.segments) > 1:
        return await self._render_dynamic_layout(request)

    # Existing single-layout rendering
    return await self._render_single_layout(request)


async def _render_dynamic_layout(self, request: RenderRequest) -> RenderResult:
    """Render clip with multiple layout segments."""
    temp_segments = []

    try:
        for i, segment in enumerate(request.layout_timeline.segments):
            segment_path = os.path.join(
                os.path.dirname(request.output_path),
                f"temp_segment_{i}_{segment.layout_type.value}.mp4"
            )

            # Create sub-request for this segment
            segment_request = self._create_segment_request(request, segment)

            # Render with appropriate layout
            if segment.layout_type == LayoutType.SCREEN_SHARE:
                await self._render_opusclip_mode(segment_request, None, segment.duration_ms)
            elif segment.layout_type == LayoutType.TALKING_HEAD:
                await self._render_focus_mode(segment_request, None, segment.duration_ms)
            elif segment.layout_type == LayoutType.FULL_SCREEN:
                await self._render_static(segment_request, None, segment.duration_ms)

            temp_segments.append(segment_path)

        # Concatenate all segments
        await self._concat_segments(temp_segments, request.output_path)

        # Overlay captions on final video
        if request.transcript_segments:
            await self._overlay_captions(request)

    finally:
        # Cleanup temp files
        for path in temp_segments:
            if os.path.exists(path):
                os.remove(path)

    return RenderResult(...)


async def _concat_segments(
    self,
    segment_paths: list[str],
    output_path: str,
) -> None:
    """Concatenate video segments using FFmpeg."""
    # Create concat list file
    concat_list_path = output_path.replace(".mp4", "_concat.txt")
    with open(concat_list_path, "w") as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path,
    ]

    await self._run_cmd(cmd)
    os.remove(concat_list_path)
```

### Phase 4: Pipeline Integration (2 hours)

**File:** `app/services/ai_clipping_pipeline.py` (modified)

```python
async def _process_clip(self, clip_plan: ClipPlanSegment, ...) -> ClipResult:
    """Process a single clip with dynamic layout detection."""

    # Existing: Run detection
    detection_frames = await self._run_detection(...)

    # NEW: Classify layouts per frame
    frame_analyses = []
    for frame in detection_frames:
        analysis = self.layout_classifier.classify_frame(
            face_detections=frame.faces,
            frame=frame.image,  # May be None
            frame_width=source_width,
            frame_height=source_height,
        )
        frame_analyses.append(analysis)

    # NEW: Build layout timeline
    layout_timeline = self.layout_segmenter.build_timeline(frame_analyses)

    logger.info(
        f"Layout timeline: {len(layout_timeline.segments)} segments, "
        f"dominant={layout_timeline.dominant_layout.value}"
    )

    # Existing: Build crop timelines (now per-segment aware)
    face_timeline = self._build_crop_timeline(...)
    screen_timeline = self._build_screen_timeline(...)

    # Render with dynamic layout
    render_request = RenderRequest(
        layout_timeline=layout_timeline,  # NEW
        face_timeline=face_timeline,
        screen_timeline=screen_timeline,
        ...
    )

    return await self.rendering_service.render_clip(render_request)
```

---

## Configuration

### New Settings (app/config.py)

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # ============================================================
    # SCENE SWITCHING SETTINGS
    # ============================================================

    # Enable/disable dynamic layout switching
    enable_scene_switching: bool = True

    # Minimum segment duration (prevents flickering)
    min_segment_duration_ms: int = 1500  # 1.5 seconds

    # Hysteresis: frames needed to trigger layout switch
    layout_switch_hysteresis: int = 3

    # Layout classification thresholds
    talking_head_min_face_ratio: float = 0.15   # Face > 15% = talking head
    screen_share_max_face_ratio: float = 0.12   # Face < 12% in corner = screen share
    full_screen_max_face_ratio: float = 0.03    # Face < 3% = full screen
    screen_content_min_edge_density: float = 0.08  # High edges = UI/text

    # Transition style (future)
    # segment_transition: Literal["cut", "crossfade"] = "cut"
    # crossfade_duration_ms: int = 100
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_layout_classifier.py

def test_talking_head_detection():
    """Large centered face should classify as TALKING_HEAD."""
    classifier = LayoutClassifier()

    face_detections = [{
        "bbox": {"x": 400, "y": 200, "width": 300, "height": 400},
        "confidence": 0.95,
    }]

    result = classifier.classify_frame(
        face_detections=face_detections,
        frame=None,
        frame_width=1920,
        frame_height=1080,
    )

    assert result.recommended_layout == LayoutType.TALKING_HEAD
    assert result.layout_confidence > 0.8


def test_screen_share_detection():
    """Small corner face should classify as SCREEN_SHARE."""
    classifier = LayoutClassifier()

    face_detections = [{
        "bbox": {"x": 1700, "y": 900, "width": 150, "height": 150},
        "confidence": 0.9,
    }]

    result = classifier.classify_frame(
        face_detections=face_detections,
        frame=None,
        frame_width=1920,
        frame_height=1080,
    )

    assert result.recommended_layout == LayoutType.SCREEN_SHARE
    assert result.webcam_overlay_detected == True


def test_full_screen_detection():
    """No face should classify as FULL_SCREEN."""
    classifier = LayoutClassifier()

    result = classifier.classify_frame(
        face_detections=[],
        frame=None,
        frame_width=1920,
        frame_height=1080,
    )

    assert result.recommended_layout == LayoutType.FULL_SCREEN
```

### Integration Tests

```python
# tests/test_scene_switching.py

async def test_dynamic_layout_rendering():
    """Test rendering a clip with multiple layout segments."""
    # Create mock layout timeline
    timeline = LayoutTimeline(
        segments=[
            LayoutSegment(0, 2000, LayoutType.SCREEN_SHARE, 0.9),
            LayoutSegment(2000, 5000, LayoutType.TALKING_HEAD, 0.85),
            LayoutSegment(5000, 8000, LayoutType.SCREEN_SHARE, 0.9),
        ],
        total_duration_ms=8000,
        dominant_layout=LayoutType.SCREEN_SHARE,
    )

    request = RenderRequest(
        video_path="test_video.mp4",
        output_path="output.mp4",
        layout_timeline=timeline,
        ...
    )

    result = await rendering_service.render_clip(request)

    assert os.path.exists(result.output_path)
    # Verify output duration matches expected
```

---

## Edge Cases & Handling

| Edge Case | Handling |
|-----------|----------|
| Single-frame layout change | Hysteresis prevents (requires 3+ frames) |
| Very short clip (< 2s) | Use single dominant layout, skip segmentation |
| No faces detected entire clip | Use FULL_SCREEN layout |
| Rapid scene changes | Minimum 1.5s segment duration enforced |
| Layout confidence tie | Prefer SCREEN_SHARE as safe default |
| Audio sync across segments | FFmpeg concat preserves sync |
| Caption timing across segments | Captions overlaid on final concatenated video |

---

## Performance Considerations

### Current Baseline
- Detection: ~2 FPS sampling (already implemented)
- Single layout render: ~5-10 seconds per 30s clip

### With Scene Switching
- Layout classification: +50ms per frame (negligible)
- Temporal segmentation: +10ms total (negligible)
- Per-segment rendering: Same as current (parallel potential)
- Concatenation: +1-2 seconds (fast stream copy)

### Estimated Total Overhead: **+2-3 seconds per clip**

---

## Future Enhancements

### 1. Smooth Transitions
- Crossfade between layout changes (0.1-0.2s)
- Scene-change detection for natural cut points

### 2. AI-Powered Classification
- Train ML model on labeled layout data
- Use video understanding models for context

### 3. User Layout Preferences
- API parameter to force specific layout
- Per-user default layout preferences

### 4. Picture-in-Picture Layout
- Small speaker overlay on screen content
- Configurable overlay position/size

### 5. Multi-Speaker Support
- Detect multiple speakers
- Switch focus based on who's talking

---

## Summary

| Component | Status | Effort |
|-----------|--------|--------|
| Layout Classifier | New | 2 hours |
| Temporal Segmentation | New | 4 hours |
| Segment-Based Renderer | Modified | 6 hours |
| Pipeline Integration | Modified | 2 hours |
| Testing | New | 4 hours |
| **Total** | | **~18 hours** |

The implementation leverages existing infrastructure (face detection, content analysis, multiple render modes) and adds two new services (classifier, segmenter) plus modifications to the renderer. The result is automatic, intelligent layout switching that adapts to video content in real-time.
