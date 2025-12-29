"""
Rendering Service - FFmpeg-based video rendering with dynamic cropping and captions.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from app.config import CaptionStyle, get_settings
from app.services.caption_generator import CaptionGeneratorService
from app.services.transcription_service import TranscriptSegment

logger = logging.getLogger(__name__)


# Boundary padding to avoid edge-case FFmpeg issues with crops at exact boundaries
BOUNDARY_PADDING = 2


@dataclass
class CropKeyframe:
    """A keyframe for dynamic crop position."""
    
    timestamp_ms: int
    center_x: int  # Pixel position
    center_y: int  # Pixel position


@dataclass
class CropTimeline:
    """Timeline for dynamic camera movement."""
    
    window_width: int
    window_height: int
    source_width: int
    source_height: int
    keyframes: list[CropKeyframe]


@dataclass
class LayoutKeyframe:
    """A keyframe defining layout at a specific timestamp for timeline-based rendering.
    
    Enables users to set different layouts and crop boxes at specific timestamps
    within a clip, allowing complex videos with multiple layout transitions.
    """
    timestamp_ms: int
    layout_type: Literal["talking_head", "screen_share", "podcast"]
    face_bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) for facecam
    podcast_bboxes: Optional[list[tuple[int, int, int, int]]] = None  # For podcast mode


@dataclass
class RenderRequest:
    """Request for rendering a clip."""
    
    video_path: str
    output_path: str
    start_time_ms: int
    end_time_ms: int
    source_width: int
    source_height: int
    layout_type: Literal["talking_head", "screen_share", "podcast"]
    
    # Crop timelines for dynamic panning
    primary_timeline: Optional[CropTimeline] = None
    secondary_timeline: Optional[CropTimeline] = None  # For stack mode
    
    # Captions
    transcript_segments: Optional[list[TranscriptSegment]] = None
    caption_style: Optional[CaptionStyle] = None
    
    # Embedded facecam flag - if True, video already has PiP facecam
    # In this case, we should NOT do split screen to avoid duplicating the face
    has_embedded_facecam: bool = False
    
    # Corner facecam bbox (x, y, w, h) - for split screen layouts
    # This tells the renderer WHERE the corner webcam is, so it crops that instead of main face
    corner_facecam_bbox: Optional[tuple[int, int, int, int]] = None
    
    # Flag to indicate if the bbox was user-drawn (Custom mode) vs auto-detected
    # User-drawn boxes should use minimal padding; auto-detected need expansion
    is_custom_bbox: bool = False
    
    # For podcast layout: list of 2 face bboxes for side-by-side rendering
    podcast_face_bboxes: Optional[list[tuple[int, int, int, int]]] = None
    
    # Timeline keyframes for multi-layout clips (Advanced Custom Mode)
    # When provided, the clip is rendered in segments with different layouts at each keyframe
    layout_keyframes: Optional[list[LayoutKeyframe]] = None



@dataclass
class RenderResult:
    """Result of rendering operation."""
    
    output_path: str
    file_size_bytes: int
    duration_ms: int


class RenderingService:
    """
    Service for rendering clips using FFmpeg.
    
    Features:
    - Dynamic cropping with interpolated keyframes
    - Focus mode (single pan) and Stack mode (split-screen)
    - ASS caption burning
    - H.264 output optimized for social media
    """

    def __init__(self):
        self.settings = get_settings()
        self.caption_generator = CaptionGeneratorService()
        self._verify_ffmpeg()
        
        # Get fonts directory path (relative to project root)
        self.fonts_dir = self._get_fonts_dir()

    def _get_fonts_dir(self) -> str:
        """Get the path to the project's fonts directory."""
        # Get the project root (parent of app directory)
        current_file = Path(__file__)  # app/services/rendering_service.py
        project_root = current_file.parent.parent.parent  # Go up 3 levels to project root
        fonts_path = project_root / "fonts"
        
        if fonts_path.exists():
            logger.debug(f"Fonts directory found: {fonts_path}")
            return str(fonts_path)
        else:
            logger.warning(f"Fonts directory not found at {fonts_path}, using system fonts")
            return ""

    def _verify_ffmpeg(self):
        """Verify ffmpeg is available."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH")
        logger.info("FFmpeg available")


    async def render_clip(self, request: RenderRequest) -> RenderResult:
        """
        Render a clip with dynamic cropping and captions.
        
        Args:
            request: RenderRequest with all rendering parameters
            
        Returns:
            RenderResult with output path and metadata
        """
        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
        
        duration_ms = request.end_time_ms - request.start_time_ms
        if duration_ms <= 0:
            raise RenderingError("Invalid clip duration")
        
        logger.info(
            f"Rendering clip: {request.start_time_ms}ms-{request.end_time_ms}ms, "
            f"layout={request.layout_type}"
        )
        
        # Generate captions if transcript available
        caption_path: Optional[str] = None
        if request.transcript_segments:
            caption_filename = f"clip-{request.start_time_ms}-{request.end_time_ms}.ass"
            caption_path = os.path.join(
                os.path.dirname(request.output_path),
                caption_filename,
            )
            caption_path = await self.caption_generator.generate_captions(
                transcript_segments=request.transcript_segments,
                clip_start_ms=request.start_time_ms,
                clip_end_ms=request.end_time_ms,
                output_path=caption_path,
                caption_style=request.caption_style,
                layout_type=request.layout_type,  # Dynamic position: center for split-screen, bottom for talking head
            )

        
        # Choose rendering method based on layout and timelines
        # Priority: layout_keyframes > podcast > split > focus > static
        
        if request.layout_keyframes and len(request.layout_keyframes) > 0:
            # Timeline Keyframes: Multi-layout rendering with transitions
            logger.info(f"Using timeline keyframes mode ({len(request.layout_keyframes)} keyframes)")
            await self._render_with_keyframes(request, caption_path, duration_ms)
        elif request.layout_type == "podcast" and request.podcast_face_bboxes:
            # Podcast: side-by-side 2-up layout for multi-speaker videos
            logger.info("Using podcast mode (side-by-side 2-up)")
            await self._render_podcast_mode(request, caption_path, duration_ms)
        elif (
            request.layout_type == "screen_share"
            and request.primary_timeline
            and request.secondary_timeline
        ):
            # Use split layout (screen top 50%, face bottom 50%, captions overlaid)
            if self.settings.use_split_layout:
                await self._render_split_mode(request, caption_path, duration_ms)
            else:
                # Fallback to standard stack mode
                await self._render_stack_mode(request, caption_path, duration_ms)
        elif (
            request.layout_type == "screen_share"
            and request.corner_facecam_bbox
            and self.settings.use_split_layout
        ):
            # NEW: Split screen using corner_facecam_bbox (Direct API path)
            # This handles the case where we have bbox but no tracking timelines
            logger.info("Using split mode with static bbox (Direct API)")
            await self._render_split_mode_static(request, caption_path, duration_ms)
        elif request.layout_type == "talking_head" and request.primary_timeline:
            # Talking head: single 9:16 full-height crop following the face
            await self._render_focus_mode(request, caption_path, duration_ms)
        elif request.primary_timeline:
            # Fallback focus mode for any other case with timeline
            await self._render_focus_mode(request, caption_path, duration_ms)
        else:
            # Static fallback
            await self._render_static(request, caption_path, duration_ms)
        
        # Verify output and get file size
        if not os.path.isfile(request.output_path):
            raise RenderingError(f"Render failed: output file not created")
        
        file_size = os.path.getsize(request.output_path)
        logger.info(
            f"Clip rendered: {request.output_path} ({file_size / 1024 / 1024:.1f} MB)"
        )
        
        # Cleanup caption file
        if caption_path and os.path.isfile(caption_path):
            try:
                os.remove(caption_path)
            except Exception:
                pass
        
        return RenderResult(
            output_path=request.output_path,
            file_size_bytes=file_size,
            duration_ms=duration_ms,
        )

    async def _get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Get actual video dimensions using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            video_path,
        ]
        try:
            # Use run_in_executor for Windows compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True)
            )
            dims = result.stdout.decode().strip().split("x")
            return int(dims[0]), int(dims[1])
        except Exception as e:
            logger.warning(f"Failed to probe video dimensions: {e}")
            return 0, 0

    async def _render_split_mode(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render split layout with 2 regions and overlaid captions.

        Layout (top to bottom):
        - Screen content: 50% (top)
        - Speaker face: 50% (bottom) - tight face-centered crop
        - Captions: Overlaid on top of combined video (no black band)
        """
        face_timeline = request.primary_timeline  # Face tracking
        screen_timeline = request.secondary_timeline  # Screen content
        assert face_timeline and screen_timeline

        # Get actual video dimensions to ensure crop is valid
        actual_width, actual_height = await self._get_video_dimensions(request.video_path)
        if actual_width > 0 and actual_height > 0:
            # Update timelines with actual dimensions if they differ
            if face_timeline.source_width != actual_width or face_timeline.source_height != actual_height:
                logger.warning(
                    f"Timeline dimensions mismatch: timeline={face_timeline.source_width}x{face_timeline.source_height}, "
                    f"actual={actual_width}x{actual_height}. Using actual dimensions."
                )
                face_timeline.source_width = actual_width
                face_timeline.source_height = actual_height
                screen_timeline.source_width = actual_width
                screen_timeline.source_height = actual_height

        # Calculate region heights for split layout (2 regions, no caption band)
        total_height = self.settings.target_output_height
        screen_height = int(total_height * self.settings.split_screen_ratio)
        face_height = total_height - screen_height  # Remaining space for face

        logger.info(
            f"Split mode render: source={face_timeline.source_width}x{face_timeline.source_height}, "
            f"screen={screen_height}px, face={face_height}px (captions overlaid)"
        )

        output_dir = os.path.dirname(request.output_path)
        timestamp = os.path.basename(request.output_path).replace(".mp4", "")
        temp_screen_path = os.path.join(output_dir, f"temp_screen_{timestamp}.mp4")
        temp_face_path = os.path.join(output_dir, f"temp_face_{timestamp}.mp4")

        try:
            # Pass 1: Render screen content (TOP 50%)
            # Square center crop, scaled to fill full width
            logger.debug("Pass 1: Rendering screen content (top)...")
            await self._render_split_screen(
                input_path=request.video_path,
                output_path=temp_screen_path,
                timeline=screen_timeline,
                target_width=self.settings.target_output_width,
                target_height=screen_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )

            # Pass 2: Render tight face crop (BOTTOM 50%)
            # Use corner_facecam_bbox if available - this is the ACTUAL webcam position
            logger.debug("Pass 2: Rendering face crop (bottom)...")
            await self._render_split_face(
                input_path=request.video_path,
                output_path=temp_face_path,
                timeline=face_timeline,
                target_width=self.settings.target_output_width,
                target_height=face_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
                corner_facecam_bbox=request.corner_facecam_bbox,
            )

            # Pass 3: Merge 2 regions with vstack and overlay captions
            logger.debug("Pass 3: Merging 2 regions with captions overlaid...")
            await self._vstack_split_with_overlay(
                screen_path=temp_screen_path,
                face_path=temp_face_path,
                output_path=request.output_path,
                caption_path=caption_path,
                original_video_path=request.video_path,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )

        finally:
            # Cleanup temp files
            for path in [temp_screen_path, temp_face_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _render_split_mode_static(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render split layout using only corner_facecam_bbox (no tracking timelines).
        
        This is used by the Direct API where we skip full detection for speed,
        but still have the detected webcam bounding box.
        
        Layout (top to bottom):
        - Screen content: 50% (top) - static center crop
        - Speaker face: 50% (bottom) - cropped from corner_facecam_bbox
        - Captions: Overlaid on top of combined video
        """
        assert request.corner_facecam_bbox, "corner_facecam_bbox required for static split mode"
        
        source_w = request.source_width
        source_h = request.source_height
        
        # Get actual video dimensions
        actual_width, actual_height = await self._get_video_dimensions(request.video_path)
        if actual_width > 0 and actual_height > 0:
            source_w = actual_width
            source_h = actual_height
        
        # Calculate region heights for split layout
        total_height = self.settings.target_output_height
        screen_height = int(total_height * self.settings.split_screen_ratio)
        face_height = total_height - screen_height
        
        logger.info(
            f"Static split mode: source={source_w}x{source_h}, "
            f"screen={screen_height}px, face={face_height}px, bbox={request.corner_facecam_bbox}"
        )
        
        output_dir = os.path.dirname(request.output_path)
        timestamp = os.path.basename(request.output_path).replace(".mp4", "")
        temp_screen_path = os.path.join(output_dir, f"temp_screen_{timestamp}.mp4")
        temp_face_path = os.path.join(output_dir, f"temp_face_{timestamp}.mp4")
        
        try:
            # Pass 1: Render screen content (TOP 50%) - static center crop
            logger.debug("Static split - Pass 1: Rendering screen content...")
            await self._render_static_screen_crop(
                input_path=request.video_path,
                output_path=temp_screen_path,
                source_width=source_w,
                source_height=source_h,
                target_width=self.settings.target_output_width,
                target_height=screen_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
            # Pass 2: Render face crop (BOTTOM 50%) - from corner_facecam_bbox  
            logger.debug("Static split - Pass 2: Rendering face from bbox...")
            await self._render_static_face_crop(
                input_path=request.video_path,
                output_path=temp_face_path,
                source_width=source_w,
                source_height=source_h,
                corner_facecam_bbox=request.corner_facecam_bbox,
                target_width=self.settings.target_output_width,
                target_height=face_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
                is_custom_bbox=request.is_custom_bbox,  # Use minimal padding for custom boxes
            )

            
            # Pass 3: Merge with vstack and overlay captions
            logger.debug("Static split - Pass 3: Merging with captions...")
            await self._vstack_split_with_overlay(
                screen_path=temp_screen_path,
                face_path=temp_face_path,
                output_path=request.output_path,
                caption_path=caption_path,
                original_video_path=request.video_path,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
        finally:
            for path in [temp_screen_path, temp_face_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _render_podcast_mode(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render podcast layout with 2 speakers stacked top/bottom.

        Layout (9:16 / 1080x1920):
        +----------------------+
        |       Face 1         |  <- Top half, 1080x960
        |      (960px)         |
        +----------------------+
        |       Face 2         |  <- Bottom half, 1080x960
        |      (960px)         |
        +----------------------+
        |     CAPTIONS         |
        +----------------------+
        """
        assert request.podcast_face_bboxes and len(request.podcast_face_bboxes) >= 2, \
            "podcast_face_bboxes requires at least 2 faces"
        
        source_w = request.source_width
        source_h = request.source_height
        
        # Get actual video dimensions
        actual_width, actual_height = await self._get_video_dimensions(request.video_path)
        if actual_width > 0 and actual_height > 0:
            source_w = actual_width
            source_h = actual_height
        
        # Each speaker gets half the height (960px each for 1920px total)
        panel_width = self.settings.target_output_width   # 1080px full width
        panel_height = self.settings.target_output_height // 2  # 960px each
        
        logger.info(
            f"Podcast mode (vstack): source={source_w}x{source_h}, "
            f"panels={panel_width}x{panel_height} each, "
            f"faces={request.podcast_face_bboxes}"
        )
        
        output_dir = os.path.dirname(request.output_path)
        timestamp = os.path.basename(request.output_path).replace(".mp4", "")
        temp_face1_path = os.path.join(output_dir, f"temp_podcast_face1_{timestamp}.mp4")
        temp_face2_path = os.path.join(output_dir, f"temp_podcast_face2_{timestamp}.mp4")
        
        try:
            # Pass 1: Render Face 1 (top panel)
            bbox1 = request.podcast_face_bboxes[0]
            await self._render_podcast_panel(
                input_path=request.video_path,
                output_path=temp_face1_path,
                face_bbox=bbox1,
                source_width=source_w,
                source_height=source_h,
                target_width=panel_width,
                target_height=panel_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
            # Pass 2: Render Face 2 (bottom panel)
            bbox2 = request.podcast_face_bboxes[1]
            await self._render_podcast_panel(
                input_path=request.video_path,
                output_path=temp_face2_path,
                face_bbox=bbox2,
                source_width=source_w,
                source_height=source_h,
                target_width=panel_width,
                target_height=panel_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
            # Pass 3: Merge top/bottom with vstack and overlay captions
            await self._vstack_podcast_with_captions(
                top_path=temp_face1_path,
                bottom_path=temp_face2_path,
                output_path=request.output_path,
                caption_path=caption_path,
                original_video_path=request.video_path,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
        finally:
            for path in [temp_face1_path, temp_face2_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _render_podcast_panel(
        self,
        input_path: str,
        output_path: str,
        face_bbox: tuple[int, int, int, int],
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Render a single podcast panel (one speaker) with face-centered crop."""
        bbox_x, bbox_y, bbox_w, bbox_h = face_bbox
        
        # Calculate crop centered on face with aspect ratio matching target
        # For vstack: target is 1080x960, aspect ratio = 1.125 (wider than tall)
        target_aspect = target_width / target_height
        
        # Start with face size and add generous padding (2.5x for head + shoulders)
        padding = 2.5
        base_size = int(max(bbox_w, bbox_h) * padding)
        
        # Calculate crop dimensions to match target aspect ratio
        if target_aspect >= 1:
            # Target is wider than tall (our case: 1080x960 = 1.125)
            crop_height = base_size
            crop_width = int(base_size * target_aspect)
        else:
            crop_width = base_size
            crop_height = int(base_size / target_aspect)
        
        # Ensure minimum crop for quality (at least 400px height for faces)
        min_height = 400
        if crop_height < min_height:
            scale_factor = min_height / crop_height
            crop_height = min_height
            crop_width = int(crop_width * scale_factor)
        
        # Center crop on face center
        center_x = bbox_x + bbox_w / 2
        center_y = bbox_y + bbox_h / 2
        crop_x = int(center_x - crop_width / 2)
        crop_y = int(center_y - crop_height / 2)
        
        # Clamp to source bounds
        crop_x = max(0, min(crop_x, source_width - crop_width))
        crop_y = max(0, min(crop_y, source_height - crop_height))
        crop_width = min(crop_width, source_width)
        crop_height = min(crop_height, source_height)
        
        logger.info(
            f"Podcast panel: bbox=({bbox_x},{bbox_y},{bbox_w}x{bbox_h}), "
            f"crop={crop_width}x{crop_height} at ({crop_x},{crop_y}) -> {target_width}x{target_height}"
        )
        
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=disable",
            "setsar=1",
        ]
        
        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _vstack_podcast_with_captions(
        self,
        top_path: str,
        bottom_path: str,
        output_path: str,
        caption_path: Optional[str],
        original_video_path: str,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Merge two podcast panels top/bottom and add captions + audio."""
        # Build filter complex for vstack + captions
        filter_complex = "[0:v][1:v]vstack=inputs=2[stacked]"
        
        if caption_path and os.path.isfile(caption_path):
            escaped_caption = self._escape_filter_path(caption_path)
            filter_complex += f";[stacked]ass={escaped_caption}[final]"
            map_video = "[final]"
        else:
            map_video = "[stacked]"
        
        # Build FFmpeg command
        start_seconds = start_time_ms / 1000.0
        duration_seconds = duration_ms / 1000.0
        
        cmd = [
            "ffmpeg", "-y",
            "-i", top_path,
            "-i", bottom_path,
            "-ss", str(start_seconds),
            "-t", str(duration_seconds),
            "-i", original_video_path,  # For audio
            "-filter_complex", filter_complex,
            "-map", map_video,
            "-map", "2:a",  # Audio from original
            "-c:v", "libx264",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        logger.debug(f"Podcast vstack command: {' '.join(cmd)}")
        
        # Use run_in_executor for Windows compatibility
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Podcast vstack failed: {error_msg}")
            raise RenderingError(f"Podcast merge failed: {error_msg[:200]}")

    async def _render_static_screen_crop(
        self,
        input_path: str,
        output_path: str,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Render screen content with static centered 1:1 square crop."""
        # Take a 1:1 square crop from center-top (avoid webcam in corner)
        screen_region_height = int(source_height * 0.70)
        square_size = min(source_width, screen_region_height)
        
        crop_x = (source_width - square_size) // 2
        crop_y = 0  # Start from top
        
        logger.info(
            f"Static screen crop: 1:1 square={square_size}x{square_size} at ({crop_x}, {crop_y})"
        )
        
        filters = [
            f"crop={square_size}:{square_size}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=disable",
            "setsar=1",
        ]
        
        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _render_static_face_crop(
        self,
        input_path: str,
        output_path: str,
        source_width: int,
        source_height: int,
        corner_facecam_bbox: tuple[int, int, int, int],
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
        is_custom_bbox: bool = False,  # True if user-drawn in Custom mode
    ) -> None:
        """Render face crop from corner_facecam_bbox.
        
        For user-drawn custom boxes: minimal padding (1.5x) to respect selection.
        For auto-detected boxes: larger expansion (3x) to show full head/shoulders.
        """
        bbox_x, bbox_y, bbox_w, bbox_h = corner_facecam_bbox
        
        # Target aspect ratio for bottom panel (e.g., 1080/960 = 1.125)
        target_aspect = target_width / target_height
        bbox_aspect = bbox_w / bbox_h if bbox_h > 0 else target_aspect
        
        # PADDING DECISION: Check is_custom_bbox flag FIRST, then fall back to size heuristic
        if is_custom_bbox:
            # User drew this box manually - use EXACT selection (no expansion)
            padding_factor = 1.0  # No padding - respect exact user selection
            logger.info(f"Custom-drawn bbox, using exact selection (1.0x, no padding)")

        else:
            # Auto-detected box - use size heuristic to decide padding
            frame_area = source_width * source_height
            bbox_area = bbox_w * bbox_h
            bbox_area_ratio = bbox_area / frame_area if frame_area > 0 else 0
            
            if bbox_area_ratio < 0.05:  # Small bbox = auto-detected corner webcam
                padding_factor = 3.0  # 3x expansion for auto-detected webcams
                logger.info(f"Auto-detected webcam bbox ({bbox_area_ratio:.2%} of frame), using 3x expansion")
            else:
                # Larger auto-detected bbox
                padding_factor = 1.5
                logger.info(f"Large auto-detected bbox ({bbox_area_ratio:.2%} of frame), using 1.5x padding")

        
        crop_w = int(bbox_w * padding_factor)
        crop_h = int(bbox_h * padding_factor)
        
        # Adjust crop to match target aspect ratio while preserving user's selection
        if bbox_aspect > target_aspect:
            # User's selection is wider than target - expand height to match
            crop_h = int(crop_w / target_aspect)
        else:
            # User's selection is taller than target - expand width to match
            crop_w = int(crop_h * target_aspect)

        
        # Center the crop on the user's bbox center
        center_x = bbox_x + bbox_w / 2
        center_y = bbox_y + bbox_h / 2
        crop_x = int(center_x - crop_w / 2)
        crop_y = int(center_y - crop_h / 2)
        
        # Clamp to source bounds
        crop_x = max(0, min(crop_x, source_width - crop_w))
        crop_y = max(0, min(crop_y, source_height - crop_h))
        crop_w = min(crop_w, source_width)
        crop_h = min(crop_h, source_height)
        
        logger.info(
            f"Static face crop: bbox=({bbox_x},{bbox_y},{bbox_w}x{bbox_h}), "
            f"crop={crop_w}x{crop_h} at ({crop_x},{crop_y}) -> {target_width}x{target_height}"
        )
        
        filters = [
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=disable",
            "setsar=1",
        ]
        
        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _render_embedded_facecam_mode(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render video that already has an embedded facecam (picture-in-picture).

        This mode preserves the original screen layout with the embedded facecam
        by taking a full-width crop centered on the content. This avoids duplicating
        the face that would happen with split screen mode.

        The crop takes the full width of the source video and calculates the height
        to match the 9:16 aspect ratio, then centers the crop vertically.
        """
        source_w = request.source_width
        source_h = request.source_height
        target_width = self.settings.target_output_width
        target_height = self.settings.target_output_height

        # Get actual video dimensions to ensure crop is valid
        actual_width, actual_height = await self._get_video_dimensions(request.video_path)
        if actual_width > 0 and actual_height > 0:
            source_w = actual_width
            source_h = actual_height

        # Calculate crop dimensions to fit 9:16 while preserving full width
        target_aspect = target_width / target_height  # 9:16 = 0.5625

        # Take FULL WIDTH of source, calculate height for 9:16
        crop_width = source_w
        crop_height = int(source_w / target_aspect)

        # If calculated height exceeds source, use full height and adjust width
        if crop_height > source_h:
            crop_height = source_h
            crop_width = int(source_h * target_aspect)

        # Center the crop (slight bias toward upper portion for screen content)
        crop_x = max(0, (source_w - crop_width) // 2)
        crop_y = max(0, (source_h - crop_height) // 3)  # Bias toward top

        # Ensure crop stays within bounds
        crop_x = min(crop_x, source_w - crop_width)
        crop_y = min(crop_y, source_h - crop_height)

        logger.info(
            f"Embedded facecam mode: source={source_w}x{source_h}, "
            f"crop={crop_width}x{crop_height} at ({crop_x}, {crop_y}) "
            f"-> scale to {target_width}x{target_height}"
        )

        # Build filter chain: crop -> scale -> optional captions
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}",
        ]

        if caption_path and os.path.isfile(caption_path):
            filters.append(f"ass={self._escape_filter_path(caption_path)}")

        await self._run_ffmpeg(
            input_path=request.video_path,
            output_path=request.output_path,
            start_time_ms=request.start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
        )

    async def _render_split_screen(
        self,
        input_path: str,
        output_path: str,
        timeline: CropTimeline,
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """
        Render screen content for split layout (top region).

        For screen share content, we take a 1:1 SQUARE CENTERED crop
        from the screen content area, then scale it to fill the top region.
        
        Strategy: 
        1. Take a 1:1 square crop from CENTER of frame (screen content)
        2. Scale it to target width x target height for the top region
        """
        source_w = timeline.source_width
        source_h = timeline.source_height

        # Take a 1:1 SQUARE crop centered in the frame
        # The square size should be the smaller of width/height to fit
        # But we want to crop from screen area (not include corner facecam)
        # So we use ~70% of frame height to avoid webcam region
        screen_region_height = int(source_h * 0.70)
        
        # Square crop size - use screen region height to ensure we get screen content
        square_size = min(source_w, screen_region_height)
        
        # Center the square crop horizontally, position at top vertically
        crop_x = (source_w - square_size) // 2
        crop_y = 0  # Start from top
        
        crop_width = square_size
        crop_height = square_size

        # Ensure bounds
        crop_x = max(0, min(crop_x, source_w - crop_width))
        crop_y = max(0, min(crop_y, source_h - crop_height))

        logger.info(
            f"Split SCREEN crop: source={source_w}x{source_h}, "
            f"1:1 square={square_size}x{square_size} at ({crop_x}, {crop_y}) "
            f"-> scale to {target_width}x{target_height}"
        )

        # Crop 1:1 square from center-top, then scale to target dimensions
        # Use force_original_aspect_ratio=disable to ensure exact fill
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=disable",
            f"setsar=1",  # Ensure square pixels
        ]

        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _render_split_face(
        self,
        input_path: str,
        output_path: str,
        timeline: CropTimeline,
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
        corner_facecam_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Render face crop for split layout (bottom region).

        For screen recordings with webcam overlays:
        - The webcam is typically a SMALL box in a corner
        - We crop that region and SCALE UP to fill the entire bottom portion
        
        Strategy:
        1. Use corner_facecam_bbox if available (from smart layout detector)
        2. Otherwise fallback to timeline keyframes (which may have wrong face)
        3. Crop a rectangle around the face matching target aspect ratio
        4. Scale it up to fill the entire bottom half (1080x960)
        """
        source_w = timeline.source_width
        source_h = timeline.source_height
        
        # PRIORITY: Use corner_facecam_bbox from smart layout detector
        # This is the ACTUAL webcam overlay position, not the main face
        if corner_facecam_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = corner_facecam_bbox
            avg_center_x = bbox_x + bbox_w / 2
            avg_center_y = bbox_y + bbox_h / 2
            webcam_size = max(bbox_w, bbox_h)
            logger.info(
                f"Using corner facecam bbox: ({bbox_x}, {bbox_y}, {bbox_w}x{bbox_h}), "
                f"center=({avg_center_x:.0f}, {avg_center_y:.0f})"
            )
        else:
            # Fallback: Use keyframes (may be wrong if it detected main face)
            keyframes = timeline.keyframes
            avg_center_x = sum(k.center_x for k in keyframes) / len(keyframes)
            avg_center_y = sum(k.center_y for k in keyframes) / len(keyframes)
            
            # Estimate webcam size from timeline
            webcam_size = max(timeline.window_width, timeline.window_height)
            logger.warning(
                f"No corner_facecam_bbox available, using timeline keyframes: "
                f"center=({avg_center_x:.0f}, {avg_center_y:.0f})"
            )

        # Don't validate webcam_size from corner_facecam_bbox - trust the detection
        # Only apply defaults when we don't have corner_facecam_bbox
        if not corner_facecam_bbox:
            max_webcam_size = int(min(source_w, source_h) * 0.40)
            min_webcam_size = int(min(source_w, source_h) * 0.10)
            
            if webcam_size > max_webcam_size or webcam_size < min_webcam_size:
                # Use default webcam overlay size estimate
                webcam_size = int(min(source_w, source_h) * 0.25)
        
        # TIGHT FACE CROP: For small webcam overlays, we want to crop TIGHT around
        # the webcam box so it fills the bottom panel when scaled up
        target_aspect = target_width / target_height  # e.g., 1080/960 = 1.125
        
        # PRIORITY 1: If we have corner_facecam_bbox, use its size directly
        # This is the most reliable source for webcam overlay size
        if corner_facecam_bbox:
            # Use the webcam box size with padding to get context around face
            # Padding of 1.5x ensures we capture the head and some background
            tight_padding = 1.5
            base_size = int(webcam_size * tight_padding)
            logger.info(f"Using corner_facecam_bbox size: webcam={webcam_size}px -> base_size={base_size}px")
        else:
            # FALLBACK: Use timeline dimensions if available
            timeline_based_size = max(timeline.window_width, timeline.window_height)
            
            if timeline_based_size > 0 and timeline_based_size < source_h * 0.4:
                base_size = timeline_based_size
                logger.info(f"Using timeline-based face crop: {base_size}px (from detected faces)")
            else:
                # Final fallback: Calculate from estimated webcam size
                tight_padding = 1.2
                base_size = int(webcam_size * tight_padding)
                logger.info(f"Using webcam-based crop: webcam={webcam_size}, base_size={base_size}")
        
        # Minimum base_size for quality (at least 300px for better context around face)
        base_size = max(base_size, 300)
        
        # Maximum base_size - cap at 500px for reasonable zoom
        # This ensures we show some context around the face, not just extreme closeup
        base_size = min(base_size, 500)
        
        # Calculate crop dimensions matching target aspect ratio
        # The crop should be SMALL so when scaled up, face fills the output
        if target_aspect >= 1:
            # Output is wider than tall (1080x960 = 1.125 aspect)
            crop_height = base_size
            crop_width = int(base_size * target_aspect)
        else:
            # Output is taller than wide  
            crop_width = base_size
            crop_height = int(base_size / target_aspect)
        
        # Ensure minimum crop for quality (at least 150px in each dimension)
        min_crop = 150
        if crop_width < min_crop:
            scale_factor = min_crop / crop_width
            crop_width = min_crop
            crop_height = int(crop_height * scale_factor)
        if crop_height < min_crop:
            scale_factor = min_crop / crop_height
            crop_height = min_crop
            crop_width = int(crop_width * scale_factor)
        
        # Ensure crop fits in frame
        crop_width = min(crop_width, source_w)
        crop_height = min(crop_height, source_h)
        
        # Position the crop centered on the detected face/webcam
        # If we have corner_facecam_bbox, trust it completely
        if corner_facecam_bbox:
            # We have explicit corner facecam position - use it directly
            crop_x = int(avg_center_x - crop_width / 2)
            crop_y = int(avg_center_y - crop_height / 2)
        elif avg_center_y > source_h * 0.3:
            # Face is in lower 70% of frame - use it
            crop_x = int(avg_center_x - crop_width / 2)
            crop_y = int(avg_center_y - crop_height / 2)
        else:
            # Face detection probably got screen content (face in top 30%)
            # Default to bottom-right corner where webcam typically is
            crop_x = source_w - crop_width - 20  # 20px margin from edge
            crop_y = source_h - crop_height - 20
            logger.warning(
                f"Face center ({avg_center_x:.0f}, {avg_center_y:.0f}) is in top 30% "
                f"- using bottom-right corner fallback for webcam"
            )

        # Ensure bounds (keep crop in frame)
        crop_x = max(0, min(crop_x, source_w - crop_width))
        crop_y = max(0, min(crop_y, source_h - crop_height))

        logger.info(
            f"Split FACE crop: source={source_w}x{source_h}, "
            f"webcam_size={webcam_size}, base_size={base_size}, "
            f"face_center=({avg_center_x:.0f}, {avg_center_y:.0f}), "
            f"crop={crop_width}x{crop_height} at ({crop_x}, {crop_y}) "
            f"-> scale to {target_width}x{target_height}"
        )

        # Crop around face region, then scale to target dimensions
        # This scales the small webcam up to FILL the entire bottom region
        # Use scale with force_original_aspect_ratio=disable to ensure exact fill
        # Then pad to ensure exact dimensions if needed
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=disable",
            f"setsar=1",  # Ensure square pixels
        ]

        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _render_caption_band(
        self,
        output_path: str,
        target_width: int,
        target_height: int,
        duration_ms: int,
        caption_path: Optional[str],
    ) -> None:
        """
        Create the caption band for split layout.
        
        This is a black background with captions positioned in the center.
        """
        duration_sec = duration_ms / 1000
        
        # Create black background with caption overlay
        if caption_path and os.path.isfile(caption_path):
            # Generate caption band with ASS subtitles
            # Create a modified ASS file with captions positioned for this band
            modified_caption_path = caption_path.replace(".ass", "_band.ass")
            await self._create_band_captions(caption_path, modified_caption_path, target_height)
            
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={target_width}x{target_height}:d={duration_sec}:r=30",
                "-vf", f"ass={self._escape_filter_path(modified_caption_path)}",
                "-c:v", "libx264",
                "-preset", self.settings.ffmpeg_preset,
                "-crf", str(self.settings.ffmpeg_crf),
                "-pix_fmt", "yuv420p",
                "-an",
                output_path,
            ]
            
            try:
                await self._run_cmd(cmd)
            finally:
                if os.path.isfile(modified_caption_path):
                    try:
                        os.remove(modified_caption_path)
                    except Exception:
                        pass
        else:
            # Just black background, no captions
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={target_width}x{target_height}:d={duration_sec}:r=30",
                "-c:v", "libx264",
                "-preset", self.settings.ffmpeg_preset,
                "-crf", str(self.settings.ffmpeg_crf),
                "-pix_fmt", "yuv420p",
                "-an",
                output_path,
            ]
            await self._run_cmd(cmd)

    async def _create_band_captions(
        self,
        source_ass: str,
        output_ass: str,
        band_height: int,
    ) -> None:
        """
        Create modified ASS file with captions centered in caption band.
        """
        try:
            with open(source_ass, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Modify the ASS to position captions in center of band
            # Update PlayResY to match band height
            lines = content.split('\n')
            modified_lines = []
            
            for line in lines:
                if line.startswith('PlayResY:'):
                    modified_lines.append(f'PlayResY: {band_height}')
                elif line.startswith('Style:'):
                    # Modify style to center vertically in band
                    parts = line.split(',')
                    if len(parts) > 10:
                        # Margin V (index ~10) controls vertical position
                        # Set to center the text in the band
                        parts[19] = str(band_height // 4)  # MarginV
                    modified_lines.append(','.join(parts))
                else:
                    modified_lines.append(line)
            
            with open(output_ass, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))
                
        except Exception as e:
            logger.warning(f"Failed to modify caption band ASS: {e}, using original")
            # Copy original as fallback
            import shutil
            shutil.copy(source_ass, output_ass)

    async def _vstack_split_with_overlay(
        self,
        screen_path: str,
        face_path: str,
        output_path: str,
        caption_path: Optional[str],
        original_video_path: str,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Merge 2 regions vertically and overlay captions on the combined video.
        
        Note: screen_path and face_path are already trimmed videos starting at 0.
        We need to extract audio from original_video_path at the correct offset.
        """
        start_sec = start_time_ms / 1000
        duration_sec = duration_ms / 1000

        # Build filter complex for 2-way vstack with optional caption overlay
        use_captions = caption_path and os.path.isfile(caption_path)
        if use_captions:
            # Stack screen and face, then overlay captions on the result
            # Use fontsdir to load fonts from project's fonts/ folder
            ass_filter = f"ass={self._escape_filter_path(caption_path)}"
            if self.fonts_dir:
                fonts_dir_escaped = self._escape_filter_path(self.fonts_dir)
                ass_filter = f"ass={self._escape_filter_path(caption_path)}:fontsdir={fonts_dir_escaped}"
            filter_complex = f"[0:v][1:v]vstack=inputs=2[stacked];[stacked]{ass_filter}[out]"
        else:
            filter_complex = "[0:v][1:v]vstack=inputs=2[out]"


        # The temp videos (screen, face) already start at 0 and have correct duration.
        # We only need to seek the original video for audio extraction.
        cmd = [
            "ffmpeg",
            "-y",
            "-i", screen_path,    # Input 0: screen (top) - already trimmed, starts at 0
            "-i", face_path,      # Input 1: face (bottom) - already trimmed, starts at 0
            "-ss", f"{start_sec:.3f}",  # Seek original video for audio
            "-t", f"{duration_sec:.3f}",
            "-i", original_video_path,  # Input 2: original for audio
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",  # Audio from original video (already seeked)
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]

        await self._run_cmd(cmd)

    async def _render_focus_mode(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """Render with dynamic panning (focus mode)."""
        timeline = request.primary_timeline
        assert timeline is not None
        
        logger.info(
            f"Focus mode render: {len(timeline.keyframes)} keyframes, "
            f"crop {timeline.window_width}x{timeline.window_height} -> "
            f"{self.settings.target_output_width}x{self.settings.target_output_height}"
        )
        
        # Build crop expression
        crop_expr = self._build_crop_expression(
            timeline,
            request.start_time_ms,
            request.source_width,
            request.source_height,
        )
        
        # Build filter chain
        filters = [
            crop_expr,
            f"scale={self.settings.target_output_width}:{self.settings.target_output_height}",
        ]
        
        if caption_path and os.path.isfile(caption_path):
            filters.append(f"ass={self._escape_filter_path(caption_path)}")
        
        await self._run_ffmpeg(
            input_path=request.video_path,
            output_path=request.output_path,
            start_time_ms=request.start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
        )

    async def _render_stack_mode(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render stack mode with two-pass rendering.
        
        Pass 1: Render face stream (bottom)
        Pass 2: Render screen stream (top)
        Pass 3: Merge with vstack and add captions + audio
        """
        primary = request.primary_timeline  # Face
        secondary = request.secondary_timeline  # Screen
        assert primary and secondary
        
        # Calculate output heights
        screen_height = int(
            self.settings.target_output_height * self.settings.stack_screen_height_ratio
        )
        face_height = int(
            self.settings.target_output_height * self.settings.stack_face_height_ratio
        )
        
        logger.debug(
            f"Stack mode render: face={face_height}px, screen={screen_height}px"
        )
        
        output_dir = os.path.dirname(request.output_path)
        timestamp = os.path.basename(request.output_path).replace(".mp4", "")
        temp_face_path = os.path.join(output_dir, f"temp_face_{timestamp}.mp4")
        temp_screen_path = os.path.join(output_dir, f"temp_screen_{timestamp}.mp4")
        
        try:
            # Pass 1: Render screen stream (TOP)
            logger.debug("Pass 1: Rendering screen stream...")
            await self._render_stream_with_static_crop(
                input_path=request.video_path,
                output_path=temp_screen_path,
                timeline=secondary,
                target_width=self.settings.target_output_width,
                target_height=screen_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
            # Pass 2: Render face stream (BOTTOM)
            logger.debug("Pass 2: Rendering face stream...")
            await self._render_stream_with_static_crop(
                input_path=request.video_path,
                output_path=temp_face_path,
                timeline=primary,
                target_width=self.settings.target_output_width,
                target_height=face_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
            # Pass 3: Merge with vstack
            logger.debug("Pass 3: Merging streams...")
            await self._vstack_streams(
                top_path=temp_screen_path,
                bottom_path=temp_face_path,
                output_path=request.output_path,
                caption_path=caption_path,
                original_video_path=request.video_path,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
            )
            
        finally:
            # Cleanup temp files
            for path in [temp_face_path, temp_screen_path]:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _render_static(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """Render with static crop, centered on detected face if available.
        
        For talking_head layouts (podcasts, interviews), detects face positions
        from sampled frames and centers the crop on the speaker.
        """
        # Calculate target crop dimensions
        target_aspect = self.settings.target_output_width / self.settings.target_output_height
        
        if request.source_width / request.source_height > target_aspect:
            # Source is wider - crop width
            crop_height = request.source_height
            crop_width = int(crop_height * target_aspect)
        else:
            # Source is taller - crop height
            crop_width = request.source_width
            crop_height = int(crop_width / target_aspect)
        
        # Default: centered crop
        crop_x = (request.source_width - crop_width) // 2
        crop_y = (request.source_height - crop_height) // 2
        
        # For talking_head layout, try to detect face and center on it
        if request.layout_type == "talking_head":
            face_center_x = await self._detect_face_center_x(
                request.video_path,
                request.start_time_ms,
                request.source_width,
                request.source_height,
            )
            if face_center_x is not None:
                # Center crop horizontally on the detected face
                crop_x = int(face_center_x - crop_width / 2)
                # Clamp to valid range
                crop_x = max(0, min(crop_x, request.source_width - crop_width))
                logger.info(f"Face-centered crop: face_x={face_center_x}, crop_x={crop_x}")
        
        # Build filter chain
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
            f"scale={self.settings.target_output_width}:{self.settings.target_output_height}",
        ]
        
        if caption_path and os.path.isfile(caption_path):
            # Add fontsdir for local fonts
            ass_filter = f"ass={self._escape_filter_path(caption_path)}"
            if self.fonts_dir:
                fonts_dir_escaped = self._escape_filter_path(self.fonts_dir)
                ass_filter = f"ass={self._escape_filter_path(caption_path)}:fontsdir={fonts_dir_escaped}"
            filters.append(ass_filter)
        
        await self._run_ffmpeg(
            input_path=request.video_path,
            output_path=request.output_path,
            start_time_ms=request.start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
        )

    async def _detect_face_center_x(
        self,
        video_path: str,
        start_time_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[int]:
        """Detect the horizontal center of the largest face in the video.
        
        Samples multiple frames to find a stable face position.
        Returns the X coordinate of the face center, or None if no face detected.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Sample 5 frames across the clip to find faces
            sample_offsets_ms = [0, 1000, 2000, 3000, 4000]
            face_centers = []
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            for offset in sample_offsets_ms:
                time_ms = start_time_ms + offset
                frame_pos = int((time_ms / 1000) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    # Find largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    face_center_x = x + w // 2
                    face_centers.append(face_center_x)
            
            cap.release()
            
            if face_centers:
                # Use median for stability
                avg_center = sorted(face_centers)[len(face_centers) // 2]
                logger.info(f"Detected face centers: {face_centers}, using median: {avg_center}")
                return avg_center
            
            return None
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None


    async def _render_stream_with_static_crop(
        self,
        input_path: str,
        output_path: str,
        timeline: CropTimeline,
        target_width: int,
        target_height: int,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Render a single stream with static averaged crop position."""
        # Use static averaged position to avoid FFmpeg expression parsing issues
        keyframes = timeline.keyframes
        
        # Clamp window dimensions to not exceed source
        window_width = min(timeline.window_width, timeline.source_width)
        window_height = min(timeline.window_height, timeline.source_height)
        
        # Calculate average position from keyframes
        avg_center_x = sum(k.center_x for k in keyframes) / len(keyframes)
        avg_center_y = sum(k.center_y for k in keyframes) / len(keyframes)
        
        # Calculate crop position (top-left corner)
        max_x = timeline.source_width - window_width
        max_y = timeline.source_height - window_height
        
        crop_x = 0 if max_x <= 0 else max(
            0, min(max_x, int(avg_center_x - window_width / 2))
        )
        crop_y = 0 if max_y <= 0 else max(
            0, min(max_y, int(avg_center_y - window_height / 2))
        )
        
        logger.debug(
            f"Static crop: {window_width}x{window_height} at ({crop_x}, {crop_y})"
        )
        
        filters = [
            f"crop={window_width}:{window_height}:{crop_x}:{crop_y}",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase",
            f"crop={target_width}:{target_height}",
        ]
        
        await self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            start_time_ms=start_time_ms,
            duration_ms=duration_ms,
            video_filters=filters,
            include_audio=False,
        )

    async def _vstack_streams(
        self,
        top_path: str,
        bottom_path: str,
        output_path: str,
        caption_path: Optional[str],
        original_video_path: str,
        start_time_ms: int,
        duration_ms: int,
    ) -> None:
        """Merge two streams vertically with vstack.
        
        Note: top_path and bottom_path are already trimmed videos starting at 0.
        We need to extract audio from original_video_path at the correct offset.
        """
        # Build filter complex
        filter_complex = "[0:v][1:v]vstack=inputs=2[stacked]"
        
        use_captions = caption_path and os.path.isfile(caption_path)
        if use_captions:
            filter_complex += f";[stacked]ass={self._escape_filter_path(caption_path)}[out]"
        else:
            filter_complex += ";[stacked]copy[out]"
        
        start_sec = start_time_ms / 1000
        duration_sec = duration_ms / 1000
        
        # The temp videos (top, bottom) already start at 0 and have correct duration.
        # We only need to seek the original video for audio extraction.
        cmd = [
            "ffmpeg",
            "-y",
            "-i", top_path,       # Already trimmed, starts at 0
            "-i", bottom_path,    # Already trimmed, starts at 0
            "-ss", f"{start_sec:.3f}",  # Seek original video for audio
            "-t", f"{duration_sec:.3f}",
            "-i", original_video_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",  # Audio from original video (already seeked)
            "-c:a", "aac",
            "-b:a", "192k",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]
        
        await self._run_cmd(cmd)

    def _build_crop_expression(
        self,
        timeline: CropTimeline,
        clip_start_ms: int,
        source_width: int,
        source_height: int,
    ) -> str:
        """Build FFmpeg crop expression with static averaged position.
        
        Dynamic panning has been disabled to avoid glitchy/jittery camera movement
        caused by inconsistent frame-to-frame face detection. Instead, we use
        the averaged position across all keyframes for stable framing.
        """
        window_width = min(timeline.window_width, source_width)
        window_height = min(timeline.window_height, source_height)
        keyframes = timeline.keyframes
        
        # DISABLED: Dynamic panning causes glitchy movement
        # Always use static averaged position for stable framing
        avg_x = sum(k.center_x for k in keyframes) / len(keyframes)
        avg_y = sum(k.center_y for k in keyframes) / len(keyframes)
        
        max_x = source_width - window_width
        max_y = source_height - window_height
        
        x = 0 if max_x <= 0 else max(0, min(max_x, int(avg_x - window_width / 2)))
        y = 0 if max_y <= 0 else max(0, min(max_y, int(avg_y - window_height / 2)))
        
        logger.info(f"Static crop: {window_width}x{window_height} at ({x}, {y}) from {len(keyframes)} keyframes")
        
        return f"crop={window_width}:{window_height}:{x}:{y}"
        
        # NOTE: The code below is preserved but disabled.
        # To re-enable dynamic panning, remove the return statement above.
        
        # Build interpolation expressions for X and Y
        half_w = window_width / 2
        half_h = window_height / 2
        
        x_expr = self._build_interpolation_expr(
            keyframes, clip_start_ms, "center_x", half_w, source_width, window_width
        )
        y_expr = self._build_interpolation_expr(
            keyframes, clip_start_ms, "center_y", half_h, source_height, window_height
        )
        
        # CRITICAL: FFmpeg uses commas to separate filters in the -vf chain.
        # Commas inside crop expressions (e.g., in max/min/if functions) must be
        # escaped with backslashes so FFmpeg interprets them as part of the expression.
        x_expr_escaped = x_expr.replace(",", "\\,")
        y_expr_escaped = y_expr.replace(",", "\\,")
        
        return f"crop={window_width}:{window_height}:{x_expr_escaped}:{y_expr_escaped}"

    def _build_interpolation_expr(
        self,
        keyframes: list[CropKeyframe],
        clip_start_ms: int,
        prop: str,
        half_window: float,
        max_dim: int,
        window_dim: int,
    ) -> str:
        """Build FFmpeg interpolation expression for smooth camera movement."""
        max_position = max_dim - window_dim - BOUNDARY_PADDING
        
        # If no room to pan, return centered static position
        if max_position <= BOUNDARY_PADDING:
            return str(max(0, (max_dim - window_dim) // 2))
        
        # Convert keyframes to relative time
        rel_keyframes = [
            {
                "t": (k.timestamp_ms - clip_start_ms) / 1000,
                "v": int(getattr(k, prop) - half_window),
            }
            for k in keyframes
        ]
        
        # Sample keyframes to avoid expression length limits
        # Use more keyframes for smoother motion (up to 15)
        if len(rel_keyframes) > 15:
            step = max(1, len(rel_keyframes) // 15)
            sampled = rel_keyframes[::step]
            if sampled[-1] != rel_keyframes[-1]:
                sampled.append(rel_keyframes[-1])
            rel_keyframes = sampled
        
        # Build nested if expression with clamping
        inner_expr = self._build_nested_if_expr(rel_keyframes)
        return f"max({BOUNDARY_PADDING},min({max_position},{inner_expr}))"

    def _build_nested_if_expr(self, keyframes: list[dict]) -> str:
        """Build nested if expression for piecewise linear interpolation."""
        if len(keyframes) == 0:
            return "0"
        if len(keyframes) == 1:
            return str(keyframes[0]["v"])
        
        # Build from the end backwards
        expr = str(keyframes[-1]["v"])
        
        for i in range(len(keyframes) - 2, -1, -1):
            curr = keyframes[i]
            next_ = keyframes[i + 1]
            dt = next_["t"] - curr["t"]
            
            if dt <= 0:
                continue
            
            # Linear interpolation: V0 + (V1-V0) * (t-T0) / (T1-T0)
            lerp = f"{curr['v']}+{next_['v'] - curr['v']}*(t-{curr['t']:.3f})/{dt:.3f}"
            expr = f"if(lte(t,{next_['t']:.3f}),{lerp},{expr})"
        
        return expr

    async def _run_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time_ms: int,
        duration_ms: int,
        video_filters: list[str],
        include_audio: bool = True,
    ) -> None:
        """Run FFmpeg with frame-accurate seeking for precise clip extraction."""
        # Apply audio padding to avoid cutting mid-syllable
        # Subtract padding from start (but not below 0) and add padding to duration
        audio_padding_ms = self.settings.audio_padding_ms
        
        # Adjust start time (subtract padding, but not below 0)
        adjusted_start_ms = max(0, start_time_ms - audio_padding_ms)
        start_adjustment = start_time_ms - adjusted_start_ms
        
        # Adjust duration to account for the start adjustment and add end padding
        adjusted_duration_ms = duration_ms + start_adjustment + audio_padding_ms
        
        start_sec = adjusted_start_ms / 1000
        duration_sec = adjusted_duration_ms / 1000
        
        # Frame-accurate seeking: Use -ss before -i for fast seek, then trim precisely
        # This ensures no frames are skipped at boundaries
        cmd = [
            "ffmpeg",
            "-y",
            "-accurate_seek",  # Enable frame-accurate seeking
            "-ss", f"{start_sec:.6f}",  # Use high precision (microseconds)
            "-i", input_path,
            "-t", f"{duration_sec:.6f}",  # High precision duration
            "-vf", ",".join(video_filters),
            "-avoid_negative_ts", "make_zero",  # Ensure timestamps start at 0
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-pix_fmt", "yuv420p",
        ]
        
        if include_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.append("-an")
        
        cmd.append(output_path)
        
        await self._run_cmd(cmd)

    async def _run_cmd(self, cmd: list[str]) -> None:
        """Run a command asynchronously."""
        logger.debug(f"Running: {' '.join(cmd[:10])}...")
        
        # Use run_in_executor for Windows compatibility
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode()[-1000:] if result.stderr else "Unknown error"
            raise RenderingError(f"FFmpeg failed: {error_msg}")

    def _escape_filter_path(self, path: str) -> str:
        """
        Escape file path for FFmpeg filter usage.
        
        FFmpeg filters have specific path escaping requirements that differ 
        across platforms. This method handles:
        - Windows: Convert backslashes to forward slashes, escape drive letter colons
        - All platforms: Escape special characters that have meaning in FFmpeg filters
        
        Args:
            path: The file path to escape
            
        Returns:
            Escaped path string safe for use in FFmpeg filter expressions
        """
        import sys
        
        # Normalize path separators to forward slashes (works on all platforms in FFmpeg)
        escaped = path.replace("\\", "/")
        
        # On Windows, escape the drive letter colon (C: -> C\:)
        # Must be done AFTER backslash replacement to avoid double-escaping
        if sys.platform == "win32" and len(escaped) >= 2 and escaped[1] == ":":
            escaped = escaped[0] + "\\:" + escaped[2:]
        
        # Escape special characters used in FFmpeg filter syntax
        escaped = escaped.replace("'", "'\\''")  # Escape single quotes for shell
        escaped = escaped.replace("[", "\\[")
        escaped = escaped.replace("]", "\\]")
        
        # Wrap in single quotes for the filter expression
        return f"'{escaped}'"

    async def _render_with_keyframes(
        self,
        request: RenderRequest,
        caption_path: Optional[str],
        duration_ms: int,
    ) -> None:
        """
        Render clip with multiple layout segments based on timeline keyframes.
        
        Each keyframe defines a layout configuration that applies from its timestamp
        until the next keyframe (or end of clip). Segments are rendered separately
        then concatenated with FFmpeg.
        """
        keyframes = sorted(request.layout_keyframes, key=lambda k: k.timestamp_ms)
        
        if len(keyframes) == 0:
            # Fallback to static rendering if no keyframes
            await self._render_static(request, caption_path, duration_ms)
            return
        
        output_dir = os.path.dirname(request.output_path)
        timestamp = os.path.basename(request.output_path).replace(".mp4", "")
        segment_paths: list[str] = []
        
        try:
            # Render each segment between keyframes
            for i, keyframe in enumerate(keyframes):
                # Calculate segment time range (relative to clip start)
                segment_start_ms = keyframe.timestamp_ms
                if i + 1 < len(keyframes):
                    segment_end_ms = keyframes[i + 1].timestamp_ms
                else:
                    segment_end_ms = duration_ms
                
                # Skip empty segments
                segment_duration = segment_end_ms - segment_start_ms
                if segment_duration <= 0:
                    continue
                
                # Create segment output path
                segment_path = os.path.join(
                    output_dir, 
                    f"temp_segment_{timestamp}_{i}.mp4"
                )
                segment_paths.append(segment_path)
                
                logger.info(
                    f"Rendering keyframe segment {i+1}/{len(keyframes)}: "
                    f"{segment_start_ms}ms-{segment_end_ms}ms, layout={keyframe.layout_type}"
                )
                
                # Render this segment with the keyframe's layout configuration
                await self._render_keyframe_segment(
                    request=request,
                    segment_path=segment_path,
                    segment_start_ms=segment_start_ms,
                    segment_end_ms=segment_end_ms,
                    keyframe=keyframe,
                )
            
            if len(segment_paths) == 0:
                raise RenderingError("No segments were rendered")
            
            if len(segment_paths) == 1:
                # Single segment - just move it to output
                shutil.move(segment_paths[0], request.output_path)
                segment_paths.clear()
            else:
                # Multiple segments - concatenate them
                await self._concat_segments(
                    segment_paths=segment_paths,
                    output_path=request.output_path,
                    caption_path=caption_path,
                    audio_source_path=request.video_path,
                    audio_start_ms=request.start_time_ms,
                    audio_duration_ms=duration_ms,
                )
        
        finally:
            # Cleanup temp segment files
            for path in segment_paths:
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
    
    async def _render_keyframe_segment(
        self,
        request: RenderRequest,
        segment_path: str,
        segment_start_ms: int,
        segment_end_ms: int,
        keyframe: LayoutKeyframe,
    ) -> None:
        """
        Render a single segment with the keyframe's layout configuration.
        
        Creates a temporary RenderRequest for this segment and routes to the
        appropriate rendering method based on the keyframe's layout_type.
        """
        segment_duration = segment_end_ms - segment_start_ms
        
        # Get actual video dimensions
        source_w, source_h = await self._get_video_dimensions(request.video_path)
        if source_w == 0 or source_h == 0:
            source_w = request.source_width
            source_h = request.source_height
        
        # Calculate region heights for split layout
        total_height = self.settings.target_output_height
        screen_height = int(total_height * self.settings.split_screen_ratio)
        face_height = total_height - screen_height
        
        # Absolute timestamps in the source video
        abs_start_ms = request.start_time_ms + segment_start_ms
        abs_end_ms = request.start_time_ms + segment_end_ms
        
        if keyframe.layout_type == "screen_share" and keyframe.face_bbox:
            # Split screen with provided facecam bbox
            output_dir = os.path.dirname(segment_path)
            basename = os.path.basename(segment_path).replace(".mp4", "")
            temp_screen_path = os.path.join(output_dir, f"temp_screen_{basename}.mp4")
            temp_face_path = os.path.join(output_dir, f"temp_face_{basename}.mp4")
            
            try:
                # Render screen content (top half)
                await self._render_static_screen_crop(
                    input_path=request.video_path,
                    output_path=temp_screen_path,
                    source_width=source_w,
                    source_height=source_h,
                    target_width=self.settings.target_output_width,
                    target_height=screen_height,
                    start_time_ms=abs_start_ms,
                    duration_ms=segment_duration,
                )
                
                # Render face crop (bottom half)
                await self._render_static_face_crop(
                    input_path=request.video_path,
                    output_path=temp_face_path,
                    source_width=source_w,
                    source_height=source_h,
                    corner_facecam_bbox=keyframe.face_bbox,
                    target_width=self.settings.target_output_width,
                    target_height=face_height,
                    start_time_ms=abs_start_ms,
                    duration_ms=segment_duration,
                    is_custom_bbox=True,  # User-defined keyframe boxes
                )
                
                # Stack them vertically (no audio yet - added during concat)
                await self._vstack_segments_no_audio(
                    top_path=temp_screen_path,
                    bottom_path=temp_face_path,
                    output_path=segment_path,
                )
            finally:
                for path in [temp_screen_path, temp_face_path]:
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
        
        elif keyframe.layout_type == "podcast" and keyframe.podcast_bboxes and len(keyframe.podcast_bboxes) >= 2:
            # Podcast mode with 2 speaker boxes
            panel_width = self.settings.target_output_width
            panel_height = self.settings.target_output_height // 2
            
            output_dir = os.path.dirname(segment_path)
            basename = os.path.basename(segment_path).replace(".mp4", "")
            temp_face1_path = os.path.join(output_dir, f"temp_podcast_face1_{basename}.mp4")
            temp_face2_path = os.path.join(output_dir, f"temp_podcast_face2_{basename}.mp4")
            
            try:
                # Render Face 1 (top panel)
                await self._render_podcast_panel(
                    input_path=request.video_path,
                    output_path=temp_face1_path,
                    face_bbox=keyframe.podcast_bboxes[0],
                    source_width=source_w,
                    source_height=source_h,
                    target_width=panel_width,
                    target_height=panel_height,
                    start_time_ms=abs_start_ms,
                    duration_ms=segment_duration,
                )
                
                # Render Face 2 (bottom panel)
                await self._render_podcast_panel(
                    input_path=request.video_path,
                    output_path=temp_face2_path,
                    face_bbox=keyframe.podcast_bboxes[1],
                    source_width=source_w,
                    source_height=source_h,
                    target_width=panel_width,
                    target_height=panel_height,
                    start_time_ms=abs_start_ms,
                    duration_ms=segment_duration,
                )
                
                # Stack them vertically (no audio yet)
                await self._vstack_segments_no_audio(
                    top_path=temp_face1_path,
                    bottom_path=temp_face2_path,
                    output_path=segment_path,
                )
            finally:
                for path in [temp_face1_path, temp_face2_path]:
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
        
        else:
            # talking_head or fallback - simple centered crop
            # Calculate 9:16 crop dimensions
            target_aspect = self.settings.target_output_width / self.settings.target_output_height
            
            if source_w / source_h > target_aspect:
                # Source is wider - crop width
                crop_h = source_h
                crop_w = int(source_h * target_aspect)
            else:
                # Source is taller - crop height
                crop_w = source_w
                crop_h = int(source_w / target_aspect)
            
            # Center the crop, optionally offset by face_bbox if provided
            center_x = source_w // 2
            center_y = source_h // 2
            
            if keyframe.face_bbox:
                # Use face bbox center as crop center
                fx, fy, fw, fh = keyframe.face_bbox
                center_x = fx + fw // 2
                center_y = fy + fh // 2
            
            crop_x = max(0, min(center_x - crop_w // 2, source_w - crop_w))
            crop_y = max(0, min(center_y - crop_h // 2, source_h - crop_h))
            
            filters = [
                f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
                f"scale={self.settings.target_output_width}:{self.settings.target_output_height}:force_original_aspect_ratio=disable",
                "setsar=1",
            ]
            
            await self._run_ffmpeg(
                input_path=request.video_path,
                output_path=segment_path,
                start_time_ms=abs_start_ms,
                duration_ms=segment_duration,
                video_filters=filters,
                include_audio=False,
            )
    
    async def _vstack_segments_no_audio(
        self,
        top_path: str,
        bottom_path: str,
        output_path: str,
    ) -> None:
        """Vertically stack two video segments without audio."""
        filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", top_path,
            "-i", bottom_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", "libx264",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-an",  # No audio
            output_path,
        ]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"vstack failed: {error_msg}")
            raise RenderingError(f"vstack failed: {error_msg[:200]}")
    
    async def _concat_segments(
        self,
        segment_paths: list[str],
        output_path: str,
        caption_path: Optional[str],
        audio_source_path: str,
        audio_start_ms: int,
        audio_duration_ms: int,
    ) -> None:
        """
        Concatenate multiple video segments and add audio + captions.
        
        Uses FFmpeg concat filter to join segments, then overlays the original
        audio track and optional captions.
        """
        if len(segment_paths) == 0:
            raise RenderingError("No segments to concatenate")
        
        # Build input args and filter complex
        input_args = []
        for path in segment_paths:
            input_args.extend(["-i", path])
        
        # Add audio source
        audio_start_sec = audio_start_ms / 1000.0
        audio_duration_sec = audio_duration_ms / 1000.0
        input_args.extend([
            "-ss", str(audio_start_sec),
            "-t", str(audio_duration_sec),
            "-i", audio_source_path,
        ])
        
        # Build concat filter
        n_segments = len(segment_paths)
        filter_inputs = "".join(f"[{i}:v]" for i in range(n_segments))
        filter_complex = f"{filter_inputs}concat=n={n_segments}:v=1:a=0[concat_v]"
        
        # Add captions if available
        if caption_path and os.path.isfile(caption_path):
            escaped_caption = self._escape_filter_path(caption_path)
            filter_complex += f";[concat_v]ass={escaped_caption}[final]"
            map_video = "[final]"
        else:
            map_video = "[concat_v]"
        
        # Audio is the last input
        audio_input_idx = n_segments
        
        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", map_video,
            "-map", f"{audio_input_idx}:a",
            "-c:v", "libx264",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.ffmpeg_crf),
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        logger.debug(f"Concat command: {' '.join(cmd)}")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"Concat failed: {error_msg}")
            raise RenderingError(f"Segment concatenation failed: {error_msg[:200]}")


class RenderingError(Exception):
    """Exception raised when rendering fails."""
    pass

