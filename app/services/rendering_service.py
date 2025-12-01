"""
Rendering Service - FFmpeg-based video rendering with dynamic cropping and captions.
"""

import asyncio
import logging
import os
import shutil
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
class RenderRequest:
    """Request for rendering a clip."""
    
    video_path: str
    output_path: str
    start_time_ms: int
    end_time_ms: int
    source_width: int
    source_height: int
    layout_type: Literal["talking_head", "screen_share"]
    
    # Crop timelines for dynamic panning
    primary_timeline: Optional[CropTimeline] = None
    secondary_timeline: Optional[CropTimeline] = None  # For stack mode
    
    # Captions
    transcript_segments: Optional[list[TranscriptSegment]] = None
    caption_style: Optional[CaptionStyle] = None


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
            )
        
        # Choose rendering method based on layout and timelines
        if (
            request.layout_type == "screen_share"
            and request.primary_timeline
            and request.secondary_timeline
        ):
            # Use split layout if enabled (screen top, face bottom, captions overlaid)
            if self.settings.use_split_layout:
                await self._render_split_mode(request, caption_path, duration_ms)
            else:
                # Fallback to standard stack mode
                await self._render_stack_mode(request, caption_path, duration_ms)
        elif request.primary_timeline:
            # Focus mode: single dynamic pan
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
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            dims = stdout.decode().strip().split("x")
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
            logger.debug("Pass 2: Rendering face crop (bottom)...")
            await self._render_split_face(
                input_path=request.video_path,
                output_path=temp_face_path,
                timeline=face_timeline,
                target_width=self.settings.target_output_width,
                target_height=face_height,
                start_time_ms=request.start_time_ms,
                duration_ms=duration_ms,
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

        Takes a SQUARE crop from the CENTER of the original frame,
        then resizes to fill the target width. This captures the screen
        content (typically in the middle of the frame).
        """
        source_w = timeline.source_width
        source_h = timeline.source_height

        # Take a square crop from the CENTER of the frame
        # Use the smaller dimension to ensure the crop fits
        square_size = min(source_w, source_h)

        # Safety check: ensure square_size doesn't exceed actual source
        square_size = min(square_size, source_w, source_h)

        # Center the square crop
        crop_x = max(0, (source_w - square_size) // 2)
        crop_y = max(0, (source_h - square_size) // 2)

        logger.info(
            f"Split SCREEN crop: source={source_w}x{source_h}, "
            f"square {square_size}x{square_size} from center at ({crop_x}, {crop_y}) "
            f"-> scale to {target_width}x{target_height}"
        )

        # Crop square from center, then scale to fill target width
        # The height will be cropped to fit the target aspect ratio
        filters = [
            f"crop={square_size}:{square_size}:{crop_x}:{crop_y}",
            f"scale={target_width}:-1",  # Scale to target width, maintain aspect
            f"crop={target_width}:{target_height}:0:(ih-{target_height})/2",  # Center crop to target height
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
    ) -> None:
        """
        Render tight face crop for split layout (bottom region).

        Creates a TIGHT SQUARE crop around the detected face position,
        then scales UP dramatically to fill the bottom region.
        This creates a close-up portrait view of the speaker.
        """
        keyframes = timeline.keyframes

        # Calculate weighted average position from keyframes (face center)
        avg_center_x = sum(k.center_x for k in keyframes) / len(keyframes)
        avg_center_y = sum(k.center_y for k in keyframes) / len(keyframes)

        source_w = timeline.source_width
        source_h = timeline.source_height

        # CRITICAL: Calculate crop size as fraction of source to ensure it fits
        # Use 1/3 of the smaller dimension for a tight face crop
        max_safe_crop = min(source_w, source_h) // 3
        max_safe_crop = max(max_safe_crop, 200)  # Minimum for quality

        # Use the window dimensions from timeline, but cap strictly
        crop_size = max(timeline.window_width, timeline.window_height)
        crop_size = max(crop_size, 150)  # Minimum size
        crop_size = min(crop_size, max_safe_crop)  # Cap at 1/3 of source

        # Center the square crop on the face position
        max_x = source_w - crop_size
        max_y = source_h - crop_size

        # Ensure we have valid crop bounds
        max_x = max(0, max_x)
        max_y = max(0, max_y)

        crop_x = max(0, min(max_x, int(avg_center_x - crop_size / 2)))
        crop_y = max(0, min(max_y, int(avg_center_y - crop_size / 2)))

        # Calculate scale factor needed to fill target width
        scale_factor = target_width / crop_size
        scaled_height = int(crop_size * scale_factor)

        logger.info(
            f"Split FACE crop: source={source_w}x{source_h}, "
            f"crop={crop_size}x{crop_size} at ({crop_x}, {crop_y}) "
            f"-> scale {scale_factor:.2f}x to {target_width}x{scaled_height} "
            f"-> final {target_width}x{target_height}"
        )

        # Crop tight square around face, scale UP to fill target width,
        # then crop height from CENTER to fit target dimensions
        filters = [
            f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y}",
            f"scale={target_width}:-1",  # Scale to target width, maintaining aspect
            f"crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2",
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
        """Merge 2 regions vertically and overlay captions on the combined video."""
        start_sec = start_time_ms / 1000
        duration_sec = duration_ms / 1000

        # Build filter complex for 2-way vstack with optional caption overlay
        use_captions = caption_path and os.path.isfile(caption_path)
        if use_captions:
            # Stack screen and face, then overlay captions on the result
            filter_complex = f"[0:v][1:v]vstack=inputs=2[stacked];[stacked]ass={self._escape_filter_path(caption_path)}[out]"
        else:
            filter_complex = "[0:v][1:v]vstack=inputs=2[out]"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", screen_path,    # Input 0: screen (top)
            "-i", face_path,      # Input 1: face (bottom)
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", original_video_path,  # Input 2: original for audio
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",  # Audio from original video
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
        
        logger.debug(f"Focus mode render: {len(timeline.keyframes)} keyframes")
        
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
        """Render with static centered crop (fallback)."""
        # Calculate centered crop
        target_aspect = self.settings.target_output_width / self.settings.target_output_height
        
        if request.source_width / request.source_height > target_aspect:
            # Source is wider - crop width
            crop_height = request.source_height
            crop_width = int(crop_height * target_aspect)
        else:
            # Source is taller - crop height
            crop_width = request.source_width
            crop_height = int(crop_width / target_aspect)
        
        crop_x = (request.source_width - crop_width) // 2
        crop_y = (request.source_height - crop_height) // 2
        
        # Build filter chain
        filters = [
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
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
        """Merge two streams vertically with vstack."""
        # Build filter complex
        filter_complex = "[0:v][1:v]vstack=inputs=2[stacked]"
        
        use_captions = caption_path and os.path.isfile(caption_path)
        if use_captions:
            filter_complex += f";[stacked]ass={self._escape_filter_path(caption_path)}[out]"
        else:
            filter_complex += ";[stacked]copy[out]"
        
        start_sec = start_time_ms / 1000
        duration_sec = duration_ms / 1000
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", top_path,
            "-i", bottom_path,
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", original_video_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "2:a?",  # Audio from original video
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
        """Build FFmpeg crop expression with dynamic X,Y."""
        window_width = min(timeline.window_width, source_width)
        window_height = min(timeline.window_height, source_height)
        keyframes = timeline.keyframes
        
        # If few keyframes, use static crop
        if len(keyframes) <= 2:
            avg_x = sum(k.center_x for k in keyframes) / len(keyframes)
            avg_y = sum(k.center_y for k in keyframes) / len(keyframes)
            
            max_x = source_width - window_width
            max_y = source_height - window_height
            
            x = 0 if max_x <= 0 else max(0, min(max_x, int(avg_x - window_width / 2)))
            y = 0 if max_y <= 0 else max(0, min(max_y, int(avg_y - window_height / 2)))
            
            return f"crop={window_width}:{window_height}:{x}:{y}"
        
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
        if len(rel_keyframes) > 5:
            step = len(rel_keyframes) // 10 or 1
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
        """Run FFmpeg with given filters."""
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time_ms / 1000),
            "-i", input_path,
            "-t", str(duration_ms / 1000),
            "-vf", ",".join(video_filters),
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
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()[-1000:] if stderr else "Unknown error"
            raise RenderingError(f"FFmpeg failed: {error_msg}")

    def _escape_filter_path(self, path: str) -> str:
        """Escape path for FFmpeg filter usage."""
        # FFmpeg filter escaping rules
        return (
            path
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(";", "\\;")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )


class RenderingError(Exception):
    """Exception raised when rendering fails."""
    pass

