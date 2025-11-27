"""
AI Clipping Pipeline - Unified orchestrator for the complete AI clipping workflow.

This service orchestrates the entire AI clipping pipeline:
1. Video download (YouTube via yt-dlp)
2. Audio extraction and transcription (Groq Whisper)
3. Intelligence planning (Gemini via OpenRouter)
4. Detection analysis (local YOLO/MediaPipe)
5. Clip rendering (FFmpeg with captions)
6. S3 upload
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import CaptionStyle, get_settings
from app.services.caption_generator import CaptionGeneratorService
from app.services.detection_pipeline import DetectionPipeline
from app.services.intelligence_planner import (
    ClipPlanResponse,
    ClipPlanSegment,
    IntelligencePlannerService,
)
from app.services.rendering_service import (
    CropKeyframe,
    CropTimeline,
    RenderRequest,
    RenderingService,
)
from app.services.s3_upload_service import (
    ClipArtifact,
    JobOutput,
    S3UploadService,
)
from app.services.transcription_service import (
    TranscriptionResult,
    TranscriptionService,
)
from app.services.video_downloader import (
    DownloadResult,
    VideoDownloaderService,
)
from app.services.content_region_detector import (
    ContentRegionDetector,
    FrameAnalysis,
)

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of an AI clipping job."""
    
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    PLANNING = "planning"
    DETECTING = "detecting"
    RENDERING = "rendering"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClippingJobRequest:
    """Request to process a video for AI clipping."""
    
    video_url: str
    job_id: Optional[str] = None
    owner_user_id: Optional[str] = None  # User ID for S3 key scoping
    max_clips: int = 5
    min_clip_duration_seconds: int = 15
    max_clip_duration_seconds: int = 90
    duration_ranges: Optional[list[str]] = None  # ['short', 'medium', 'long']
    target_platform: str = "tiktok"  # tiktok, youtube_shorts, instagram_reels
    include_captions: bool = True
    caption_style: Optional[CaptionStyle] = None
    callback_url: Optional[str] = None
    
    def __post_init__(self):
        if self.job_id is None:
            self.job_id = str(uuid.uuid4())


@dataclass
class ClippingJobProgress:
    """Progress update for a clipping job."""
    
    job_id: str
    status: JobStatus
    progress_percent: float
    current_step: str
    clips_completed: int = 0
    total_clips: int = 0
    error: Optional[str] = None


@dataclass
class ClippingJobResult:
    """Final result of a clipping job."""
    
    job_id: str
    status: JobStatus
    output: Optional[JobOutput] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0


class AIClippingPipeline:
    """
    Unified pipeline for AI-powered video clipping.
    
    This service orchestrates:
    - Video downloading from YouTube
    - Audio transcription via Whisper
    - Intelligent clip planning via Gemini
    - Face/pose detection for smart cropping
    - FFmpeg rendering with captions
    - S3 upload of final clips
    """

    def __init__(
        self,
        detection_pipeline: DetectionPipeline,
        progress_callback: Optional[Callable[[ClippingJobProgress], None]] = None,
    ):
        self.settings = get_settings()
        self.detection_pipeline = detection_pipeline
        self.progress_callback = progress_callback
        
        # Initialize services
        self.video_downloader = VideoDownloaderService()
        self.transcription_service = TranscriptionService()
        self.intelligence_planner = IntelligencePlannerService()
        self.rendering_service = RenderingService()
        self.s3_upload_service = S3UploadService()
        self.content_region_detector = ContentRegionDetector()

    async def process_video(
        self,
        request: ClippingJobRequest,
    ) -> ClippingJobResult:
        """
        Process a video through the full AI clipping pipeline.
        
        Args:
            request: ClippingJobRequest with video URL and options
            
        Returns:
            ClippingJobResult with output or error
        """
        start_time = time.time()
        job_id = request.job_id
        work_dir = os.path.join(self.settings.temp_directory, job_id)
        
        try:
            os.makedirs(work_dir, exist_ok=True)
            logger.info(f"Starting AI clipping job: {job_id}")
            
            # Step 1: Download video
            self._update_progress(job_id, JobStatus.DOWNLOADING, 5, "Downloading video...")
            download_result = await self.video_downloader.download_video(
                url=request.video_url,
                output_dir=work_dir,
            )
            logger.info(f"Downloaded: {download_result.metadata.title}")
            
            # Step 2: Transcribe audio
            self._update_progress(job_id, JobStatus.TRANSCRIBING, 15, "Transcribing audio...")
            transcription_result = await self.transcription_service.transcribe(
                video_path=download_result.video_path,
                work_dir=work_dir,
            )
            logger.info(f"Transcription complete: {len(transcription_result.segments)} segments")
            
            # Upload transcript artifact
            transcript_upload = await self.s3_upload_service.upload_json_artifact(
                data={
                    "segments": [asdict(s) for s in transcription_result.segments],
                    "full_text": transcription_result.full_text,
                    "language": transcription_result.language,
                },
                job_id=job_id,
                artifact_name="transcript",
                user_id=request.owner_user_id,
            )
            
            # Step 3: Plan clips using AI
            self._update_progress(job_id, JobStatus.PLANNING, 30, "Planning viral clips...")
            clip_plan = await self.intelligence_planner.plan_clips(
                transcript_result=transcription_result,
                video_metadata=download_result.metadata,
                max_clips=request.max_clips,
                min_duration_seconds=request.min_clip_duration_seconds,
                max_duration_seconds=request.max_clip_duration_seconds,
                duration_ranges=request.duration_ranges,
                target_platform=request.target_platform,
            )
            logger.info(f"Planned {len(clip_plan.segments)} clips")
            
            # Upload plan artifact
            plan_upload = await self.s3_upload_service.upload_json_artifact(
                data={
                    "segments": [asdict(s) for s in clip_plan.segments],
                    "total_clips": clip_plan.total_clips,
                    "target_platform": clip_plan.target_platform,
                },
                job_id=job_id,
                artifact_name="plan",
                user_id=request.owner_user_id,
            )
            
            # Step 4: Run detection for smart cropping
            self._update_progress(job_id, JobStatus.DETECTING, 45, "Analyzing video for smart cropping...")
            detection_results = await self._run_detection_for_clips(
                video_path=download_result.video_path,
                clip_segments=clip_plan.segments,
                source_width=download_result.metadata.width,
                source_height=download_result.metadata.height,
            )
            logger.info("Detection analysis complete")
            
            # Step 5: Render clips
            clips_dir = os.path.join(work_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            
            rendered_clips: list[tuple[str, ClipPlanSegment]] = []
            total_clips = len(clip_plan.segments)
            
            for i, segment in enumerate(clip_plan.segments):
                progress = 50 + (35 * i / total_clips)
                self._update_progress(
                    job_id, JobStatus.RENDERING, progress,
                    f"Rendering clip {i + 1}/{total_clips}...",
                    clips_completed=i, total_clips=total_clips,
                )
                
                output_path = os.path.join(clips_dir, f"clip_{i:02d}.mp4")
                
                # Get detection results for this clip's timeframe
                crop_timeline = self._build_crop_timeline(
                    detection_results.get(i),
                    segment,
                    download_result.metadata.width,
                    download_result.metadata.height,
                )
                
                # For screen_share layout, also build the screen timeline
                screen_timeline = None
                if segment.layout_type == "screen_share":
                    screen_timeline = self._build_screen_timeline(
                        segment,
                        download_result.metadata.width,
                        download_result.metadata.height,
                        detection_results.get(i),  # Pass detection frames for webcam detection
                    )
                    logger.info(f"Screen share layout: face timeline={crop_timeline is not None}, screen timeline={screen_timeline is not None}")
                
                # Get transcript segments for this clip
                clip_transcript = self._filter_transcript_for_clip(
                    transcription_result.segments,
                    segment.start_time_ms,
                    segment.end_time_ms,
                )
                
                # Build render request
                render_request = RenderRequest(
                    video_path=download_result.video_path,
                    output_path=output_path,
                    start_time_ms=segment.start_time_ms,
                    end_time_ms=segment.end_time_ms,
                    source_width=download_result.metadata.width,
                    source_height=download_result.metadata.height,
                    layout_type=segment.layout_type,
                    primary_timeline=crop_timeline,  # Face tracking for both layouts
                    secondary_timeline=screen_timeline,  # Screen crop for screen_share only
                    transcript_segments=clip_transcript if request.include_captions else None,
                    caption_style=request.caption_style,
                )
                
                render_result = await self.rendering_service.render_clip(render_request)
                rendered_clips.append((render_result.output_path, segment))
                
                logger.info(f"Rendered clip {i + 1}: {render_result.file_size_bytes / 1024 / 1024:.1f} MB")
            
            # Step 6: Upload clips to S3
            self._update_progress(
                job_id, JobStatus.UPLOADING, 90,
                "Uploading clips to storage...",
                clips_completed=total_clips, total_clips=total_clips,
            )
            
            clip_artifacts: list[ClipArtifact] = []
            
            for i, (clip_path, segment) in enumerate(rendered_clips):
                upload_result = await self.s3_upload_service.upload_clip(
                    local_path=clip_path,
                    job_id=job_id,
                    clip_index=i,
                    user_id=request.owner_user_id,
                    metadata={
                        "virality_score": segment.virality_score,
                        "layout_type": segment.layout_type,
                        "start_time_ms": segment.start_time_ms,
                        "end_time_ms": segment.end_time_ms,
                    },
                )
                
                clip_artifacts.append(ClipArtifact(
                    clip_index=i,
                    s3_url=upload_result.s3_url,
                    duration_ms=segment.end_time_ms - segment.start_time_ms,
                    start_time_ms=segment.start_time_ms,
                    end_time_ms=segment.end_time_ms,
                    virality_score=segment.virality_score,
                    layout_type=segment.layout_type,
                    summary=segment.summary,
                    tags=segment.tags or [],
                ))
            
            # Create job output
            processing_time = time.time() - start_time

            job_output = JobOutput(
                job_id=job_id,
                source_video_url=request.video_url,
                source_video_title=download_result.metadata.title,
                source_video_duration_seconds=download_result.metadata.duration_seconds,
                total_clips=len(clip_artifacts),
                clips=clip_artifacts,
                user_id=request.owner_user_id,
                transcript_url=transcript_upload.s3_url,
                plan_url=plan_upload.s3_url,
                processing_time_seconds=processing_time,
            )
            
            # Upload job manifest
            await self.s3_upload_service.upload_job_output(job_output)
            
            self._update_progress(
                job_id, JobStatus.COMPLETED, 100,
                "Processing complete!",
                clips_completed=total_clips, total_clips=total_clips,
            )
            
            logger.info(f"Job {job_id} completed in {processing_time:.1f}s with {len(clip_artifacts)} clips")
            
            return ClippingJobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                output=job_output,
                processing_time_seconds=processing_time,
            )
            
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            
            self._update_progress(
                job_id, JobStatus.FAILED, 0,
                "Processing failed",
                error=str(e),
            )
            
            return ClippingJobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )
            
        finally:
            # Cleanup work directory
            if os.path.isdir(work_dir):
                try:
                    shutil.rmtree(work_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup work dir: {e}")

    async def _run_detection_for_clips(
        self,
        video_path: str,
        clip_segments: list[ClipPlanSegment],
        source_width: int,
        source_height: int,
    ) -> dict[int, list[dict]]:
        """
        Run detection pipeline for each planned clip segment.
        
        Returns a mapping of clip_index -> detection frames.
        
        Note: Detection is run for ALL layout types to enable smart cropping:
        - talking_head: Uses face detection for dynamic panning
        - screen_share: Uses face detection for the face portion of stack layout
        """
        results = {}
        
        for i, segment in enumerate(clip_segments):
            try:
                # Run detection for ALL clips (both talking_head and screen_share need face detection)
                detection_result = await self.detection_pipeline.process_video(
                    video_path=video_path,
                    start_time_ms=segment.start_time_ms,
                    end_time_ms=segment.end_time_ms,
                    frame_interval_seconds=0.5,  # 2 FPS for detection
                )
                
                results[i] = detection_result.get("frames", [])
                logger.info(f"Detection for clip {i} ({segment.layout_type}): {len(results[i])} frames")
                
            except Exception as e:
                logger.warning(f"Detection failed for clip {i}: {e}")
                results[i] = []
        
        return results

    def _build_crop_timeline(
        self,
        detection_frames: Optional[list[dict]],
        segment: ClipPlanSegment,
        source_width: int,
        source_height: int,
    ) -> Optional[CropTimeline]:
        """
        Build a crop timeline from detection results for face tracking.

        SIMPLIFIED to match epiriumaiclips architecture:
        - For talking_head: Full 9:16 crop window following face
        - For screen_share: Tight crop around face for bottom region
        - Simple fallback to center-bottom when no faces detected
        """
        # Calculate crop window based on layout type
        if segment.layout_type == "screen_share":
            # For screen_share: tight crop around face for close-up effect
            face_output_height = int(self.settings.target_output_height * self.settings.opusclip_face_ratio)
            face_crop_aspect = self.settings.target_output_width / face_output_height

            # Calculate face size from detections (like epiriumaiclips)
            face_areas = []
            if detection_frames:
                for frame in detection_frames:
                    for face in frame.get("detections", []):
                        bbox = face.get("bbox", {})
                        face_w = bbox.get("width", 0)
                        face_h = bbox.get("height", 0)
                        if face_w > 30 and face_h > 30:
                            face_areas.append(face_w * face_h)

            if face_areas:
                # Use average face area (like epiriumaiclips)
                avg_face_area = sum(face_areas) / len(face_areas)
                avg_face_size = int(avg_face_area ** 0.5)

                # Create TIGHT crop: 2x face size (matching epiriumaiclips)
                # This smaller crop will scale UP dramatically to fill the bottom region
                face_crop_size = avg_face_size * 2

                # Ensure minimum reasonable size (face + context)
                min_crop = min(source_width, source_height) // 4  # 1/4 of frame
                face_crop_size = max(face_crop_size, min_crop)

                # Cap at 1/3 of source to ensure good scale-up
                max_crop = min(source_width, source_height) // 3
                face_crop_size = min(face_crop_size, max_crop)

                # Make window square for best scaling
                window_height = face_crop_size
                window_width = face_crop_size  # Keep square for clean scale-up

                logger.info(f"Face crop (tight): avg_face_size={avg_face_size}, crop={face_crop_size}x{face_crop_size}")
            else:
                # Fallback: smaller crop from center-bottom for better scale-up
                # Use 1/3 of height to ensure it scales up to fill
                window_height = source_height // 3
                window_width = window_height  # Keep square
                logger.info(f"No faces - using center-bottom fallback: window={window_width}x{window_height}")
        else:
            # For talking_head: standard 9:16 crop
            target_aspect = 9 / 16
            if source_width / source_height > target_aspect:
                window_height = source_height
                window_width = int(source_height * target_aspect)
            else:
                window_width = source_width
                window_height = int(source_width / target_aspect)
        
        frame_area = source_width * source_height
        
        # Collect raw face positions using WEIGHTED AVERAGE (epiriumaiclips style)
        raw_positions: list[tuple[int, int, int]] = []  # (timestamp_ms, center_x, center_y)
        
        for frame in detection_frames:
            timestamp_ms = int(frame.get("timestamp_sec", 0) * 1000)
            faces = frame.get("detections", [])
            
            if faces:
                # Calculate weighted average position using confidence × area
                # This prioritizes large, high-confidence faces over small/uncertain ones
                total_weight = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                
                for face in faces:
                    bbox = face.get("bbox", {})
                    confidence = face.get("confidence", 0.5)
                    face_w = bbox.get("width", 0)
                    face_h = bbox.get("height", 0)
                    face_area = face_w * face_h
                    
                    # Skip very small faces (likely false positives)
                    area_ratio = face_area / frame_area if frame_area > 0 else 0
                    if area_ratio < 0.005 or area_ratio > 0.30:
                        continue
                    
                    # Weight = confidence × area (multiplicative weighting)
                    weight = confidence * face_area
                    
                    center_x = bbox.get("x", 0) + face_w / 2
                    center_y = bbox.get("y", 0) + face_h / 2
                    
                    weighted_x += center_x * weight
                    weighted_y += center_y * weight
                    total_weight += weight
                
                if total_weight > 0:
                    # Calculate final weighted position
                    final_x = int(weighted_x / total_weight)
                    final_y = int(weighted_y / total_weight)
                    
                    # Apply UPPER BIAS for better face framing (epiriumaiclips style)
                    # Shift the target Y position up by 10% of window height
                    # This places the face in the upper portion of the crop frame
                    upper_bias = int(window_height * 0.1)
                    final_y = max(0, final_y - upper_bias)
                    
                    raw_positions.append((timestamp_ms, final_x, final_y))
                    logger.debug(f"Weighted face position: ({final_x}, {final_y}) from {len(faces)} faces, total_weight={total_weight:.0f}")
                else:
                    # All faces filtered out - use last known or fallback
                    if raw_positions:
                        _, last_x, last_y = raw_positions[-1]
                        raw_positions.append((timestamp_ms, last_x, last_y))
                    else:
                        # Fallback: center-bottom (like epiriumaiclips)
                        center_x = source_width // 2
                        if segment.layout_type == "screen_share":
                            center_y = int(source_height * 0.75)  # Center-bottom area
                        else:
                            center_y = int(source_height * 0.4)
                        raw_positions.append((timestamp_ms, center_x, center_y))
            else:
                # No face detected - use previous position or center-bottom fallback
                if raw_positions:
                    # Use last known position for continuity
                    _, last_x, last_y = raw_positions[-1]
                    raw_positions.append((timestamp_ms, last_x, last_y))
                else:
                    # Fallback: center-bottom (matching epiriumaiclips)
                    center_x = source_width // 2
                    if segment.layout_type == "screen_share":
                        center_y = int(source_height * 0.75)  # Center-bottom
                    else:
                        center_y = int(source_height * 0.4)  # Upper center for talking head
                    raw_positions.append((timestamp_ms, center_x, center_y))
        
        if not raw_positions:
            # Should not happen given the logic above, but safe fallback
            # Use center-bottom for screen_share, centered for others
            center_x = source_width // 2
            if segment.layout_type == "screen_share":
                center_y = int(source_height * 0.75)
            else:
                center_y = source_height // 2
            raw_positions = [(segment.start_time_ms, center_x, center_y)]
        
        # Apply outlier rejection to remove sudden position jumps
        raw_positions = self._filter_position_outliers(raw_positions)
        
        # Apply smoothing to reduce camera jitter
        smoothed_positions = self._smooth_positions(raw_positions, window_size=3)
        
        # Build keyframes from smoothed positions
        keyframes = [
            CropKeyframe(timestamp_ms=ts, center_x=cx, center_y=cy)
            for ts, cx, cy in smoothed_positions
        ]
        
        return CropTimeline(
            window_width=window_width,
            window_height=window_height,
            source_width=source_width,
            source_height=source_height,
            keyframes=keyframes,
        )
    
    def _filter_position_outliers(
        self,
        positions: list[tuple[int, int, int]],
        std_threshold: float = 2.0,
    ) -> list[tuple[int, int, int]]:
        """
        Filter outlier positions using statistical analysis (epiriumaiclips style).
        
        Removes positions more than std_threshold standard deviations from median.
        This prevents sudden crop jumps from false positive detections.
        """
        if len(positions) < 3:
            return positions
        
        try:
            import numpy as np
            
            x_values = [p[1] for p in positions]
            y_values = [p[2] for p in positions]
            
            median_x = np.median(x_values)
            median_y = np.median(y_values)
            std_x = np.std(x_values)
            std_y = np.std(y_values)
            
            filtered = []
            for ts, x, y in positions:
                x_ok = std_x == 0 or abs(x - median_x) <= std_threshold * std_x
                y_ok = std_y == 0 or abs(y - median_y) <= std_threshold * std_y
                
                if x_ok and y_ok:
                    filtered.append((ts, x, y))
                else:
                    # Replace outlier with median position but keep timestamp
                    filtered.append((ts, int(median_x), int(median_y)))
                    logger.debug(f"Replaced outlier position ({x}, {y}) with median ({int(median_x)}, {int(median_y)})")
            
            return filtered if filtered else positions
            
        except Exception as e:
            logger.warning(f"Position outlier filtering failed: {e}")
            return positions

    def _smooth_positions(
        self,
        positions: list[tuple[int, int, int]],
        window_size: int = 3,
    ) -> list[tuple[int, int, int]]:
        """
        Apply moving average smoothing to reduce camera jitter.
        
        Args:
            positions: List of (timestamp_ms, center_x, center_y)
            window_size: Number of frames to average
            
        Returns:
            Smoothed positions
        """
        if len(positions) <= window_size:
            return positions
        
        smoothed = []
        half_window = window_size // 2
        
        for i, (ts, _, _) in enumerate(positions):
            # Get window of positions
            start_idx = max(0, i - half_window)
            end_idx = min(len(positions), i + half_window + 1)
            
            # Calculate average position
            window_x = [positions[j][1] for j in range(start_idx, end_idx)]
            window_y = [positions[j][2] for j in range(start_idx, end_idx)]
            
            avg_x = int(sum(window_x) / len(window_x))
            avg_y = int(sum(window_y) / len(window_y))
            
            smoothed.append((ts, avg_x, avg_y))
        
        return smoothed
    
    def _build_screen_timeline(
        self,
        segment: ClipPlanSegment,
        source_width: int,
        source_height: int,
        detection_frames: Optional[list[dict]] = None,
    ) -> CropTimeline:
        """
        Build a crop timeline for the screen portion of a screen_share layout.

        SIMPLIFIED to match epiriumaiclips:
        - Always take a SQUARE crop from the CENTER of the frame
        - No webcam position detection needed
        - The screen content is typically centered in stream layouts
        """
        # epiriumaiclips approach: square crop from center
        # square_size = min(original_width, original_height)
        # x_start = (original_width - square_size) // 2
        # y_start = (original_height - square_size) // 2
        square_size = min(source_width, source_height)
        center_x = source_width // 2
        center_y = source_height // 2

        logger.info(f"Screen crop (epiriumaiclips style): square {square_size}x{square_size} from center")

        return CropTimeline(
            window_width=square_size,
            window_height=square_size,
            source_width=source_width,
            source_height=source_height,
            keyframes=[
                CropKeyframe(
                    timestamp_ms=segment.start_time_ms,
                    center_x=center_x,
                    center_y=center_y,
                ),
                CropKeyframe(
                    timestamp_ms=segment.end_time_ms,
                    center_x=center_x,
                    center_y=center_y,
                ),
            ],
        )

    def _build_centered_timeline(
        self,
        segment: ClipPlanSegment,
        source_width: int,
        source_height: int,
    ) -> CropTimeline:
        """Build a centered static crop timeline."""
        target_aspect = 9 / 16

        if source_width / source_height > target_aspect:
            window_height = source_height
            window_width = int(source_height * target_aspect)
        else:
            window_width = source_width
            window_height = int(source_width / target_aspect)

        center_x = source_width // 2
        center_y = source_height // 2

        return CropTimeline(
            window_width=window_width,
            window_height=window_height,
            source_width=source_width,
            source_height=source_height,
            keyframes=[
                CropKeyframe(
                    timestamp_ms=segment.start_time_ms,
                    center_x=center_x,
                    center_y=center_y,
                ),
                CropKeyframe(
                    timestamp_ms=segment.end_time_ms,
                    center_x=center_x,
                    center_y=center_y,
                ),
            ],
        )


    def _filter_transcript_for_clip(
        self,
        all_segments,
        start_time_ms: int,
        end_time_ms: int,
    ):
        """Filter transcript segments that overlap with clip timeframe."""
        from app.services.transcription_service import TranscriptSegment
        
        filtered = []
        
        for seg in all_segments:
            # Check if segment overlaps with clip
            if seg.end_time_ms <= start_time_ms:
                continue
            if seg.start_time_ms >= end_time_ms:
                continue
            
            filtered.append(seg)
        
        return filtered

    def _update_progress(
        self,
        job_id: str,
        status: JobStatus,
        progress: float,
        step: str,
        clips_completed: int = 0,
        total_clips: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update job progress via callback."""
        if self.progress_callback:
            try:
                self.progress_callback(ClippingJobProgress(
                    job_id=job_id,
                    status=status,
                    progress_percent=progress,
                    current_step=step,
                    clips_completed=clips_completed,
                    total_clips=total_clips,
                    error=error,
                ))
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

