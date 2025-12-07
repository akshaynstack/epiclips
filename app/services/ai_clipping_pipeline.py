"""
AI Clipping Pipeline - Unified orchestrator for the complete AI clipping workflow.

This service orchestrates the entire AI clipping pipeline:
1. Video download (YouTube via yt-dlp)
2. Audio extraction and transcription (Groq Whisper)
3. Intelligence planning (Gemini via OpenRouter)
4. Detection analysis (local MediaPipe with parallel processing)
5. Clip rendering (FFmpeg with captions - parallel execution)
6. S3 upload (parallel uploads)
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
    RenderResult,
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
from app.services.smart_layout_detector import (
    SmartLayoutDetector,
    LayoutAnalysis,
    LayoutSegment,
)
from app.services.webhook_service import (
    WebhookService,
    get_webhook_service,
)
from app.services.face_tracker import (
    FaceTracker,
    track_faces_in_video,
    get_dominant_face_crop_region,
    TrackingResult,
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
    external_job_id: Optional[str] = None  # External reference (e.g., API job ID)
    owner_user_id: Optional[str] = None  # User ID for S3 key scoping
    max_clips: int = 5
    min_clip_duration_seconds: int = 15
    max_clip_duration_seconds: int = 90
    duration_ranges: Optional[list[str]] = None  # ['short', 'medium', 'long']
    target_platform: str = "tiktok"  # tiktok, youtube_shorts, instagram_reels
    include_captions: bool = True
    caption_style: Optional[CaptionStyle] = None
    layout_type: str = "split_screen"  # split_screen, full_screen, talking_head
    callback_url: Optional[str] = None  # URL to receive webhook notifications

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
        webhook_service: Optional[WebhookService] = None,
    ):
        self.settings = get_settings()
        self.detection_pipeline = detection_pipeline
        self.progress_callback = progress_callback
        self.webhook_service = webhook_service or get_webhook_service()

        # Initialize services
        self.video_downloader = VideoDownloaderService()
        self.transcription_service = TranscriptionService()
        self.intelligence_planner = IntelligencePlannerService()
        self.rendering_service = RenderingService()
        self.s3_upload_service = S3UploadService()
        self.content_region_detector = ContentRegionDetector()
        self.smart_layout_detector = SmartLayoutDetector()

        # Current request context for webhooks (set during process_video)
        self._current_callback_url: Optional[str] = None
        self._current_external_job_id: Optional[str] = None
        self._current_owner_user_id: Optional[str] = None
        
        # Job-specific file handler for logging
        self._job_log_handler: Optional[logging.FileHandler] = None

    def _setup_job_logging(self, job_id: str) -> Optional[logging.FileHandler]:
        """
        Set up job-specific file logging.
        
        Creates a log file in the logs folder for this specific job.
        All log messages from app.services.* will be written to this file.
        
        Args:
            job_id: The job ID to use for the log filename
            
        Returns:
            The file handler (to be removed later) or None if setup fails
        """
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Create log file with job_id and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = logs_dir / f"job_{job_id}_{timestamp}.log"
            
            # Create file handler with detailed formatting
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed format for file logs
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to the root logger to capture all app.services.* logs
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            
            # Also add to specific service loggers for better coverage
            service_loggers = [
                'app.services.ai_clipping_pipeline',
                'app.services.smart_layout_detector',
                'app.services.rendering_service',
                'app.services.detection_pipeline',
                'app.services.transcription_service',
                'app.services.intelligence_planner',
                'app.services.video_downloader',
                'app.services.s3_upload_service',
            ]
            for logger_name in service_loggers:
                logging.getLogger(logger_name).addHandler(file_handler)
            
            logger.info(f"Job logging initialized: {log_filename}")
            return file_handler
            
        except Exception as e:
            logger.warning(f"Failed to setup job logging: {e}")
            return None
    
    def _cleanup_job_logging(self, file_handler: Optional[logging.FileHandler]):
        """Remove the job-specific file handler from all loggers."""
        if file_handler is None:
            return
            
        try:
            # Remove from root logger
            root_logger = logging.getLogger()
            root_logger.removeHandler(file_handler)
            
            # Remove from service loggers
            service_loggers = [
                'app.services.ai_clipping_pipeline',
                'app.services.smart_layout_detector',
                'app.services.rendering_service',
                'app.services.detection_pipeline',
                'app.services.transcription_service',
                'app.services.intelligence_planner',
                'app.services.video_downloader',
                'app.services.s3_upload_service',
            ]
            for logger_name in service_loggers:
                logging.getLogger(logger_name).removeHandler(file_handler)
            
            # Close the file handler
            file_handler.close()
            
        except Exception as e:
            logger.warning(f"Failed to cleanup job logging: {e}")

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

        # Set webhook context for this request
        self._current_callback_url = request.callback_url
        self._current_external_job_id = request.external_job_id
        self._current_owner_user_id = request.owner_user_id
        
        # Set up job-specific file logging
        job_log_handler = self._setup_job_logging(job_id)

        try:
            os.makedirs(work_dir, exist_ok=True)
            logger.info(f"Starting AI clipping job: {job_id}")
            logger.info(f"Video URL: {request.video_url}")
            logger.info(f"Max clips: {request.max_clips}, Duration ranges: {request.duration_ranges}")
            logger.info(f"Layout type: {request.layout_type}, Include captions: {request.include_captions}")
            logger.info(f"Webhook callback URL: {self._current_callback_url or 'NOT SET'}")
            
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
                layout_type=request.layout_type,
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
            
            # Step 4: Run detection for smart cropping (with speaker-driven layout)
            self._update_progress(job_id, JobStatus.DETECTING, 45, "Analyzing video for smart cropping...")
            detection_results = await self._run_detection_for_clips(
                video_path=download_result.video_path,
                clip_segments=clip_plan.segments,
                source_width=download_result.metadata.width,
                source_height=download_result.metadata.height,
                layout_type=request.layout_type,  # Pass user's layout choice (auto, split_screen, talking_head)
                transcript_segments=transcription_result.segments,  # Pass for speaker diarization
                enable_speaker_driven_layout=True,  # Use active speaker detection for layout
            )
            logger.info("Detection analysis complete")
            
            # Log layout analysis results for debugging
            for idx, clip_data in detection_results.items():
                layout_analysis = clip_data.get('layout_analysis')
                if layout_analysis:
                    logger.info(
                        f"Clip {idx} detection result: has_transitions={layout_analysis.has_transitions}, "
                        f"segments={len(layout_analysis.layout_segments)}, "
                        f"dominant={layout_analysis.dominant_layout}"
                    )
            
            # Step 5: Render clips IN PARALLEL
            clips_dir = os.path.join(work_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            
            total_clips = len(clip_plan.segments)
            self._update_progress(
                job_id, JobStatus.RENDERING, 50,
                f"Rendering {total_clips} clips in parallel...",
                clips_completed=0, total_clips=total_clips,
            )
            
            # Prepare render tasks for parallel execution
            # Limit concurrent FFmpeg processes to avoid memory exhaustion
            render_semaphore = asyncio.Semaphore(self.settings.max_concurrent_renders)  # Configurable via MAX_RENDER_WORKERS
            
            async def render_single_clip(i: int, segment: ClipPlanSegment) -> tuple[int, str, ClipPlanSegment]:
                """Render a single clip with semaphore-based concurrency control."""
                async with render_semaphore:
                    output_path = os.path.join(clips_dir, f"clip_{i:02d}.mp4")
                    
                    # Get detection results and layout analysis for this clip
                    clip_data = detection_results.get(i, {})
                    detection_frames = clip_data.get('detection_frames', [])
                    layout_analysis: Optional[LayoutAnalysis] = clip_data.get('layout_analysis')
                    
                    # Get transcript segments for this clip
                    clip_transcript = self._filter_transcript_for_clip(
                        transcription_result.segments,
                        segment.start_time_ms,
                        segment.end_time_ms,
                    )
                    
                    # Check if clip has layout transitions - use segment-based rendering
                    logger.info(
                        f"Clip {i + 1} layout check: layout_analysis={layout_analysis is not None}, "
                        f"has_transitions={layout_analysis.has_transitions if layout_analysis else 'N/A'}, "
                        f"segments={len(layout_analysis.layout_segments) if layout_analysis else 0}"
                    )
                    if layout_analysis and layout_analysis.has_transitions and len(layout_analysis.layout_segments) > 1:
                        logger.info(
                            f"Clip {i + 1} has {len(layout_analysis.layout_segments)} layout segments - "
                            f"using segment-based rendering"
                        )
                        
                        render_result = await self._render_with_layout_transitions(
                            video_path=download_result.video_path,
                            segment=segment,
                            layout_segments=layout_analysis.layout_segments,
                            detection_frames=detection_frames,
                            transcript_segments=clip_transcript if request.include_captions else None,
                            output_path=output_path,
                            source_width=download_result.metadata.width,
                            source_height=download_result.metadata.height,
                            caption_style=request.caption_style,
                        )
                    else:
                        # Single layout - use standard rendering
                        effective_layout = segment.layout_type
                        if layout_analysis and layout_analysis.layout_segments:
                            effective_layout = layout_analysis.layout_segments[0].layout_type
                        
                        crop_timeline = self._build_crop_timeline(
                            detection_frames,
                            segment,
                            download_result.metadata.width,
                            download_result.metadata.height,
                            effective_layout=effective_layout,  # Pass actual layout from smart detector
                        )
                        
                        screen_timeline = None
                        if effective_layout == "screen_share":
                            screen_timeline = self._build_screen_timeline(
                                segment,
                                download_result.metadata.width,
                                download_result.metadata.height,
                                detection_frames,
                            )
                        
                        # Adjust caption position based on layout type:
                        # - talking_head: bottom (natural look)
                        # - screen_share: center (between screen and face)
                        adjusted_caption_style = None
                        if request.caption_style:
                            adjusted_caption_style = CaptionStyle()
                            adjusted_caption_style.font_name = request.caption_style.font_name
                            adjusted_caption_style.font_size = request.caption_style.font_size
                            adjusted_caption_style.primary_color = request.caption_style.primary_color
                            adjusted_caption_style.highlight_color = request.caption_style.highlight_color
                            adjusted_caption_style.outline_color = request.caption_style.outline_color
                            adjusted_caption_style.outline_width = request.caption_style.outline_width
                            adjusted_caption_style.max_words_per_line = request.caption_style.max_words_per_line
                            adjusted_caption_style.word_by_word_highlight = request.caption_style.word_by_word_highlight
                            adjusted_caption_style.alignment = request.caption_style.alignment
                            adjusted_caption_style.bold = request.caption_style.bold
                            adjusted_caption_style.uppercase = request.caption_style.uppercase
                            
                            if effective_layout == "talking_head":
                                adjusted_caption_style.position = "bottom"
                            else:  # screen_share
                                adjusted_caption_style.position = "center"
                        
                        # DISABLED: has_embedded_facecam detection was blocking split screen
                        # Always set to False - we want split screen for face + screen content
                        has_embedded = False  # layout_analysis.has_embedded_facecam if layout_analysis else False
                        
                        # Get corner facecam bbox for proper face cropping in split screen
                        corner_facecam_bbox = None
                        if layout_analysis and layout_analysis.corner_facecam_bbox:
                            corner_facecam_bbox = layout_analysis.corner_facecam_bbox
                        
                        logger.info(
                            f"Clip {i + 1} render config: layout={effective_layout}, "
                            f"has_embedded_facecam={has_embedded} (disabled), "
                            f"corner_facecam_bbox={corner_facecam_bbox}"
                        )
                        
                        render_request = RenderRequest(
                            video_path=download_result.video_path,
                            output_path=output_path,
                            start_time_ms=segment.start_time_ms,
                            end_time_ms=segment.end_time_ms,
                            source_width=download_result.metadata.width,
                            source_height=download_result.metadata.height,
                            layout_type=effective_layout,
                            primary_timeline=crop_timeline,
                            secondary_timeline=screen_timeline,
                            transcript_segments=clip_transcript if request.include_captions else None,
                            caption_style=adjusted_caption_style,
                            has_embedded_facecam=has_embedded,
                            corner_facecam_bbox=corner_facecam_bbox,
                        )
                        
                        render_result = await self.rendering_service.render_clip(render_request)
                    
                    logger.info(f"Rendered clip {i + 1}: {render_result.file_size_bytes / 1024 / 1024:.1f} MB")
                    return (i, render_result.output_path, segment)
            
            # Execute all render tasks in parallel
            render_tasks = [
                render_single_clip(i, segment)
                for i, segment in enumerate(clip_plan.segments)
            ]
            render_results = await asyncio.gather(*render_tasks)
            
            # Sort by index to maintain order
            render_results.sort(key=lambda x: x[0])
            rendered_clips = [(path, segment) for _, path, segment in render_results]
            
            logger.info(f"All {len(rendered_clips)} clips rendered in parallel")
            
            # Step 6: Upload clips to S3 IN PARALLEL
            self._update_progress(
                job_id, JobStatus.UPLOADING, 90,
                "Uploading clips to storage...",
                clips_completed=total_clips, total_clips=total_clips,
            )
            
            async def upload_single_clip(i: int, clip_path: str, segment: ClipPlanSegment) -> ClipArtifact:
                """Upload a single clip to S3."""
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
                
                return ClipArtifact(
                    clip_index=i,
                    s3_url=upload_result.s3_url,
                    duration_ms=segment.end_time_ms - segment.start_time_ms,
                    start_time_ms=segment.start_time_ms,
                    end_time_ms=segment.end_time_ms,
                    virality_score=segment.virality_score,
                    layout_type=segment.layout_type,
                    summary=segment.summary,
                    tags=segment.tags or [],
                )
            
            # Execute all upload tasks in parallel
            upload_tasks = [
                upload_single_clip(i, clip_path, segment)
                for i, (clip_path, segment) in enumerate(rendered_clips)
            ]
            clip_artifacts = await asyncio.gather(*upload_tasks)
            
            # Sort by clip index to maintain order
            clip_artifacts = sorted(clip_artifacts, key=lambda x: x.clip_index)
            
            logger.info(f"All {len(clip_artifacts)} clips uploaded in parallel")
            
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
            
            # Build webhook output payload with clip URLs and metadata
            webhook_output = {
                "total_clips": len(clip_artifacts),
                "source_video_title": job_output.source_video_title,
                "processing_time_seconds": processing_time,
                "clips": [
                    {
                        "clip_index": clip.clip_index,
                        "s3_url": clip.s3_url,
                        "duration_ms": clip.duration_ms,
                        "start_time_ms": clip.start_time_ms,
                        "end_time_ms": clip.end_time_ms,
                        "virality_score": clip.virality_score,
                        "layout_type": clip.layout_type,
                        "summary": clip.summary,
                        "tags": clip.tags,
                    }
                    for clip in clip_artifacts
                ],
                "transcript_url": job_output.transcript_url,
                "plan_url": job_output.plan_url,
            }

            self._update_progress(
                job_id, JobStatus.COMPLETED, 100,
                "Processing complete!",
                clips_completed=total_clips, total_clips=total_clips,
                output=webhook_output,
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

            # Clear webhook context and tracking
            self.webhook_service.clear_job_tracking(job_id)
            self._current_callback_url = None
            self._current_external_job_id = None
            self._current_owner_user_id = None
            
            # Cleanup job-specific logging
            self._cleanup_job_logging(job_log_handler)

    async def _run_detection_for_clips(
        self,
        video_path: str,
        clip_segments: list[ClipPlanSegment],
        source_width: int,
        source_height: int,
        layout_type: str = "auto",
        transcript_segments: Optional[list] = None,
        enable_speaker_driven_layout: bool = True,
    ) -> dict[int, dict]:
        """
        Run detection pipeline and layout analysis for each planned clip segment IN PARALLEL.

        Returns a mapping of clip_index -> {
            'detection_frames': list of face detection frames,
            'layout_analysis': LayoutAnalysis with segments and transitions
        }

        Args:
            video_path: Path to video file
            clip_segments: List of planned clip segments
            source_width: Source video width
            source_height: Source video height
            layout_type: User-requested layout type:
                - "auto": Run SmartLayoutDetector to detect transitions
                - "split_screen"/"talking_head": Force this layout, skip transition detection
            transcript_segments: Optional transcript segments for speaker diarization
            enable_speaker_driven_layout: Whether to use speaker-driven layout (recommended)

        Note: Detection is run for ALL layout types to enable smart cropping:
        - talking_head: Uses face detection for dynamic panning
        - screen_share: Uses face detection for the face portion of stack layout

        Layout analysis enables dynamic layout switching within clips by:
        - Detecting layout transitions (talking_head <-> screen_share)
        - Creating layout segments for per-segment rendering
        
        NEW: Speaker-driven layout mode (when enable_speaker_driven_layout=True)
        - Uses active speaker detection instead of face count
        - SINGLE (talking_head) when only one speaker is active
        - SPLIT (screen_share) only when 2+ speakers are active
        - Background faces (< 2% speaking time) are ignored
        """
        use_auto_layout = layout_type == "auto"
        
        async def process_single_clip(i: int, segment: ClipPlanSegment) -> tuple[int, dict]:
            """Process detection for a single clip segment."""
            try:
                # Run detection for ALL clips (both talking_head and screen_share need face detection)
                detection_result = await self.detection_pipeline.process_video(
                    video_path=video_path,
                    start_time_ms=segment.start_time_ms,
                    end_time_ms=segment.end_time_ms,
                    frame_interval_seconds=0.5,  # 2 FPS for detection
                )

                detection_frames = detection_result.get("frames", [])
                logger.info(f"Detection for clip {i} ({segment.layout_type}): {len(detection_frames)} frames")

                if use_auto_layout:
                    # AUTO mode: Use speaker-driven layout if available, else fall back to frame analysis
                    if enable_speaker_driven_layout and hasattr(self.smart_layout_detector, 'analyze_clip_layout_with_speakers'):
                        # NEW: Speaker-driven layout detection
                        # - SINGLE (talking_head) when only one speaker is active
                        # - SPLIT (screen_share) only when 2+ speakers are active
                        # - Background faces are ignored
                        layout_analysis = await self.smart_layout_detector.analyze_clip_layout_with_speakers(
                            video_path=video_path,
                            start_ms=segment.start_time_ms,
                            end_ms=segment.end_time_ms,
                            face_detections=detection_frames,
                            transcript_segments=transcript_segments,
                            frame_width=source_width,
                            frame_height=source_height,
                            sample_fps=2.0,
                        )
                        
                        logger.info(
                            f"Speaker-driven layout for clip {i}: "
                            f"has_transitions={layout_analysis.has_transitions}, "
                            f"segments={layout_analysis.segment_count}, "
                            f"dominant={layout_analysis.dominant_layout}, "
                            f"active_speakers={layout_analysis.active_speaker_count}"
                        )
                        
                        # NOTE: We trust the frame-by-frame analysis for layout transitions.
                        # The AI planner's suggestion is only used as context, not as override.
                        # This allows dynamic layout switching within clips.
                        
                        # NEW: Run face tracking for screen_share layouts to get dominant face
                        # ALWAYS run tracking to verify/refine VLM detection
                        if layout_analysis.dominant_layout == "screen_share":
                            try:
                                # Use VLM's detected webcam position as a hint for the tracker
                                preferred_pos = getattr(layout_analysis, 'webcam_position', None)
                                
                                tracking_result = await track_faces_in_video(
                                    video_path=video_path,
                                    start_ms=segment.start_time_ms,
                                    end_ms=segment.end_time_ms,
                                    sample_fps=5.0,  # 5 FPS for tracking
                                    preferred_position=preferred_pos,
                                )
                                if tracking_result.dominant_track and tracking_result.dominant_track.smoothed_bbox:
                                    dom_bbox = tracking_result.dominant_track.smoothed_bbox
                                    dom_score = tracking_result.dominant_track.dominance_score
                                    x, y, w, h = dom_bbox
                                    face_area_ratio = (w * h) / (source_width * source_height) if source_width > 0 and source_height > 0 else 0
                                    
                                    # Check if the dominant face is actually in the requested quadrant
                                    # (negative score means it was disqualified from position matching)
                                    if dom_score < 0:
                                        logger.warning(
                                            f"Face tracking: no face in {preferred_pos} quadrant. "
                                            f"Best match at ({x},{y}) has score {dom_score:.2f}. Using fallback bbox."
                                        )
                                        # Use VLM position-based fallback instead
                                        fallback_bbox = self._estimate_webcam_bbox_from_position(
                                            preferred_pos, source_width, source_height
                                        )
                                        if fallback_bbox:
                                            layout_analysis.corner_facecam_bbox = fallback_bbox
                                            for seg in layout_analysis.layout_segments:
                                                if seg.layout_type == "screen_share":
                                                    seg.corner_facecam_bbox = fallback_bbox
                                            logger.info(f"Using VLM position-based fallback bbox: {fallback_bbox}")
                                    # QUALITY GATE: Face must be small (< 10%) - visibility threshold removed
                                    # When VLM detects webcam, we trust it and just validate the face is corner-sized
                                    elif face_area_ratio < 0.10:
                                        layout_analysis.corner_facecam_bbox = dom_bbox
                                        for seg in layout_analysis.layout_segments:
                                            if seg.layout_type == "screen_share":
                                                seg.corner_facecam_bbox = dom_bbox
                                        logger.info(
                                            f"Face tracking found corner webcam: {w}x{h} at ({x},{y}), "
                                            f"visibility={tracking_result.dominant_face_visibility:.1%}, "
                                            f"preferred_pos={preferred_pos}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Face tracking rejected: area_ratio={face_area_ratio:.3f} (limit 0.10) - face too large for corner webcam"
                                        )
                                        # Large face = not a corner webcam, but also not necessarily full_screen
                                        # Keep the layout as is, just don't set corner_facecam_bbox
                                else:
                                    # No dominant face found by FaceTracker
                                    # If VLM already set a bbox estimate, keep it; otherwise log warning
                                    if not layout_analysis.corner_facecam_bbox:
                                        logger.warning(f"Clip {i}: Detected screen_share but no dominant face found. Using VLM's webcam position hint for rendering.")
                                        # Generate a fallback bbox based on VLM's position hint
                                        preferred_pos = getattr(layout_analysis, 'webcam_position', None)
                                        if preferred_pos:
                                            # Create estimated bbox based on position (bottom-right, etc.)
                                            fallback_bbox = self._estimate_webcam_bbox_from_position(
                                                preferred_pos, source_width, source_height
                                            )
                                            if fallback_bbox:
                                                layout_analysis.corner_facecam_bbox = fallback_bbox
                                                for seg in layout_analysis.layout_segments:
                                                    if seg.layout_type == "screen_share":
                                                        seg.corner_facecam_bbox = fallback_bbox
                                                logger.info(f"Using VLM position-based fallback bbox: {fallback_bbox}")
                            except Exception as e:
                                logger.debug(f"Face tracking for clip {i} failed (non-critical): {e}")
                    else:
                        # Fallback to frame-based layout analysis
                        layout_analysis = await self.smart_layout_detector.analyze_clip_layout(
                            video_path=video_path,
                            start_ms=segment.start_time_ms,
                            end_ms=segment.end_time_ms,
                            face_detections=detection_frames,
                            sample_fps=2.0,  # Sample at 2 FPS for layout analysis
                        )

                        logger.info(
                            f"Layout analysis for clip {i}: "
                            f"has_transitions={layout_analysis.has_transitions}, "
                            f"segments={layout_analysis.segment_count}, "
                            f"dominant={layout_analysis.dominant_layout}"
                        )
                        
                        # NEW: Run face tracking for screen_share layouts to get dominant face
                        # ALWAYS run tracking to verify/refine VLM detection
                        if layout_analysis.dominant_layout == "screen_share":
                            try:
                                # Use VLM's detected webcam position as a hint for the tracker
                                preferred_pos = layout_analysis.webcam_position
                                
                                tracking_result = await track_faces_in_video(
                                    video_path=video_path,
                                    start_ms=segment.start_time_ms,
                                    end_ms=segment.end_time_ms,
                                    sample_fps=5.0,
                                    preferred_position=preferred_pos,
                                )
                                has_valid_face = False
                                if tracking_result.dominant_track and tracking_result.dominant_track.smoothed_bbox:
                                    dom_bbox = tracking_result.dominant_track.smoothed_bbox
                                    x, y, w, h = dom_bbox
                                    face_area_ratio = (w * h) / (source_width * source_height) if source_width > 0 and source_height > 0 else 0
                                    
                                    # QUALITY GATE: Face must be small (<10%) AND persistent (>4% visibility)
                                    # Lowered from 20% to 4% to support static webcams that are only intermittently tracked
                                    if face_area_ratio < 0.10 and tracking_result.dominant_face_visibility > 0.04:
                                        has_valid_face = True
                                        layout_analysis.corner_facecam_bbox = dom_bbox
                                        for seg in layout_analysis.layout_segments:
                                            if seg.layout_type == "screen_share":
                                                seg.corner_facecam_bbox = dom_bbox
                                        logger.info(
                                            f"Face tracking found corner webcam: {w}x{h} at ({x},{y}), "
                                            f"visibility={tracking_result.dominant_face_visibility:.1%}, "
                                            f"preferred_pos={preferred_pos}"
                                        )
                                    elif face_area_ratio < 0.10:
                                        logger.warning(
                                            f"Face tracking rejected: visibility={tracking_result.dominant_face_visibility:.1%} (limit 4%), "
                                            f"bbox={dom_bbox}"
                                        )

                                if not has_valid_face:
                                    # VLM detected screen_share but FaceTracker couldn't verify
                                    # Instead of falling back to full_screen, trust VLM and estimate bbox
                                    if layout_analysis.webcam_position:
                                        logger.warning(
                                            f"Clip {i}: VLM detected webcam at '{layout_analysis.webcam_position}' but FaceTracker rejected it. "
                                            f"Trusting VLM detection and estimating bbox."
                                        )
                                        # Estimate webcam bbox from VLM position
                                        bbox_tuple = self._estimate_webcam_bbox_from_position(
                                            layout_analysis.webcam_position,
                                            source_width,
                                            source_height,
                                        )
                                        layout_analysis.corner_facecam_bbox = bbox_tuple
                                        for seg in layout_analysis.layout_segments:
                                            if seg.layout_type == "screen_share":
                                                seg.corner_facecam_bbox = bbox_tuple
                                    else:
                                        # No VLM webcam position hint - must fallback
                                        logger.warning(f"Clip {i}: Detected screen_share but no valid dominant face and no VLM position. Falling back to full_screen.")
                                        layout_analysis.dominant_layout = "full_screen"
                                        layout_analysis.corner_facecam_bbox = None
                                        for seg in layout_analysis.layout_segments:
                                            seg.layout_type = "full_screen"
                                            seg.corner_facecam_bbox = None
                            except Exception as e:
                                logger.debug(f"Face tracking failed (non-critical): {e}")
                else:
                    # EXPLICIT mode: User specified a layout, force it for entire clip
                    # Map split_screen -> screen_share for rendering compatibility
                    forced_layout = "screen_share" if layout_type == "split_screen" else layout_type

                    layout_analysis = LayoutAnalysis(
                        has_transitions=False,
                        dominant_layout=forced_layout,
                        layout_segments=[
                            LayoutSegment(
                                start_ms=segment.start_time_ms,
                                end_ms=segment.end_time_ms,
                                layout_type=forced_layout,
                                confidence=1.0,  # User explicitly chose this
                            )
                        ],
                    )
                    logger.info(
                        f"Using forced layout '{forced_layout}' for clip {i} (user selected: {layout_type})"
                    )
                    
                    # NEW: For explicit screen_share/split_screen, run face tracking to find webcam
                    if forced_layout == "screen_share":
                        try:
                            tracking_result = await track_faces_in_video(
                                video_path=video_path,
                                start_ms=segment.start_time_ms,
                                end_ms=segment.end_time_ms,
                                sample_fps=5.0,
                            )
                            if tracking_result.dominant_track and tracking_result.dominant_track.smoothed_bbox:
                                dom_bbox = tracking_result.dominant_track.smoothed_bbox
                                x, y, w, h = dom_bbox
                                face_area_ratio = (w * h) / (source_width * source_height) if source_width > 0 and source_height > 0 else 0
                                
                                # For explicit mode, be more lenient with face size (< 20%)
                                if face_area_ratio < 0.20:
                                    layout_analysis.corner_facecam_bbox = dom_bbox
                                    layout_analysis.layout_segments[0].corner_facecam_bbox = dom_bbox
                                    logger.info(
                                        f"Face tracking for explicit split_screen: found face {w}x{h} at ({x},{y}), "
                                        f"visibility={tracking_result.dominant_face_visibility:.1%}"
                                    )
                                else:
                                    # Face too large - estimate a corner region instead
                                    logger.warning(f"Clip {i}: Face too large for split_screen ({face_area_ratio:.1%}). Estimating corner webcam.")
                                    bbox_tuple = self._estimate_webcam_bbox_from_position("bottom-right", source_width, source_height)
                                    layout_analysis.corner_facecam_bbox = bbox_tuple
                                    layout_analysis.layout_segments[0].corner_facecam_bbox = bbox_tuple
                            else:
                                # No dominant face found - estimate a corner region
                                logger.warning(f"Clip {i}: Explicit split_screen requested but no dominant face found. Estimating bottom-right webcam.")
                                bbox_tuple = self._estimate_webcam_bbox_from_position("bottom-right", source_width, source_height)
                                layout_analysis.corner_facecam_bbox = bbox_tuple
                                layout_analysis.layout_segments[0].corner_facecam_bbox = bbox_tuple
                        except Exception as e:
                            logger.debug(f"Face tracking for explicit layout failed (non-critical): {e}")

                return (i, {
                    'detection_frames': detection_frames,
                    'layout_analysis': layout_analysis,
                })

            except Exception as e:
                logger.warning(f"Detection/layout analysis failed for clip {i}: {e}")
                # Fallback: no transitions, use planned layout or forced layout
                fallback_layout = segment.layout_type
                if not use_auto_layout:
                    fallback_layout = "screen_share" if layout_type == "split_screen" else layout_type

                return (i, {
                    'detection_frames': [],
                    'layout_analysis': LayoutAnalysis(
                        has_transitions=False,
                        dominant_layout=fallback_layout,
                        layout_segments=[
                            LayoutSegment(
                                start_ms=segment.start_time_ms,
                                end_ms=segment.end_time_ms,
                                layout_type=fallback_layout,
                                confidence=0.5,
                            )
                        ],
                    ),
                })
        
        # Run all detection tasks in parallel
        detection_tasks = [
            process_single_clip(i, segment)
            for i, segment in enumerate(clip_segments)
        ]
        detection_results_list = await asyncio.gather(*detection_tasks)
        
        # Convert to dict
        results = {i: data for i, data in detection_results_list}
        logger.info(f"Completed parallel detection for {len(results)} clips")
        
        return results

    def _build_crop_timeline(
        self,
        detection_frames: Optional[list[dict]],
        segment: ClipPlanSegment,
        source_width: int,
        source_height: int,
        effective_layout: Optional[str] = None,
        corner_facecam_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[CropTimeline]:
        """
        Build a crop timeline from detection results for face tracking.

        SIMPLIFIED LOGIC:
        - For talking_head: 9:16 crop CENTERED on frame (no face tracking needed)
        - For screen_share: Tight crop for face region (corner_facecam_bbox handles position)
        
        Args:
            effective_layout: Override layout type from smart detector (takes priority over segment.layout_type)
            corner_facecam_bbox: (x, y, w, h) of webcam overlay for filtering faces in screen_share mode
        """
        # Use effective_layout from smart detector if provided, otherwise fallback to segment's layout
        layout_type = effective_layout if effective_layout else segment.layout_type
        
        # Calculate crop window based on layout type
        if layout_type == "screen_share":
            # For screen_share: tight crop around face for close-up effect
            # The actual position comes from corner_facecam_bbox in the renderer
            face_output_height = int(self.settings.target_output_height * self.settings.split_face_ratio)

            # Calculate face size from detections
            face_areas = []
            filtered_faces_count = 0
            total_faces_count = 0
            
            if detection_frames:
                for frame in detection_frames:
                    for face in frame.get("detections", []):
                        bbox = face.get("bbox", {})
                        face_w = bbox.get("width", 0)
                        face_h = bbox.get("height", 0)
                        face_x = bbox.get("x", 0) + face_w / 2  # Face center X
                        face_y = bbox.get("y", 0) + face_h / 2  # Face center Y
                        
                        if face_w > 30 and face_h > 30:
                            total_faces_count += 1
                            
                            # If we have corner_facecam_bbox, ONLY use faces near that location
                            # This filters out large faces from whiteboard sections
                            if corner_facecam_bbox:
                                webcam_x, webcam_y, webcam_w, webcam_h = corner_facecam_bbox
                                webcam_center_x = webcam_x + webcam_w / 2
                                webcam_center_y = webcam_y + webcam_h / 2
                                
                                # Only count faces within webcam region (allow 50% margin)
                                margin = 1.5
                                if (abs(face_x - webcam_center_x) < webcam_w * margin and
                                    abs(face_y - webcam_center_y) < webcam_h * margin):
                                    face_areas.append(face_w * face_h)
                                    filtered_faces_count += 1
                            else:
                                # No bbox filter - use all faces
                                face_areas.append(face_w * face_h)
                                filtered_faces_count += 1
            
            if corner_facecam_bbox and total_faces_count > 0:
                logger.info(
                    f"Webcam face filter: {filtered_faces_count}/{total_faces_count} faces kept "
                    f"(bbox center: {webcam_center_x:.0f},{webcam_center_y:.0f}, margin: {webcam_w * margin:.0f}px)"
                )

            if face_areas:
                # Use average face area
                avg_face_area = sum(face_areas) / len(face_areas)
                avg_face_size = int(avg_face_area ** 0.5)

                # Create TIGHT crop: 2x face size
                face_crop_size = avg_face_size * 2

                # Ensure minimum reasonable size (face + context)
                min_crop = min(source_width, source_height) // 4
                face_crop_size = max(face_crop_size, min_crop)

                # Cap at 1/3 of source to ensure good scale-up
                max_crop = min(source_width, source_height) // 3
                face_crop_size = min(face_crop_size, max_crop)

                window_height = face_crop_size
                window_width = face_crop_size

                logger.info(f"Face crop (tight): avg_face_size={avg_face_size}, crop={face_crop_size}x{face_crop_size}")
            else:
                window_height = source_height // 3
                window_width = window_height
                logger.info(f"No faces - using fallback crop: window={window_width}x{window_height}")
        else:
            # For talking_head: standard 9:16 crop CENTERED on frame
            # No face tracking - just center the crop horizontally
            target_aspect = 9 / 16
            if source_width / source_height > target_aspect:
                window_height = source_height
                window_width = int(source_height * target_aspect)
            else:
                window_width = source_width
                window_height = int(source_width / target_aspect)
            logger.info(f"Talking head crop: {window_width}x{window_height} (9:16 from {source_width}x{source_height})")
        
        # For talking_head: Track face position to center crop on the face
        # This is important when transitioning from webcam overlay to full-screen talking head
        if layout_type != "screen_share":
            # Try to find faces in the detection frames
            face_positions = []
            frame_area = source_width * source_height
            
            if detection_frames:
                for frame in detection_frames:
                    faces = frame.get("detections", [])
                    for face in faces:
                        bbox = face.get("bbox", {})
                        face_w = bbox.get("width", 0)
                        face_h = bbox.get("height", 0)
                        face_x = bbox.get("x", 0)
                        face_y = bbox.get("y", 0)
                        face_area = face_w * face_h
                        
                        # For talking_head, we want the MAIN visible face
                        # Accept faces that are 1% - 80% of frame area
                        area_ratio = face_area / frame_area if frame_area > 0 else 0
                        if area_ratio < 0.01 or area_ratio > 0.80:
                            continue
                            
                        face_center_x = face_x + face_w // 2
                        face_center_y = face_y + face_h // 2
                        confidence = face.get("confidence", 0.5)
                        
                        face_positions.append({
                            "x": face_center_x,
                            "y": face_center_y,
                            "weight": confidence * face_area,
                        })
            
            if face_positions:
                # Calculate weighted average face position
                total_weight = sum(p["weight"] for p in face_positions)
                if total_weight > 0:
                    avg_x = sum(p["x"] * p["weight"] for p in face_positions) / total_weight
                    avg_y = sum(p["y"] * p["weight"] for p in face_positions) / total_weight
                    center_x = int(avg_x)
                    center_y = int(avg_y)
                    logger.info(f"Talking head: using FACE position ({center_x}, {center_y}) from {len(face_positions)} detections")
                else:
                    center_x = source_width // 2
                    center_y = source_height // 2
                    logger.info(f"Talking head: using CENTER position ({center_x}, {center_y}) - no valid face weights")
            else:
                # No faces found - fallback to center
                center_x = source_width // 2
                center_y = source_height // 2
                logger.info(f"Talking head: using CENTER position ({center_x}, {center_y}) - no faces detected")
            
            # Clamp center position to ensure crop stays within frame
            half_width = window_width // 2
            half_height = window_height // 2
            center_x = max(half_width, min(center_x, source_width - half_width))
            center_y = max(half_height, min(center_y, source_height - half_height))
            
            # Create a single keyframe at face position
            keyframes = [
                CropKeyframe(
                    timestamp_ms=segment.start_time_ms,
                    center_x=center_x,
                    center_y=center_y,
                )
            ]
            
            return CropTimeline(
                window_width=window_width,
                window_height=window_height,
                source_width=source_width,
                source_height=source_height,
                keyframes=keyframes,
            )
        
        # For screen_share: collect face positions for the corner webcam region
        # (The actual bbox comes from corner_facecam_bbox in the renderer)
        frame_area = source_width * source_height
        
        # Collect raw face positions using WEIGHTED AVERAGE
        raw_positions: list[tuple[int, int, int]] = []  # (timestamp_ms, center_x, center_y)
        
        for frame in detection_frames:
            timestamp_ms = int(frame.get("timestamp_sec", 0) * 1000)
            faces = frame.get("detections", [])
            
            if faces:
                # Calculate weighted average position using confidence × area
                total_weight = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                
                for face in faces:
                    bbox = face.get("bbox", {})
                    confidence = face.get("confidence", 0.5)
                    face_w = bbox.get("width", 0)
                    face_h = bbox.get("height", 0)
                    face_x = bbox.get("x", 0)
                    face_y = bbox.get("y", 0)
                    face_area = face_w * face_h
                    
                    # For screen_share: ONLY use faces in the CAMERA region (bottom 70%)
                    face_center_y_ratio = (face_y + face_h / 2) / source_height
                    if face_center_y_ratio < 0.30:
                        continue  # Skip faces in screen region

                    # Accept small webcam faces (0.5% - 25% of frame)
                    area_ratio = face_area / frame_area if frame_area > 0 else 0
                    if area_ratio < 0.005 or area_ratio > 0.25:
                        continue

                    weight = confidence * face_area
                    center_x = face_x + face_w / 2
                    center_y = face_y + face_h / 2
                    
                    weighted_x += center_x * weight
                    weighted_y += center_y * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_x = int(weighted_x / total_weight)
                    final_y = int(weighted_y / total_weight)
                    raw_positions.append((timestamp_ms, final_x, final_y))
                else:
                    # No valid faces - use bottom-right fallback for webcam
                    if raw_positions:
                        _, last_x, last_y = raw_positions[-1]
                        raw_positions.append((timestamp_ms, last_x, last_y))
                    else:
                        center_x = int(source_width * 0.85)
                        center_y = int(source_height * 0.85)
                        raw_positions.append((timestamp_ms, center_x, center_y))
            else:
                # No face detected - use last known or bottom-right fallback
                if raw_positions:
                    _, last_x, last_y = raw_positions[-1]
                    raw_positions.append((timestamp_ms, last_x, last_y))
                else:
                    center_x = int(source_width * 0.85)
                    center_y = int(source_height * 0.85)
                    raw_positions.append((timestamp_ms, center_x, center_y))
        
        if not raw_positions:
            # Fallback to bottom-right for webcam
            center_x = int(source_width * 0.85)
            center_y = int(source_height * 0.85)
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
        Filter outlier positions using statistical analysis.
        
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
        window_size: int = 5,
    ) -> list[tuple[int, int, int]]:
        """
        Apply moving average smoothing to reduce camera jitter.
        Uses weighted average with higher weight for center frames.
        
        Args:
            positions: List of (timestamp_ms, center_x, center_y)
            window_size: Number of frames to average (should be odd)
            
        Returns:
            Smoothed positions
        """
        if len(positions) <= window_size:
            return positions
        
        smoothed = []
        half_window = window_size // 2
        
        # Generate gaussian-like weights for smoother blending
        weights = []
        for j in range(-half_window, half_window + 1):
            # Simple triangular weighting (center has highest weight)
            weight = half_window + 1 - abs(j)
            weights.append(weight)
        
        for i, (ts, _, _) in enumerate(positions):
            # Get window of positions
            start_idx = max(0, i - half_window)
            end_idx = min(len(positions), i + half_window + 1)
            
            # Calculate weighted average position
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            
            for j, idx in enumerate(range(start_idx, end_idx)):
                # Adjust weight index based on actual window position
                weight_idx = j + (half_window - (i - start_idx))
                if 0 <= weight_idx < len(weights):
                    w = weights[weight_idx]
                else:
                    w = 1
                
                weighted_x += positions[idx][1] * w
                weighted_y += positions[idx][2] * w
                total_weight += w
            
            avg_x = int(weighted_x / total_weight) if total_weight > 0 else positions[i][1]
            avg_y = int(weighted_y / total_weight) if total_weight > 0 else positions[i][2]
            
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

        Takes a SQUARE crop from the CENTER of the frame.
        The screen content is typically centered in stream layouts.
        """
        # Square crop from center of frame
        square_size = min(source_width, source_height)
        center_x = source_width // 2
        center_y = source_height // 2

        logger.info(f"Screen crop: square {square_size}x{square_size} from center")

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

    async def _render_with_layout_transitions(
        self,
        video_path: str,
        segment: ClipPlanSegment,
        layout_segments: list[LayoutSegment],
        detection_frames: list[dict],
        transcript_segments: Optional[list],
        output_path: str,
        source_width: int,
        source_height: int,
        caption_style: Optional[CaptionStyle] = None,
    ) -> RenderResult:
        """
        Render a clip with multiple layout segments and stitch them together.

        This method handles dynamic layout switching by:
        1. Rendering each layout segment separately with its appropriate layout
        2. Concatenating all segments using FFmpeg
        3. Adding captions to the final merged video

        Args:
            video_path: Path to source video
            segment: Original clip plan segment (for metadata)
            layout_segments: List of LayoutSegment with start/end times and layout types
            detection_frames: Face detection frames for crop timeline building
            transcript_segments: Transcript for captions (applied to final output)
            output_path: Final output path
            source_width: Source video width
            source_height: Source video height
            caption_style: Caption styling options

        Returns:
            RenderResult with final output path and metadata
        """
        from pathlib import Path

        temp_dir = Path(output_path).parent / f"segments_{segment.start_time_ms}"
        temp_dir.mkdir(exist_ok=True)
        segment_files: list[Path] = []

        try:
            logger.info(
                f"Rendering {len(layout_segments)} layout segments for clip "
                f"{segment.start_time_ms}ms - {segment.end_time_ms}ms"
            )

            for i, layout_seg in enumerate(layout_segments):
                segment_output = temp_dir / f"segment_{i:03d}.mp4"

                logger.info(
                    f"Rendering layout segment {i + 1}/{len(layout_segments)}: "
                    f"{layout_seg.start_ms}ms - {layout_seg.end_ms}ms "
                    f"(duration: {(layout_seg.end_ms - layout_seg.start_ms) / 1000:.1f}s) "
                    f"layout={layout_seg.layout_type}, corner_facecam_bbox={layout_seg.corner_facecam_bbox}"
                )

                # Filter detection frames for this segment's time range
                seg_detection_frames = self._filter_detection_frames_for_segment(
                    detection_frames, layout_seg.start_ms, layout_seg.end_ms
                )
                logger.info(
                    f"Segment {i + 1} detection frames: {len(seg_detection_frames)} frames "
                    f"(from {len(detection_frames)} total, time range {layout_seg.start_ms}ms-{layout_seg.end_ms}ms)"
                )

                # Create a temporary segment for crop timeline building
                temp_segment = ClipPlanSegment(
                    start_time_ms=layout_seg.start_ms,
                    end_time_ms=layout_seg.end_ms,
                    layout_type=layout_seg.layout_type,
                    virality_score=segment.virality_score,
                    summary=segment.summary,
                    tags=segment.tags,
                )

                # Build crop timeline for this segment
                # Pass corner_facecam_bbox to filter out wrong faces
                crop_timeline = self._build_crop_timeline(
                    seg_detection_frames,
                    temp_segment,
                    source_width,
                    source_height,
                    corner_facecam_bbox=layout_seg.corner_facecam_bbox,
                )

                # For screen_share layout, build screen timeline
                screen_timeline = None
                if layout_seg.layout_type == "screen_share":
                    screen_timeline = self._build_screen_timeline(
                        temp_segment,
                        source_width,
                        source_height,
                        seg_detection_frames,
                    )

                # Filter transcript segments for this layout segment's time range
                seg_transcript = None
                seg_caption_style = None
                if transcript_segments:
                    seg_transcript = [
                        ts for ts in transcript_segments
                        if ts.start_time_ms < layout_seg.end_ms and ts.end_time_ms > layout_seg.start_ms
                    ]
                    if seg_transcript and caption_style:
                        # Adjust caption position based on layout type:
                        # - talking_head: bottom (natural look, face is centered/upper)
                        # - screen_share: center (between screen top and face bottom)
                        seg_caption_style = CaptionStyle()
                        seg_caption_style.font_name = caption_style.font_name
                        seg_caption_style.font_size = caption_style.font_size
                        seg_caption_style.primary_color = caption_style.primary_color
                        seg_caption_style.highlight_color = caption_style.highlight_color
                        seg_caption_style.outline_color = caption_style.outline_color
                        seg_caption_style.outline_width = caption_style.outline_width
                        seg_caption_style.max_words_per_line = caption_style.max_words_per_line
                        seg_caption_style.word_by_word_highlight = caption_style.word_by_word_highlight
                        seg_caption_style.alignment = caption_style.alignment
                        seg_caption_style.bold = caption_style.bold
                        seg_caption_style.uppercase = caption_style.uppercase
                        
                        # Dynamic position based on layout
                        if layout_seg.layout_type == "talking_head":
                            seg_caption_style.position = "bottom"
                        else:  # screen_share
                            seg_caption_style.position = "center"

                # Build render request for this segment WITH captions per-segment
                # Include corner_facecam_bbox from the layout segment for proper face cropping
                render_request = RenderRequest(
                    video_path=video_path,
                    output_path=str(segment_output),
                    start_time_ms=layout_seg.start_ms,
                    end_time_ms=layout_seg.end_ms,
                    source_width=source_width,
                    source_height=source_height,
                    layout_type=layout_seg.layout_type,
                    primary_timeline=crop_timeline,
                    secondary_timeline=screen_timeline,
                    transcript_segments=seg_transcript if seg_transcript else None,
                    caption_style=seg_caption_style,
                    corner_facecam_bbox=layout_seg.corner_facecam_bbox,
                )

                # Render segment
                await self.rendering_service.render_clip(render_request)
                segment_files.append(segment_output)

                segment_size = segment_output.stat().st_size / 1024 / 1024
                segment_duration = (layout_seg.end_ms - layout_seg.start_ms) / 1000
                logger.info(
                    f"Segment {i + 1} rendered: {segment_output.name}, "
                    f"{segment_size:.1f} MB, {segment_duration:.1f}s, "
                    f"boundary: {layout_seg.start_ms}ms-{layout_seg.end_ms}ms"
                )

            # Concatenate all segments (captions already burned into each segment)
            await self._concatenate_segments(segment_files, output_path)

            # Get final file size
            file_size = os.path.getsize(output_path)
            duration_ms = segment.end_time_ms - segment.start_time_ms

            logger.info(
                f"Multi-segment clip rendered: {len(layout_segments)} segments, "
                f"{file_size / 1024 / 1024:.1f} MB"
            )

            return RenderResult(
                output_path=output_path,
                file_size_bytes=file_size,
                duration_ms=duration_ms,
            )

        finally:
            # Cleanup temp segment files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp segments dir: {e}")

    def _filter_detection_frames_for_segment(
        self,
        detection_frames: list[dict],
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        """Filter detection frames that fall within the given time range."""
        filtered = []
        for frame in detection_frames:
            timestamp_ms = int(frame.get("timestamp_sec", 0) * 1000)
            if start_ms <= timestamp_ms <= end_ms:
                filtered.append(frame)
        return filtered

    async def _concatenate_segments(
        self,
        segment_files: list[Path],
        output_path: str,
    ) -> None:
        """
        Concatenate video segments using FFmpeg concat demuxer.

        Args:
            segment_files: List of paths to segment video files
            output_path: Output path for merged video
        """
        from pathlib import Path

        if not segment_files:
            raise ValueError("No segments to concatenate")

        if len(segment_files) == 1:
            # Single segment - just copy
            shutil.copy(str(segment_files[0]), output_path)
            return

        concat_file = Path(output_path).with_suffix('.concat.txt')

        try:
            # Write concat file
            with open(concat_file, 'w') as f:
                for seg_file in segment_files:
                    # Use absolute path and escape single quotes
                    safe_path = str(seg_file.absolute()).replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            logger.debug(f"Concatenating {len(segment_files)} segments with frame-accurate boundaries")

            # Frame-accurate concatenation to prevent gaps/overlaps
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-accurate_seek',  # Ensure frame accuracy
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', self.settings.ffmpeg_preset,
                '-crf', str(self.settings.ffmpeg_crf),
                '-pix_fmt', 'yuv420p',
                '-avoid_negative_ts', 'make_zero',  # Normalize timestamps
                '-fflags', '+genpts',  # Generate presentation timestamps
                output_path,
            ]

            # Use run_in_executor for Windows compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True)
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode()[-1000:] if result.stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg concat failed: {error_msg}")

            # Validate concatenated output
            actual_size = os.path.getsize(output_path) / 1024 / 1024
            
            logger.info(
                f"Segments concatenated successfully: {len(segment_files)} segments, "
                f"{actual_size:.1f} MB"
            )

        finally:
            # Cleanup concat file
            try:
                concat_file.unlink(missing_ok=True)
            except Exception:
                pass

    async def _add_captions_to_merged_video(
        self,
        input_path: str,
        output_path: str,
        transcript_segments: list,
        clip_start_ms: int,
        clip_end_ms: int,
        caption_style: Optional[CaptionStyle] = None,
    ) -> None:
        """
        Add captions to a merged video file.

        Args:
            input_path: Path to input video (merged segments)
            output_path: Path to final output with captions
            transcript_segments: Transcript segments for caption generation
            clip_start_ms: Original clip start time (for time offset calculation)
            clip_end_ms: Original clip end time
            caption_style: Caption styling options
        """
        from app.services.caption_generator import CaptionGeneratorService
        from pathlib import Path

        caption_generator = CaptionGeneratorService()

        # Generate caption file
        caption_filename = f"merged-{clip_start_ms}-{clip_end_ms}.ass"
        caption_path = str(Path(output_path).parent / caption_filename)

        try:
            caption_path = await caption_generator.generate_captions(
                transcript_segments=transcript_segments,
                clip_start_ms=clip_start_ms,
                clip_end_ms=clip_end_ms,
                output_path=caption_path,
                caption_style=caption_style,
            )

            # Burn captions into video
            # FFmpeg filter paths need special escaping for cross-platform support
            escaped_caption_path = self._escape_ffmpeg_filter_path(caption_path)

            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', f"ass={escaped_caption_path}",
                '-c:v', 'libx264',
                '-c:a', 'copy',
                '-preset', self.settings.ffmpeg_preset,
                '-crf', str(self.settings.ffmpeg_crf),
                '-pix_fmt', 'yuv420p',
                output_path,
            ]

            # Use run_in_executor for Windows compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True)
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode()[-1000:] if result.stderr else "Unknown error"
                logger.warning(f"Caption burning failed: {error_msg}")
                # Fallback: copy input to output without captions
                shutil.copy(input_path, output_path)

        finally:
            # Cleanup caption file
            try:
                if os.path.isfile(caption_path):
                    os.remove(caption_path)
            except Exception:
                pass

    def _escape_ffmpeg_filter_path(self, path: str) -> str:
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
        # Order matters: escape backslashes first if we add any
        escaped = escaped.replace("'", "'\\''")  # Escape single quotes for shell
        escaped = escaped.replace("[", "\\[")
        escaped = escaped.replace("]", "\\]")
        
        # Wrap in single quotes for the filter expression
        return f"'{escaped}'"

    def _estimate_webcam_bbox_from_position(
        self,
        position: str,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Estimate webcam bounding box from VLM-detected position.
        
        When FaceTracker fails to find a face but VLM detected a webcam overlay,
        we use the VLM position hint to estimate where the webcam should be.
        This creates a reasonable crop region for split-screen rendering.
        
        Args:
            position: VLM-detected position (e.g., "bottom-right", "bottom-left", etc.)
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Tuple (x, y, width, height) for estimated webcam region
        """
        # Assume webcam overlays are typically 20-25% of frame dimensions
        webcam_width = int(frame_width * 0.20)
        webcam_height = int(frame_height * 0.20)
        
        # Calculate position based on VLM detection
        position_lower = (position or "bottom-right").lower()
        
        # Horizontal position
        if "left" in position_lower:
            x = int(frame_width * 0.02)  # 2% margin from left
        elif "center" in position_lower:
            x = (frame_width - webcam_width) // 2
        else:  # default to right
            x = frame_width - webcam_width - int(frame_width * 0.02)
        
        # Vertical position
        if "top" in position_lower:
            y = int(frame_height * 0.02)  # 2% margin from top
        elif "middle" in position_lower:
            y = (frame_height - webcam_height) // 2
        else:  # default to bottom
            y = frame_height - webcam_height - int(frame_height * 0.02)
        
        logger.info(f"Estimated webcam bbox from VLM position '{position}': x={x}, y={y}, w={webcam_width}, h={webcam_height}")
        
        return (x, y, webcam_width, webcam_height)

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
        output: Optional[dict] = None,
    ) -> None:
        """Update job progress via callback and webhook."""
        # Call in-memory progress callback
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

        # Send webhook if callback URL is configured
        if self._current_callback_url:
            self._send_webhook(
                job_id=job_id,
                status=status,
                progress=progress,
                step=step,
                clips_completed=clips_completed,
                total_clips=total_clips,
                error=error,
                output=output,
            )

    def _send_webhook(
        self,
        job_id: str,
        status: JobStatus,
        progress: float,
        step: str,
        clips_completed: int = 0,
        total_clips: int = 0,
        error: Optional[str] = None,
        output: Optional[dict] = None,
    ) -> None:
        """Send webhook notification for job status update."""
        # Map JobStatus to webhook event names
        event_map = {
            JobStatus.PENDING: "job.started",
            JobStatus.DOWNLOADING: "job.progress",
            JobStatus.TRANSCRIBING: "job.progress",
            JobStatus.PLANNING: "job.progress",
            JobStatus.DETECTING: "job.progress",
            JobStatus.RENDERING: "job.progress",
            JobStatus.UPLOADING: "job.progress",
            JobStatus.COMPLETED: "job.completed",
            JobStatus.FAILED: "job.failed",
        }
        event = event_map.get(status, "job.progress")

        # Map JobStatus to API-compatible status values
        status_map = {
            JobStatus.PENDING: "queued",
            JobStatus.DOWNLOADING: "running",
            JobStatus.TRANSCRIBING: "running",
            JobStatus.PLANNING: "running",
            JobStatus.DETECTING: "running",
            JobStatus.RENDERING: "running",
            JobStatus.UPLOADING: "running",
            JobStatus.COMPLETED: "succeeded",
            JobStatus.FAILED: "failed",
        }
        api_status = status_map.get(status, "running")

        # For progress events, throttle to avoid overwhelming the receiver
        is_terminal = status in (JobStatus.COMPLETED, JobStatus.FAILED)
        if not is_terminal and not self.webhook_service.should_send_progress(job_id):
            return

        # Build and send the webhook payload
        payload = self.webhook_service.build_payload(
            event=event,
            job_id=job_id,
            status=api_status,
            progress_percent=progress,
            current_step=step,
            external_job_id=self._current_external_job_id,
            owner_user_id=self._current_owner_user_id,
            clips_completed=clips_completed,
            total_clips=total_clips,
            error=error,
            output=output,
        )

        logger.info(f"Sending webhook: {event} to {self._current_callback_url}")

        # Fire and forget for progress updates, await for terminal events
        try:
            asyncio.create_task(
                self.webhook_service.send(self._current_callback_url, payload)
            )
        except RuntimeError as e:
            # No event loop running - log and skip
            logger.warning(f"Could not send webhook (no event loop): {e}")

