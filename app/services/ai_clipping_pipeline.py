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

        try:
            os.makedirs(work_dir, exist_ok=True)
            logger.info(f"Starting AI clipping job: {job_id}")
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
            
            # Step 4: Run detection for smart cropping
            self._update_progress(job_id, JobStatus.DETECTING, 45, "Analyzing video for smart cropping...")
            detection_results = await self._run_detection_for_clips(
                video_path=download_result.video_path,
                clip_segments=clip_plan.segments,
                source_width=download_result.metadata.width,
                source_height=download_result.metadata.height,
                layout_type=request.layout_type,  # Pass user's layout choice (auto, split_screen, talking_head)
            )
            logger.info("Detection analysis complete")
            
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
                        
                        # Log embedded facecam detection status
                        has_embedded = layout_analysis.has_embedded_facecam if layout_analysis else False
                        logger.info(
                            f"Clip {i + 1} render config: layout={effective_layout}, "
                            f"has_embedded_facecam={has_embedded}"
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
                        "virality_score": clip.virality_score,
                        "summary": clip.summary,
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

    async def _run_detection_for_clips(
        self,
        video_path: str,
        clip_segments: list[ClipPlanSegment],
        source_width: int,
        source_height: int,
        layout_type: str = "auto",
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

        Note: Detection is run for ALL layout types to enable smart cropping:
        - talking_head: Uses face detection for dynamic panning
        - screen_share: Uses face detection for the face portion of stack layout

        Layout analysis enables dynamic layout switching within clips by:
        - Detecting layout transitions (talking_head <-> screen_share)
        - Creating layout segments for per-segment rendering
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
                    # AUTO mode: Run SmartLayoutDetector to detect transitions within the clip
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
    ) -> Optional[CropTimeline]:
        """
        Build a crop timeline from detection results for face tracking.

        Build crop timeline for face tracking:
        - For talking_head: Full 9:16 crop window following face
        - For screen_share: Tight crop around face for bottom region
        - Simple fallback to center-bottom when no faces detected
        
        Args:
            effective_layout: Override layout type from smart detector (takes priority over segment.layout_type)
        """
        # Use effective_layout from smart detector if provided, otherwise fallback to segment's layout
        layout_type = effective_layout if effective_layout else segment.layout_type
        
        # Calculate crop window based on layout type
        if layout_type == "screen_share":
            # For screen_share: tight crop around face for close-up effect
            face_output_height = int(self.settings.target_output_height * self.settings.split_face_ratio)
            face_crop_aspect = self.settings.target_output_width / face_output_height

            # Calculate face size from detections
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
                # Use average face area
                avg_face_area = sum(face_areas) / len(face_areas)
                avg_face_size = int(avg_face_area ** 0.5)

                # Create TIGHT crop: 2x face size
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
            logger.info(f"Talking head crop: {window_width}x{window_height} (9:16 from {source_width}x{source_height})")
        
        frame_area = source_width * source_height
        
        # Collect raw face positions using WEIGHTED AVERAGE
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
                    # For screen_share layouts, use a much lower threshold since
                    # webcam faces in corners are typically small (< 5% of frame)
                    area_ratio = face_area / frame_area if frame_area > 0 else 0
                    if layout_type == "screen_share":
                        # Screen share: accept smaller faces (webcams in corners)
                        # Minimum: 0.1% of frame, Maximum: 20% (if larger, it's talking_head)
                        if area_ratio < 0.001 or area_ratio > 0.20:
                            continue
                    else:
                        # Talking head: faces should be prominent
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
                    
                    # Apply UPPER BIAS for better face framing
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
                        # Fallback position based on layout type
                        if layout_type == "screen_share":
                            # Webcams are typically in corners - default to bottom-right
                            # (most common webcam position)
                            center_x = int(source_width * 0.85)  # Right side
                            center_y = int(source_height * 0.85)  # Bottom
                            logger.warning(f"No faces found for screen_share - using bottom-right fallback")
                        else:
                            center_x = source_width // 2
                            center_y = int(source_height * 0.4)
                        raw_positions.append((timestamp_ms, center_x, center_y))
            else:
                # No face detected - use previous position or corner fallback
                if raw_positions:
                    # Use last known position for continuity
                    _, last_x, last_y = raw_positions[-1]
                    raw_positions.append((timestamp_ms, last_x, last_y))
                else:
                    # Fallback position based on layout type
                    if layout_type == "screen_share":
                        # Webcams are typically in corners - default to bottom-right
                        center_x = int(source_width * 0.85)  # Right side
                        center_y = int(source_height * 0.85)  # Bottom
                        logger.warning(f"No faces detected for screen_share - using bottom-right fallback")
                    else:
                        center_x = source_width // 2
                        center_y = int(source_height * 0.4)  # Upper center for talking head
                    raw_positions.append((timestamp_ms, center_x, center_y))
        
        if not raw_positions:
            # Should not happen given the logic above, but safe fallback
            if layout_type == "screen_share":
                # Webcams are typically in corners - default to bottom-right
                center_x = int(source_width * 0.85)
                center_y = int(source_height * 0.85)
                logger.warning(f"Empty positions for screen_share - using bottom-right fallback")
            else:
                center_x = source_width // 2
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

                logger.debug(
                    f"Rendering segment {i + 1}/{len(layout_segments)}: "
                    f"{layout_seg.start_ms}ms - {layout_seg.end_ms}ms ({layout_seg.layout_type})"
                )

                # Filter detection frames for this segment's time range
                seg_detection_frames = self._filter_detection_frames_for_segment(
                    detection_frames, layout_seg.start_ms, layout_seg.end_ms
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
                crop_timeline = self._build_crop_timeline(
                    seg_detection_frames,
                    temp_segment,
                    source_width,
                    source_height,
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
                )

                # Render segment
                await self.rendering_service.render_clip(render_request)
                segment_files.append(segment_output)

                logger.debug(f"Segment {i + 1} rendered: {segment_output}")

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

            logger.debug(f"Concatenating {len(segment_files)} segments")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-c:a', 'aac',
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
                raise RuntimeError(f"FFmpeg concat failed: {error_msg}")

            logger.debug(f"Segments concatenated: {output_path}")

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

