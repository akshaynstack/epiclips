"""
AI Clipping API Router - Endpoints for the full AI clipping workflow.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, HttpUrl, model_validator

from app.auth import verify_api_key

# Duration range type for multi-select
DurationRangeType = Literal["short", "medium", "long"]

# Layout type for clip rendering
# NOTE: Layout is now always automatic - the AI detects and switches layouts dynamically.
# The "split_screen" and "talking_head" options are deprecated and will be ignored.
# Kept for backward compatibility with older API clients.
LayoutTypeInput = Literal["auto", "split_screen", "talking_head"]

# Duration range configuration
DURATION_RANGE_CONFIG = {
    "short": {"min": 15, "max": 30, "label": "15-30 seconds"},
    "medium": {"min": 30, "max": 60, "label": "30-60 seconds"},
    "long": {"min": 60, "max": 120, "label": "1-2 minutes"},
}

from app.config import (
    CaptionStyle,
    CaptionPreset,
    get_caption_preset,
    get_available_presets,
    LayoutType,
    get_layout_preset,
    get_available_layouts,
    get_settings,
)
from app.services.ai_clipping_pipeline import (
    AIClippingPipeline,
    ClippingJobProgress,
    ClippingJobRequest,
    ClippingJobResult,
    JobStatus,
)
from app.services.detection_pipeline import DetectionPipeline
from app.services.video_downloader import VideoDownloaderService, VideoDownloadError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-clipping", tags=["AI Clipping"])


# ============================================================================
# Request/Response Models
# ============================================================================


class CaptionStyleInput(BaseModel):
    """Caption style configuration."""
    
    font_family: str = "Arial Black"
    font_size: int = 72
    primary_color: str = "FFFFFF"  # White
    highlight_color: str = "00FF00"  # Green
    outline_color: str = "000000"  # Black
    outline_width: int = 4
    position: str = "bottom"  # top, center, bottom
    alignment: str = "center"  # left, center, right
    words_per_group: int = 3


class ClipJobSubmitRequest(BaseModel):
    """Request to submit a new AI clipping job.

    Supports multiple video sources:
    - YouTube URL: https://youtube.com/watch?v=...
    - S3 URL: s3://bucket/key or https://bucket.s3.region.amazonaws.com/key
    - S3 Key: path/to/video.mp4 (uses configured bucket)
    - Direct URL: https://example.com/video.mp4
    """

    video_url: Optional[str] = Field(None, description="Video URL (YouTube, S3, or direct)")
    s3_key: Optional[str] = Field(None, description="S3 key (if using configured bucket)")
    max_clips: Optional[int] = Field(
        None, 
        ge=1, 
        le=50, 
        description="Maximum number of clips to generate. Optional when auto_clip_count=true (uses config max_clips_absolute). Required when auto_clip_count=false."
    )
    auto_clip_count: bool = Field(
        True,
        description="If true, automatically determine optimal clip count based on video duration (1 clip per 5 min by default). max_clips becomes optional upper limit."
    )
    min_clip_duration_seconds: Optional[int] = Field(None, ge=5, le=120, description="Minimum clip duration (legacy, use duration_ranges)")
    max_clip_duration_seconds: Optional[int] = Field(None, ge=15, le=300, description="Maximum clip duration (legacy, use duration_ranges)")
    duration_ranges: Optional[list[DurationRangeType]] = Field(
        None,
        description="Selected duration ranges: 'short' (15-30s), 'medium' (30-60s), 'long' (60-120s). Overrides min/max when provided."
    )
    target_platform: str = Field("tiktok", description="Target platform (tiktok, youtube_shorts, instagram_reels)")
    include_captions: bool = Field(True, description="Whether to include viral-style captions")
    caption_preset: Optional[str] = Field(
        None,
        description="Caption preset ID: 'viral_gold', 'clean_white', 'neon_pop', 'bold_boxed', 'gradient_glow'. Takes precedence over caption_style."
    )
    caption_style: Optional[CaptionStyleInput] = Field(None, description="Custom caption styling (used if caption_preset not provided)")
    layout_type: Optional[LayoutTypeInput] = Field(
        "auto",
        deprecated=True,
        description="DEPRECATED: Layout detection is now always automatic. The AI analyzes each frame and dynamically switches between layouts mid-clip for optimal framing. This field is ignored."
    )
    callback_url: Optional[str] = Field(None, description="Webhook URL for progress updates")
    
    # Time range selection for processing specific portion of video
    start_time_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="Start time in seconds from video beginning. If not provided, starts from 0."
    )
    end_time_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="End time in seconds from video beginning. If not provided, processes until end of video."
    )

    # For API integration - job tracking
    external_job_id: Optional[str] = Field(None, description="External job ID from calling service")
    owner_user_id: Optional[str] = Field(None, description="User ID for tracking")

    @model_validator(mode='after')
    def validate_max_clips_requirement(self) -> 'ClipJobSubmitRequest':
        """Ensure max_clips is provided when auto_clip_count is disabled."""
        if not self.auto_clip_count and self.max_clips is None:
            raise ValueError("max_clips is required when auto_clip_count is false")
        return self
    
    @model_validator(mode='after')
    def validate_time_range(self) -> 'ClipJobSubmitRequest':
        """Ensure time range is valid if provided."""
        if self.start_time_seconds is not None and self.end_time_seconds is not None:
            if self.end_time_seconds <= self.start_time_seconds:
                raise ValueError(
                    f"end_time_seconds ({self.end_time_seconds}) must be greater than "
                    f"start_time_seconds ({self.start_time_seconds})"
                )
            # Check minimum range (at least 60 seconds for meaningful clip generation)
            range_duration = self.end_time_seconds - self.start_time_seconds
            if range_duration < 30:
                raise ValueError(
                    f"Selected time range ({range_duration:.1f}s) is too short. "
                    f"Minimum is 30 seconds for clip generation."
                )
        return self


class ClipArtifactResponse(BaseModel):
    """Response model for a single clip artifact."""
    
    clip_index: int
    s3_url: str
    duration_ms: int
    start_time_ms: int
    end_time_ms: int
    virality_score: float
    layout_type: str
    summary: Optional[str] = None
    tags: list[str] = []


class JobOutputResponse(BaseModel):
    """Response model for job output."""

    job_id: str
    source_video_url: str
    source_video_title: str
    source_video_duration_seconds: float  # Source video duration for credit calculation
    total_clips: int
    clips: list[ClipArtifactResponse]
    transcript_url: Optional[str] = None
    plan_url: Optional[str] = None
    processing_time_seconds: float
    created_at: str


class ClipJobSubmitResponse(BaseModel):
    """Response after submitting a clipping job."""
    
    job_id: str
    status: str
    message: str
    estimated_processing_minutes: int


class ClipJobStatusResponse(BaseModel):
    """Response for job status query."""
    
    job_id: str
    status: str
    progress_percent: float
    current_step: str
    clips_completed: int
    total_clips: int
    error: Optional[str] = None
    output: Optional[JobOutputResponse] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    services: dict[str, str]
    timestamp: str


class VideoMetadataRequest(BaseModel):
    """Request to get video metadata for cost estimation."""

    video_url: Optional[str] = Field(None, description="Video URL (YouTube, S3, or direct)")
    s3_key: Optional[str] = Field(None, description="S3 key (if using configured bucket)")


class VideoMetadataResponse(BaseModel):
    """Response with video metadata for cost estimation."""

    title: str
    duration_seconds: float
    duration_minutes: float
    width: int
    height: int
    source_type: str
    estimated_credits: int = Field(description="Estimated credits at 10 credits/minute")


# ============================================================================
# In-Memory Job Storage (for demo; use Redis in production)
# ============================================================================

# Store job progress and results
_job_store: dict[str, ClippingJobProgress] = {}
_job_results: dict[str, ClippingJobResult] = {}

# Semaphore for limiting concurrent jobs
_job_semaphore: Optional[asyncio.Semaphore] = None


def get_job_semaphore() -> asyncio.Semaphore:
    """Get or create job semaphore."""
    global _job_semaphore
    settings = get_settings()
    if _job_semaphore is None:
        _job_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _job_semaphore


# ============================================================================
# Dependencies
# ============================================================================


async def get_detection_pipeline(request: Request) -> DetectionPipeline:
    """Get the detection pipeline from app state (initialized at startup)."""
    if not hasattr(request.app.state, "detection_pipeline"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection pipeline not initialized",
        )
    return request.app.state.detection_pipeline


def progress_callback(progress: ClippingJobProgress) -> None:
    """Callback to store job progress."""
    _job_store[progress.job_id] = progress
    logger.debug(f"Job {progress.job_id}: {progress.status} - {progress.progress_percent:.0f}%")


# ============================================================================
# Endpoints
# ============================================================================


class CaptionPresetResponse(BaseModel):
    """Response model for a caption preset."""

    id: str
    name: str
    description: str
    preview_colors: dict[str, str]


@router.get("/caption-presets", response_model=list[CaptionPresetResponse])
async def list_caption_presets() -> list[CaptionPresetResponse]:
    """
    List all available caption presets.

    Returns preset IDs, names, descriptions, and preview colors for UI rendering.
    """
    presets = get_available_presets()
    return [
        CaptionPresetResponse(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            preview_colors=p["preview_colors"],
        )
        for p in presets
    ]


class LayoutPresetResponse(BaseModel):
    """Response model for a layout preset."""

    id: str
    name: str
    description: str
    icon: str
    preview_layout: dict[str, str]


@router.get("/layout-presets", response_model=list[LayoutPresetResponse], deprecated=True)
async def list_layout_presets() -> list[LayoutPresetResponse]:
    """
    DEPRECATED: Layout detection is now always automatic.

    The AI analyzes each frame and dynamically switches between talking head
    and screen share layouts mid-clip for optimal framing. Users no longer
    need to select a layout.

    This endpoint is maintained for backward compatibility but only returns
    the automatic layout option.
    """
    # Return only the auto preset - layout selection is now automatic
    return [
        LayoutPresetResponse(
            id="auto",
            name="Automatic",
            description="AI automatically detects and switches between layouts for optimal framing",
            icon="sparkles",
            preview_layout={"type": "dynamic", "description": "AI-powered scene detection"},
        )
    ]


@router.get("/health", response_model=HealthResponse)
async def health_check(
    detection_pipeline: DetectionPipeline = Depends(get_detection_pipeline),
) -> HealthResponse:
    """
    Check health of all AI clipping services.
    """
    services = {
        "detection_pipeline": "healthy" if detection_pipeline else "unavailable",
        "video_downloader": "healthy",  # Will fail at runtime if yt-dlp missing
        "transcription": "healthy",  # Will fail at runtime if API keys missing
        "intelligence_planner": "healthy",
        "rendering": "healthy",  # Will fail at runtime if ffmpeg missing
        "s3_upload": "healthy",
    }
    
    return HealthResponse(
        status="healthy",
        services=services,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post("/metadata", response_model=VideoMetadataResponse)
async def get_video_metadata(
    request: VideoMetadataRequest,
    _: None = Depends(verify_api_key),
) -> VideoMetadataResponse:
    """
    Get video metadata for cost estimation without starting a job.

    This lightweight endpoint extracts video duration and metadata to allow
    calculating the credit cost before submitting a job.

    Pricing: 10 credits per minute of source video (minimum 10 credits).

    Args:
        request: Video URL or S3 key

    Returns:
        Video metadata including duration and estimated credit cost
    """
    # Validate that either video_url or s3_key is provided
    if not request.video_url and not request.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either video_url or s3_key must be provided",
        )

    video_source = request.video_url or request.s3_key
    downloader = VideoDownloaderService()

    try:
        source_type = downloader.detect_source_type(video_source)

        if source_type == "youtube":
            # For YouTube, use yt-dlp to get metadata without downloading
            metadata = await downloader._get_video_info(video_source)
        elif source_type == "s3":
            # For S3, we'd need to download to get metadata - return error for now
            # In production, could use HEAD request + ffprobe on range request
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="S3 video metadata requires download. Please provide a YouTube URL for cost estimation, or submit the job directly.",
            )
        else:
            # For direct URLs, similar limitation
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direct URL metadata requires download. Please provide a YouTube URL for cost estimation, or submit the job directly.",
            )

        # Calculate estimated credits: 10 credits per minute, minimum 10
        duration_minutes = metadata.duration_seconds / 60
        estimated_credits = max(10, int(duration_minutes * 10 + 0.5))  # Round to nearest

        return VideoMetadataResponse(
            title=metadata.title,
            duration_seconds=metadata.duration_seconds,
            duration_minutes=round(duration_minutes, 2),
            width=metadata.width,
            height=metadata.height,
            source_type=source_type,
            estimated_credits=estimated_credits,
        )

    except VideoDownloadError as e:
        logger.error(f"Failed to get video metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get video metadata: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error getting video metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve video metadata",
        )


@router.post("/jobs", response_model=ClipJobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_clipping_job(
    request: ClipJobSubmitRequest,
    background_tasks: BackgroundTasks,
    detection_pipeline: DetectionPipeline = Depends(get_detection_pipeline),
    _: None = Depends(verify_api_key),  # Require API key authentication
) -> ClipJobSubmitResponse:
    """
    Submit a new AI clipping job.
    
    The job will be processed asynchronously. Use GET /jobs/{job_id} to check status.
    
    Accepts videos from:
    - YouTube URLs
    - S3 URLs or keys
    - Direct video URLs
    
    Args:
        request: Job configuration including video URL/S3 key and options
        
    Returns:
        Job ID and estimated processing time
    """
    settings = get_settings()
    
    # Validate that either video_url or s3_key is provided
    if not request.video_url and not request.s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either video_url or s3_key must be provided",
        )
    
    # Determine the video source
    video_source = request.video_url or request.s3_key
    
    # Validate platform
    valid_platforms = ["tiktok", "youtube_shorts", "instagram_reels"]
    if request.target_platform not in valid_platforms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid platform. Must be one of: {valid_platforms}",
        )
    
    # Resolve caption style: preset takes precedence over custom style
    caption_style: Optional[CaptionStyle] = None
    if request.caption_preset:
        # Use preset if specified
        try:
            caption_style = get_caption_preset(request.caption_preset)
            logger.info(f"Using caption preset: {request.caption_preset}")
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
    elif request.caption_style:
        # Fall back to custom style
        caption_style = CaptionStyle()
        caption_style.font_name = request.caption_style.font_family
        caption_style.font_size = request.caption_style.font_size
        caption_style.primary_color = f"#{request.caption_style.primary_color}" if not request.caption_style.primary_color.startswith("#") else request.caption_style.primary_color
        caption_style.highlight_color = f"#{request.caption_style.highlight_color}" if not request.caption_style.highlight_color.startswith("#") else request.caption_style.highlight_color
        caption_style.outline_color = f"#{request.caption_style.outline_color}" if not request.caption_style.outline_color.startswith("#") else request.caption_style.outline_color
        caption_style.outline_width = request.caption_style.outline_width
        caption_style.position = request.caption_style.position
        caption_style.alignment = request.caption_style.alignment
        caption_style.max_words_per_line = request.caption_style.words_per_group
    
    # Calculate min/max duration from duration_ranges if provided
    min_duration = request.min_clip_duration_seconds
    max_duration = request.max_clip_duration_seconds
    
    if request.duration_ranges and len(request.duration_ranges) > 0:
        # Calculate the overall min and max from selected ranges
        calc_min = float('inf')
        calc_max = 0
        for range_name in request.duration_ranges:
            if range_name in DURATION_RANGE_CONFIG:
                config = DURATION_RANGE_CONFIG[range_name]
                calc_min = min(calc_min, config["min"])
                calc_max = max(calc_max, config["max"])
        
        if calc_min != float('inf') and calc_max > 0:
            min_duration = int(calc_min)
            max_duration = int(calc_max)
            logger.info(f"Duration ranges {request.duration_ranges} resolved to {min_duration}-{max_duration}s")
    
    # Use defaults if still None
    if min_duration is None:
        min_duration = 15
    if max_duration is None:
        max_duration = 90
    
    # Log if deprecated layout_type was requested (for migration tracking)
    if request.layout_type and request.layout_type != "auto":
        logger.info(f"Ignoring deprecated layout_type '{request.layout_type}' - using automatic layout detection")

    # Create job request - always use "auto" for dynamic layout detection
    job_request = ClippingJobRequest(
        video_url=video_source,
        job_id=request.external_job_id,  # Use external ID if provided
        external_job_id=request.external_job_id,  # Track for webhooks
        owner_user_id=request.owner_user_id,  # Pass user ID for S3 key scoping
        max_clips=request.max_clips,
        auto_clip_count=request.auto_clip_count,  # Pass auto-scaling flag
        min_clip_duration_seconds=min_duration,
        max_clip_duration_seconds=max_duration,
        duration_ranges=request.duration_ranges,
        target_platform=request.target_platform,
        include_captions=request.include_captions,
        caption_style=caption_style,
        layout_type="auto",  # Always use automatic layout detection
        callback_url=request.callback_url,
        start_time_seconds=request.start_time_seconds,  # Time range selection
        end_time_seconds=request.end_time_seconds,  # Time range selection
    )
    
    # Log time range if specified
    if request.start_time_seconds is not None or request.end_time_seconds is not None:
        logger.info(
            f"Time range selection: start={request.start_time_seconds}s, end={request.end_time_seconds}s"
        )
    
    # Initialize progress
    _job_store[job_request.job_id] = ClippingJobProgress(
        job_id=job_request.job_id,
        status=JobStatus.PENDING,
        progress_percent=0,
        current_step="Queued for processing",
    )
    
    # Schedule background processing
    background_tasks.add_task(
        _process_job_background,
        job_request,
        detection_pipeline,
    )
    
    # Estimate processing time (rough heuristic)
    # Use max_clips if provided, otherwise assume default of 5 for estimate
    estimated_clip_count = request.max_clips if request.max_clips is not None else 5
    estimated_minutes = 3 + (estimated_clip_count * 2)
    
    logger.info(f"Job {job_request.job_id} submitted for video: {video_source[:100]}...")
    
    return ClipJobSubmitResponse(
        job_id=job_request.job_id,
        status="accepted",
        message="Job queued for processing",
        estimated_processing_minutes=estimated_minutes,
    )


@router.get("/jobs/{job_id}", response_model=ClipJobStatusResponse)
async def get_job_status(job_id: str) -> ClipJobStatusResponse:
    """
    Get the status of a clipping job.
    
    Args:
        job_id: The job ID returned from POST /jobs
        
    Returns:
        Current job status, progress, and output if completed
    """
    # Check for completed result
    if job_id in _job_results:
        result = _job_results[job_id]
        
        output_response = None
        if result.output:
            output_response = JobOutputResponse(
                job_id=result.output.job_id,
                source_video_url=result.output.source_video_url,
                source_video_title=result.output.source_video_title,
                source_video_duration_seconds=result.output.source_video_duration_seconds,
                total_clips=result.output.total_clips,
                clips=[
                    ClipArtifactResponse(
                        clip_index=c.clip_index,
                        s3_url=c.s3_url,
                        duration_ms=c.duration_ms,
                        start_time_ms=c.start_time_ms,
                        end_time_ms=c.end_time_ms,
                        virality_score=c.virality_score,
                        layout_type=c.layout_type,
                        summary=c.summary,
                        tags=c.tags or [],
                    )
                    for c in result.output.clips
                ],
                transcript_url=result.output.transcript_url,
                plan_url=result.output.plan_url,
                processing_time_seconds=result.output.processing_time_seconds,
                created_at=result.output.created_at,
            )
        
        return ClipJobStatusResponse(
            job_id=job_id,
            status=result.status.value,
            progress_percent=100 if result.status == JobStatus.COMPLETED else 0,
            current_step="Completed" if result.status == JobStatus.COMPLETED else "Failed",
            clips_completed=result.output.total_clips if result.output else 0,
            total_clips=result.output.total_clips if result.output else 0,
            error=result.error,
            output=output_response,
        )
    
    # Check for in-progress job
    if job_id in _job_store:
        progress = _job_store[job_id]
        return ClipJobStatusResponse(
            job_id=job_id,
            status=progress.status.value,
            progress_percent=progress.progress_percent,
            current_step=progress.current_step,
            clips_completed=progress.clips_completed,
            total_clips=progress.total_clips,
            error=progress.error,
            output=None,
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Job not found: {job_id}",
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str,
    _: None = Depends(verify_api_key),  # Require API key authentication
) -> None:
    """
    Cancel a pending or running job.
    
    Note: This only removes the job from tracking. Active processing may continue
    until the next checkpoint.
    """
    if job_id in _job_store:
        _job_store[job_id] = ClippingJobProgress(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress_percent=0,
            current_step="Cancelled by user",
            error="Job cancelled",
        )
        return
    
    if job_id not in _job_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )


@router.get("/jobs", response_model=list[ClipJobStatusResponse])
async def list_jobs(
    status_filter: Optional[str] = None,
    limit: int = 20,
) -> list[ClipJobStatusResponse]:
    """
    List recent clipping jobs.
    
    Args:
        status_filter: Filter by status (pending, processing, completed, failed)
        limit: Maximum number of jobs to return
        
    Returns:
        List of job statuses
    """
    all_jobs = []
    
    # Add completed jobs
    for job_id, result in _job_results.items():
        if status_filter and result.status.value != status_filter:
            continue
        
        all_jobs.append(ClipJobStatusResponse(
            job_id=job_id,
            status=result.status.value,
            progress_percent=100 if result.status == JobStatus.COMPLETED else 0,
            current_step="Completed" if result.status == JobStatus.COMPLETED else "Failed",
            clips_completed=result.output.total_clips if result.output else 0,
            total_clips=result.output.total_clips if result.output else 0,
            error=result.error,
            output=None,  # Don't include full output in list
        ))
    
    # Add in-progress jobs
    for job_id, progress in _job_store.items():
        if job_id in _job_results:
            continue  # Already added
        if status_filter and progress.status.value != status_filter:
            continue
        
        all_jobs.append(ClipJobStatusResponse(
            job_id=job_id,
            status=progress.status.value,
            progress_percent=progress.progress_percent,
            current_step=progress.current_step,
            clips_completed=progress.clips_completed,
            total_clips=progress.total_clips,
            error=progress.error,
            output=None,
        ))
    
    return all_jobs[:limit]


# ============================================================================
# Background Processing
# ============================================================================


async def _process_job_background(
    request: ClippingJobRequest,
    detection_pipeline: DetectionPipeline,
) -> None:
    """
    Process a clipping job in the background with concurrency control.
    """
    semaphore = get_job_semaphore()
    
    async with semaphore:
        try:
            pipeline = AIClippingPipeline(
                detection_pipeline=detection_pipeline,
                progress_callback=progress_callback,
            )
            
            result = await pipeline.process_video(request)
            
            # Store result
            _job_results[request.job_id] = result
            
            # Clean up progress store
            if request.job_id in _job_store:
                del _job_store[request.job_id]
                
        except Exception as e:
            logger.exception(f"Background job failed: {e}")
            
            # Store failure
            _job_results[request.job_id] = ClippingJobResult(
                job_id=request.job_id,
                status=JobStatus.FAILED,
                error=str(e),
            )

