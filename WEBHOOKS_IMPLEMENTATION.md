# Webhooks Implementation Plan

This document outlines the webhook system for real-time job status updates between the clipping worker and viewcreator-api.

---

## Overview

### Why Webhooks?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         POLLING (Current - Inefficient)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

viewcreator-api                                          clipping-worker
      │                                                        │
      │  POST /ai-clipping/jobs                                │
      │───────────────────────────────────────────────────────▶│
      │◀─────────────────────────────────────────────────────── │ 202 {job_id}
      │                                                        │
      │  GET /jobs/abc123  "is it done yet?"                   │
      │───────────────────────────────────────────────────────▶│
      │◀─────────────────────────────────────────────────────── │ {status: 10%}
      │                                                        │
      │  GET /jobs/abc123  "is it done yet?"                   │
      │───────────────────────────────────────────────────────▶│
      │◀─────────────────────────────────────────────────────── │ {status: 25%}
      │                                                        │
      │         ... repeats every 5-10 seconds ...             │
      │         ... for 10+ minutes per job ...                │
      │         ... 50-100+ wasted requests per job ...        │


┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WEBHOOKS (Proposed - Efficient)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

viewcreator-api                                          clipping-worker
      │                                                        │
      │  POST /ai-clipping/jobs                                │
      │  { callback_url: "https://api.viewcreator.ai/webhooks" │
      │───────────────────────────────────────────────────────▶│
      │                                                        │
      │◀─────────────────────────────────────────────────────── │ 202 {job_id}
      │                                                        │
      │         ... viewcreator-api does other work ...        │
      │         ... no polling needed ...                      │
      │                                                        │
      │  POST /webhooks/clipping (worker calls API)            │
      │◀───────────────────────────────────────────────────────│ {status: 50%}
      │                                                        │
      │  POST /webhooks/clipping (job complete!)               │
      │◀───────────────────────────────────────────────────────│ {status: done, clips: [...]}

      Only 3 HTTP requests vs 50+ with polling!
```

### Benefits

| Aspect | Polling | Webhooks |
|--------|---------|----------|
| HTTP requests per job | 50-100+ | 3-6 |
| Status update latency | 5-10 seconds | Instant |
| Server load | High | Low |
| User experience | Laggy progress | Real-time updates |
| Scalability | Poor | Excellent |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WEBHOOK FLOW ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│                  │         │                  │         │                  │
│  viewcreator-api │◀────────│ clipping-worker  │────────▶│    AWS S3        │
│    (NestJS)      │ webhook │    (FastAPI)     │ upload  │                  │
│                  │         │                  │         │                  │
└────────┬─────────┘         └──────────────────┘         └──────────────────┘
         │                            │
         │                            │
         ▼                            │
┌──────────────────┐                  │
│                  │                  │
│    PostgreSQL    │                  │
│   (Job Status)   │◀─────────────────┘
│                  │        updates job record
└──────────────────┘


Webhook Events:
───────────────
1. job.started      → Job processing has begun
2. job.progress     → Progress update (downloading, transcribing, etc.)
3. job.completed    → Job finished successfully with output
4. job.failed       → Job failed with error details
```

---

## Webhook Payload Specification

### Common Fields

All webhook payloads include these fields:

```typescript
interface WebhookPayload {
  // Event metadata
  event: 'job.started' | 'job.progress' | 'job.completed' | 'job.failed';
  timestamp: string;  // ISO 8601

  // Job identification
  job_id: string;
  external_job_id?: string;  // ID from viewcreator-api
  owner_user_id?: string;

  // Status
  status: 'downloading' | 'transcribing' | 'planning' | 'detecting' | 'rendering' | 'uploading' | 'completed' | 'failed';
  progress_percent: number;  // 0-100
  current_step: string;      // Human-readable description

  // Optional fields
  error?: string;
  output?: JobOutput;
}
```

### Event: job.started

Sent when job processing begins.

```json
{
  "event": "job.started",
  "timestamp": "2025-01-15T10:30:00Z",
  "job_id": "abc-123-def-456",
  "external_job_id": "user-submitted-id",
  "owner_user_id": "user_789",
  "status": "downloading",
  "progress_percent": 0,
  "current_step": "Starting job processing"
}
```

### Event: job.progress

Sent at key pipeline stages.

```json
{
  "event": "job.progress",
  "timestamp": "2025-01-15T10:31:00Z",
  "job_id": "abc-123-def-456",
  "external_job_id": "user-submitted-id",
  "owner_user_id": "user_789",
  "status": "transcribing",
  "progress_percent": 15,
  "current_step": "Transcribing audio with Whisper..."
}
```

```json
{
  "event": "job.progress",
  "timestamp": "2025-01-15T10:33:00Z",
  "job_id": "abc-123-def-456",
  "external_job_id": "user-submitted-id",
  "owner_user_id": "user_789",
  "status": "rendering",
  "progress_percent": 65,
  "current_step": "Rendering clip 3/5...",
  "clips_completed": 2,
  "total_clips": 5
}
```

### Event: job.completed

Sent when job finishes successfully.

```json
{
  "event": "job.completed",
  "timestamp": "2025-01-15T10:38:00Z",
  "job_id": "abc-123-def-456",
  "external_job_id": "user-submitted-id",
  "owner_user_id": "user_789",
  "status": "completed",
  "progress_percent": 100,
  "current_step": "Processing complete!",
  "clips_completed": 5,
  "total_clips": 5,
  "output": {
    "job_id": "abc-123-def-456",
    "source_video_url": "https://youtube.com/watch?v=...",
    "source_video_title": "My Amazing Video",
    "source_video_duration_seconds": 1847.5,
    "total_clips": 5,
    "clips": [
      {
        "clip_index": 0,
        "s3_url": "https://viewcreator-media.s3.amazonaws.com/clips/user_789/abc-123/clip_00.mp4",
        "duration_ms": 45000,
        "start_time_ms": 134200,
        "end_time_ms": 179200,
        "virality_score": 0.92,
        "layout_type": "screen_share",
        "summary": "Key insight about product pricing strategy",
        "tags": ["business", "pricing", "strategy"]
      },
      {
        "clip_index": 1,
        "s3_url": "https://viewcreator-media.s3.amazonaws.com/clips/user_789/abc-123/clip_01.mp4",
        "duration_ms": 38000,
        "start_time_ms": 445000,
        "end_time_ms": 483000,
        "virality_score": 0.87,
        "layout_type": "talking_head",
        "summary": "Emotional story about customer success",
        "tags": ["story", "customers", "success"]
      }
    ],
    "transcript_url": "https://viewcreator-media.s3.amazonaws.com/clips/user_789/abc-123/transcript.json",
    "plan_url": "https://viewcreator-media.s3.amazonaws.com/clips/user_789/abc-123/plan.json",
    "processing_time_seconds": 482.5,
    "created_at": "2025-01-15T10:38:00Z"
  }
}
```

### Event: job.failed

Sent when job fails.

```json
{
  "event": "job.failed",
  "timestamp": "2025-01-15T10:32:00Z",
  "job_id": "abc-123-def-456",
  "external_job_id": "user-submitted-id",
  "owner_user_id": "user_789",
  "status": "failed",
  "progress_percent": 15,
  "current_step": "Processing failed",
  "error": "Failed to download video: Video is unavailable or private"
}
```

---

## Implementation Plan

### Phase 1: Webhook Service (Worker Side)

Create a new service to handle webhook delivery with retries.

**File: `app/services/webhook_service.py`**

```python
"""
Webhook service for sending job status updates to callback URLs.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebhookResult:
    """Result of a webhook delivery attempt."""
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 0


class WebhookService:
    """
    Service for delivering webhook notifications with retry logic.

    Features:
    - Automatic retries with exponential backoff
    - Non-blocking async delivery
    - Configurable timeouts and retry limits
    """

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds

    async def send(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> WebhookResult:
        """
        Send a webhook with automatic retries.

        Args:
            url: The callback URL to send the webhook to
            payload: The JSON payload to send
            headers: Optional additional headers

        Returns:
            WebhookResult with success status and details
        """
        if not url:
            return WebhookResult(success=False, error="No callback URL provided")

        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "ViewCreator-ClippingWorker/2.0",
            "X-Webhook-Event": payload.get("event", "unknown"),
        }
        if headers:
            default_headers.update(headers)

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=default_headers,
                    )

                    if response.status_code < 300:
                        logger.info(
                            f"Webhook delivered successfully: {url} "
                            f"(attempt {attempt}, status {response.status_code})"
                        )
                        return WebhookResult(
                            success=True,
                            status_code=response.status_code,
                            attempts=attempt,
                        )
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger.warning(
                            f"Webhook failed: {url} (attempt {attempt}, {last_error})"
                        )

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning(f"Webhook timeout: {url} (attempt {attempt})")

            except httpx.RequestError as e:
                last_error = f"Request error: {str(e)}"
                logger.warning(f"Webhook error: {url} (attempt {attempt}, {last_error})")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        logger.error(f"Webhook failed after {self.max_retries} attempts: {url}")
        return WebhookResult(
            success=False,
            error=last_error,
            attempts=self.max_retries,
        )

    async def send_fire_and_forget(
        self,
        url: str,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Send a webhook without waiting for the result.

        Use this for progress updates where delivery is best-effort.
        """
        asyncio.create_task(self.send(url, payload, headers))

    def build_payload(
        self,
        event: str,
        job_id: str,
        status: str,
        progress_percent: float,
        current_step: str,
        external_job_id: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        clips_completed: int = 0,
        total_clips: int = 0,
        error: Optional[str] = None,
        output: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Build a standardized webhook payload."""
        payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "job_id": job_id,
            "status": status,
            "progress_percent": progress_percent,
            "current_step": current_step,
        }

        if external_job_id:
            payload["external_job_id"] = external_job_id
        if owner_user_id:
            payload["owner_user_id"] = owner_user_id
        if clips_completed or total_clips:
            payload["clips_completed"] = clips_completed
            payload["total_clips"] = total_clips
        if error:
            payload["error"] = error
        if output:
            payload["output"] = output

        return payload
```

### Phase 2: Integrate with Pipeline

Modify `AIClippingPipeline` to send webhooks at each stage.

**File: `app/services/ai_clipping_pipeline.py` (modifications)**

```python
# Add to imports
from app.services.webhook_service import WebhookService

# Add to __init__
class AIClippingPipeline:
    def __init__(
        self,
        detection_pipeline: DetectionPipeline,
        progress_callback: Optional[Callable[[ClippingJobProgress], None]] = None,
    ):
        # ... existing code ...
        self.webhook_service = WebhookService()

    # Modify _update_progress to include webhook delivery
    def _update_progress(
        self,
        job_id: str,
        status: JobStatus,
        progress: float,
        step: str,
        callback_url: Optional[str] = None,  # NEW PARAMETER
        external_job_id: Optional[str] = None,  # NEW PARAMETER
        owner_user_id: Optional[str] = None,  # NEW PARAMETER
        clips_completed: int = 0,
        total_clips: int = 0,
        error: Optional[str] = None,
        output: Optional[dict] = None,  # NEW PARAMETER for completion
    ) -> None:
        """Update job progress via callback and webhook."""

        # Existing in-memory progress callback
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

        # NEW: Send webhook notification
        if callback_url:
            # Determine event type
            if status == JobStatus.COMPLETED:
                event = "job.completed"
            elif status == JobStatus.FAILED:
                event = "job.failed"
            elif progress == 0:
                event = "job.started"
            else:
                event = "job.progress"

            payload = self.webhook_service.build_payload(
                event=event,
                job_id=job_id,
                status=status.value,
                progress_percent=progress,
                current_step=step,
                external_job_id=external_job_id,
                owner_user_id=owner_user_id,
                clips_completed=clips_completed,
                total_clips=total_clips,
                error=error,
                output=output,
            )

            # Fire and forget for progress, await for completion/failure
            if event in ("job.completed", "job.failed"):
                asyncio.create_task(self.webhook_service.send(callback_url, payload))
            else:
                self.webhook_service.send_fire_and_forget(callback_url, payload)
```

### Phase 3: Update Pipeline Calls

Update all `_update_progress` calls in `process_video()` to pass the callback URL.

```python
async def process_video(self, request: ClippingJobRequest) -> ClippingJobResult:
    # ... existing code ...

    # Store callback info for all progress updates
    callback_url = request.callback_url
    external_job_id = request.external_job_id if hasattr(request, 'external_job_id') else None
    owner_user_id = request.owner_user_id

    # Step 1: Download video
    self._update_progress(
        job_id, JobStatus.DOWNLOADING, 5, "Downloading video...",
        callback_url=callback_url,
        external_job_id=external_job_id,
        owner_user_id=owner_user_id,
    )

    # ... continue for all stages ...

    # Final completion
    self._update_progress(
        job_id, JobStatus.COMPLETED, 100, "Processing complete!",
        callback_url=callback_url,
        external_job_id=external_job_id,
        owner_user_id=owner_user_id,
        clips_completed=total_clips,
        total_clips=total_clips,
        output=job_output.to_dict(),  # Include full output
    )
```

### Phase 4: Add to Job Request Schema

Ensure `callback_url` is properly passed through.

**File: `app/services/ai_clipping_pipeline.py`**

```python
@dataclass
class ClippingJobRequest:
    """Request to process a video for AI clipping."""

    video_url: str
    job_id: Optional[str] = None
    external_job_id: Optional[str] = None  # ADD THIS
    owner_user_id: Optional[str] = None
    max_clips: int = 5
    min_clip_duration_seconds: int = 15
    max_clip_duration_seconds: int = 90
    duration_ranges: Optional[list[str]] = None
    target_platform: str = "tiktok"
    include_captions: bool = True
    caption_style: Optional[CaptionStyle] = None
    callback_url: Optional[str] = None  # ALREADY EXISTS - ensure it's used
```

---

## viewcreator-api Integration

### Webhook Endpoint (NestJS)

**File: `src/modules/webhooks/webhooks.controller.ts`**

```typescript
import { Controller, Post, Body, Headers, HttpCode, Logger } from '@nestjs/common';
import { WebhooksService } from './webhooks.service';
import { ClippingWebhookDto } from './dto/clipping-webhook.dto';

@Controller('webhooks')
export class WebhooksController {
  private readonly logger = new Logger(WebhooksController.name);

  constructor(private readonly webhooksService: WebhooksService) {}

  @Post('clipping')
  @HttpCode(200)
  async handleClippingWebhook(
    @Body() payload: ClippingWebhookDto,
    @Headers('x-webhook-event') event: string,
  ) {
    this.logger.log(`Received webhook: ${event} for job ${payload.job_id}`);

    await this.webhooksService.processClippingWebhook(payload);

    return { received: true };
  }
}
```

**File: `src/modules/webhooks/dto/clipping-webhook.dto.ts`**

```typescript
import { IsString, IsNumber, IsOptional, IsObject, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';

class ClipArtifactDto {
  @IsNumber()
  clip_index: number;

  @IsString()
  s3_url: string;

  @IsNumber()
  duration_ms: number;

  @IsNumber()
  start_time_ms: number;

  @IsNumber()
  end_time_ms: number;

  @IsNumber()
  virality_score: number;

  @IsString()
  layout_type: string;

  @IsOptional()
  @IsString()
  summary?: string;

  @IsOptional()
  tags?: string[];
}

class JobOutputDto {
  @IsString()
  job_id: string;

  @IsString()
  source_video_url: string;

  @IsString()
  source_video_title: string;

  @IsNumber()
  source_video_duration_seconds: number;

  @IsNumber()
  total_clips: number;

  @ValidateNested({ each: true })
  @Type(() => ClipArtifactDto)
  clips: ClipArtifactDto[];

  @IsOptional()
  @IsString()
  transcript_url?: string;

  @IsOptional()
  @IsString()
  plan_url?: string;

  @IsNumber()
  processing_time_seconds: number;

  @IsString()
  created_at: string;
}

export class ClippingWebhookDto {
  @IsString()
  event: 'job.started' | 'job.progress' | 'job.completed' | 'job.failed';

  @IsString()
  timestamp: string;

  @IsString()
  job_id: string;

  @IsOptional()
  @IsString()
  external_job_id?: string;

  @IsOptional()
  @IsString()
  owner_user_id?: string;

  @IsString()
  status: string;

  @IsNumber()
  progress_percent: number;

  @IsString()
  current_step: string;

  @IsOptional()
  @IsNumber()
  clips_completed?: number;

  @IsOptional()
  @IsNumber()
  total_clips?: number;

  @IsOptional()
  @IsString()
  error?: string;

  @IsOptional()
  @ValidateNested()
  @Type(() => JobOutputDto)
  output?: JobOutputDto;
}
```

**File: `src/modules/webhooks/webhooks.service.ts`**

```typescript
import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ClippingJob } from '../ai-clipping-agent/entities/clipping-job.entity';
import { Clip } from '../ai-clipping-agent/entities/clip.entity';
import { NotificationsService } from '../notifications/notifications.service';
import { ClippingWebhookDto } from './dto/clipping-webhook.dto';

@Injectable()
export class WebhooksService {
  private readonly logger = new Logger(WebhooksService.name);

  constructor(
    @InjectRepository(ClippingJob)
    private readonly jobRepository: Repository<ClippingJob>,
    @InjectRepository(Clip)
    private readonly clipRepository: Repository<Clip>,
    private readonly notificationsService: NotificationsService,
  ) {}

  async processClippingWebhook(payload: ClippingWebhookDto): Promise<void> {
    const { event, job_id, external_job_id } = payload;

    // Find the job by external_job_id (our internal ID)
    const job = await this.jobRepository.findOne({
      where: { id: external_job_id || job_id },
      relations: ['user'],
    });

    if (!job) {
      this.logger.warn(`Job not found for webhook: ${job_id}`);
      return;
    }

    switch (event) {
      case 'job.started':
        await this.handleJobStarted(job, payload);
        break;

      case 'job.progress':
        await this.handleJobProgress(job, payload);
        break;

      case 'job.completed':
        await this.handleJobCompleted(job, payload);
        break;

      case 'job.failed':
        await this.handleJobFailed(job, payload);
        break;
    }
  }

  private async handleJobStarted(job: ClippingJob, payload: ClippingWebhookDto) {
    job.status = 'processing';
    job.workerJobId = payload.job_id;
    job.progressPercent = payload.progress_percent;
    job.currentStep = payload.current_step;
    await this.jobRepository.save(job);

    // Notify user via WebSocket
    await this.notificationsService.sendToUser(job.userId, {
      type: 'clipping.started',
      jobId: job.id,
      message: 'Your video is being processed',
    });
  }

  private async handleJobProgress(job: ClippingJob, payload: ClippingWebhookDto) {
    job.progressPercent = payload.progress_percent;
    job.currentStep = payload.current_step;
    job.clipsCompleted = payload.clips_completed || 0;
    job.totalClips = payload.total_clips || 0;
    await this.jobRepository.save(job);

    // Notify user via WebSocket (throttled to every 10%)
    if (Math.floor(payload.progress_percent / 10) > Math.floor((job.progressPercent - 1) / 10)) {
      await this.notificationsService.sendToUser(job.userId, {
        type: 'clipping.progress',
        jobId: job.id,
        progressPercent: payload.progress_percent,
        currentStep: payload.current_step,
      });
    }
  }

  private async handleJobCompleted(job: ClippingJob, payload: ClippingWebhookDto) {
    job.status = 'completed';
    job.progressPercent = 100;
    job.currentStep = 'Completed';
    job.completedAt = new Date();
    job.processingTimeSeconds = payload.output?.processing_time_seconds;
    job.sourceVideoTitle = payload.output?.source_video_title;
    job.sourceVideoDurationSeconds = payload.output?.source_video_duration_seconds;
    await this.jobRepository.save(job);

    // Save clips
    if (payload.output?.clips) {
      for (const clipData of payload.output.clips) {
        const clip = this.clipRepository.create({
          jobId: job.id,
          userId: job.userId,
          clipIndex: clipData.clip_index,
          s3Url: this.transformToCdnUrl(clipData.s3_url),
          durationMs: clipData.duration_ms,
          startTimeMs: clipData.start_time_ms,
          endTimeMs: clipData.end_time_ms,
          viralityScore: clipData.virality_score,
          layoutType: clipData.layout_type,
          summary: clipData.summary,
          tags: clipData.tags,
        });
        await this.clipRepository.save(clip);
      }
    }

    // Notify user
    await this.notificationsService.sendToUser(job.userId, {
      type: 'clipping.completed',
      jobId: job.id,
      clipCount: payload.output?.total_clips || 0,
      message: `Your ${payload.output?.total_clips} clips are ready!`,
    });
  }

  private async handleJobFailed(job: ClippingJob, payload: ClippingWebhookDto) {
    job.status = 'failed';
    job.error = payload.error;
    job.failedAt = new Date();
    await this.jobRepository.save(job);

    // Notify user
    await this.notificationsService.sendToUser(job.userId, {
      type: 'clipping.failed',
      jobId: job.id,
      error: payload.error,
      message: 'Video processing failed. Please try again.',
    });
  }

  private transformToCdnUrl(s3Url: string): string {
    // Transform S3 URL to CDN URL
    // s3://bucket/key → https://cdn.viewcreator.ai/key
    const cdnDomain = process.env.MEDIA_CDN_DOMAIN || 'cdn.viewcreator.ai';
    return s3Url.replace(
      /^https:\/\/[^\/]+\.s3\.[^\/]+\.amazonaws\.com\//,
      `https://${cdnDomain}/`,
    );
  }
}
```

---

## Submitting Jobs with Callback URL

**viewcreator-api submitting a job:**

```typescript
// In your clipping job submission service
async submitClippingJob(userId: string, request: CreateClippingJobDto) {
  // Create job record first
  const job = await this.jobRepository.save({
    userId,
    status: 'pending',
    videoUrl: request.videoUrl,
    // ... other fields
  });

  // Build callback URL
  const callbackUrl = `${process.env.API_BASE_URL}/webhooks/clipping`;

  // Submit to worker with callback
  const response = await this.httpService.post(
    `${process.env.CLIPPING_WORKER_URL}/ai-clipping/jobs`,
    {
      video_url: request.videoUrl,
      callback_url: callbackUrl,  // ← The webhook URL
      external_job_id: job.id,     // ← Our internal job ID
      owner_user_id: userId,
      max_clips: request.maxClips,
      duration_ranges: request.durationRanges,
      // ... other options
    },
  ).toPromise();

  // Update job with worker job ID
  job.workerJobId = response.data.job_id;
  job.status = 'submitted';
  await this.jobRepository.save(job);

  return job;
}
```

---

## Security Considerations

### 1. Webhook Authentication

Add a shared secret for webhook verification:

```python
# Worker: Add signature to webhook
import hmac
import hashlib

def sign_payload(payload: dict, secret: str) -> str:
    payload_str = json.dumps(payload, sort_keys=True)
    return hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

# Include in headers
headers = {
    "X-Webhook-Signature": sign_payload(payload, WEBHOOK_SECRET)
}
```

```typescript
// API: Verify signature
function verifyWebhookSignature(payload: any, signature: string): boolean {
  const expectedSignature = crypto
    .createHmac('sha256', process.env.WEBHOOK_SECRET)
    .update(JSON.stringify(payload))
    .digest('hex');

  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature),
  );
}
```

### 2. Rate Limiting

Limit webhook calls per job to prevent abuse:

```python
# Worker: Throttle progress updates
_last_webhook_time: dict[str, float] = {}
MIN_WEBHOOK_INTERVAL = 5.0  # seconds

def should_send_progress_webhook(job_id: str) -> bool:
    now = time.time()
    last = _last_webhook_time.get(job_id, 0)
    if now - last >= MIN_WEBHOOK_INTERVAL:
        _last_webhook_time[job_id] = now
        return True
    return False
```

### 3. Internal Network Only

In production, webhooks should only be sent within the VPC:

```
Worker (ECS) ──── internal ALB ────▶ API (ECS)
                  (not public)
```

---

## Testing

### Manual Testing

```bash
# 1. Submit job with callback URL
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "callback_url": "https://webhook.site/your-unique-id",
    "max_clips": 1
  }'

# 2. Watch webhooks arrive at webhook.site

# 3. Or use a local endpoint
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "callback_url": "http://localhost:3000/webhooks/clipping",
    "max_clips": 1
  }'
```

### Unit Tests

```python
# tests/test_webhook_service.py
import pytest
from unittest.mock import AsyncMock, patch
from app.services.webhook_service import WebhookService

@pytest.mark.asyncio
async def test_webhook_success():
    service = WebhookService()

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200

        result = await service.send(
            url="https://example.com/webhook",
            payload={"event": "job.started", "job_id": "123"},
        )

        assert result.success is True
        assert result.attempts == 1

@pytest.mark.asyncio
async def test_webhook_retry_on_failure():
    service = WebhookService(max_retries=3, retry_delay_seconds=0.1)

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 500

        result = await service.send(
            url="https://example.com/webhook",
            payload={"event": "job.started", "job_id": "123"},
        )

        assert result.success is False
        assert result.attempts == 3
```

---

## Implementation Checklist

- [ ] **Phase 1: Webhook Service**
  - [ ] Create `app/services/webhook_service.py`
  - [ ] Add retry logic with exponential backoff
  - [ ] Add payload builder helper

- [ ] **Phase 2: Pipeline Integration**
  - [ ] Modify `_update_progress()` to accept callback_url
  - [ ] Add webhook calls at each pipeline stage
  - [ ] Ensure output is included in completion webhook

- [ ] **Phase 3: API Integration**
  - [ ] Create webhook controller in viewcreator-api
  - [ ] Create DTOs for webhook payloads
  - [ ] Create webhook service to process events
  - [ ] Update job status in database
  - [ ] Send WebSocket notifications to users

- [ ] **Phase 4: Security**
  - [ ] Add webhook signature verification
  - [ ] Implement rate limiting
  - [ ] Configure internal-only networking in AWS

- [ ] **Phase 5: Testing**
  - [ ] Unit tests for webhook service
  - [ ] Integration tests with mock receiver
  - [ ] End-to-end test with real job

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WEBHOOK IMPLEMENTATION                                   │
│                                                                                  │
│  Benefits:                                                                       │
│  ✅ Real-time status updates (no polling delay)                                 │
│  ✅ 95% reduction in HTTP requests                                              │
│  ✅ Better user experience (instant progress)                                   │
│  ✅ Reduced server load on both services                                        │
│  ✅ Cleaner separation of concerns                                              │
│                                                                                  │
│  Implementation Effort:                                                          │
│  • Worker side: ~100 lines of new code                                          │
│  • API side: ~200 lines of new code                                             │
│  • Testing: ~50 lines                                                           │
│  • Total: ~1-2 days of work                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
