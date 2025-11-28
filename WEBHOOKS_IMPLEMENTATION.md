# Webhooks Implementation

This document outlines the webhook system for real-time job status updates between the clipping worker and viewcreator-api.

> **Last Updated**: 2025-11-28
>
> **Entity Reference**: `AiClippingAgentJob` (table: `ai_clipping_agent_jobs`)
>
> **Implementation Status**: ✅ Complete and Working

---

## Current Implementation Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `AiClippingAgentJob` entity | ✅ Complete | viewcreator-database | Typed columns for progress tracking |
| `ClippingWorkerClientService` | ✅ Complete | viewcreator-api | Supports `callback_url` parameter |
| `WebhookService` (Python) | ✅ Complete | viewcreator-genesis | `app/services/webhook_service.py` |
| Pipeline Integration | ✅ Complete | viewcreator-genesis | `ai_clipping_pipeline.py` modified |
| Webhook endpoint (API) | ✅ Complete | viewcreator-api | `/api/webhooks/ai-clipping` |
| `WebhooksService` (NestJS) | ✅ Complete | viewcreator-api | `src/modules/webhooks/` |
| Callback URL passing | ✅ Complete | viewcreator-api | `AiClippingAgentJobsService` |
| Polling fallback | ❌ Removed | viewcreator-api | Webhooks are the sole mechanism |
| Real-time notifications | ⏳ Future | - | WebSocket integration pending |
| Webhook authentication | ⏳ Future | - | HMAC signature verification |

**Implementation approach**: Webhooks only (polling removed for simplicity)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WEBHOOK FLOW ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│                  │         │                  │         │                  │
│  viewcreator-api │◀────────│ viewcreator-     │────────▶│    AWS S3        │
│    (NestJS)      │ webhook │ genesis (FastAPI)│ upload  │                  │
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


Flow:
──────
1. API creates AiClippingAgentJob, submits to genesis with callback_url
2. Genesis processes video, sends webhooks at each stage
3. API receives webhooks, updates job record in PostgreSQL
4. Frontend polls API for status (can be replaced with WebSockets later)
```

---

## Implemented Components

### 1. Genesis Worker (viewcreator-genesis)

#### WebhookService (`app/services/webhook_service.py`)

```python
# Key features:
- Automatic retries with exponential backoff (3 retries, 1s base delay)
- Throttling for progress webhooks (2s minimum interval)
- Fire-and-forget for progress updates, awaited for terminal events
- Standardized payload builder with WebhookPayload dataclass

# Usage:
webhook_service = get_webhook_service()
payload = webhook_service.build_payload(
    event="job.progress",
    job_id=job_id,
    status="running",
    progress_percent=50,
    current_step="Rendering clips...",
)
await webhook_service.send(callback_url, payload)
```

#### Pipeline Integration (`app/services/ai_clipping_pipeline.py`)

```python
# Modified components:
- ClippingJobRequest: Added external_job_id field
- AIClippingPipeline.__init__: Added webhook_service
- AIClippingPipeline._update_progress(): Sends webhooks when callback_url set
- AIClippingPipeline._send_webhook(): Maps JobStatus to webhook events

# Webhook events sent at:
- Job start: event="job.started"
- Progress updates: event="job.progress" (throttled to every 2s)
- Completion: event="job.completed" with full output
- Failure: event="job.failed" with error message
```

#### Router Update (`app/routers/ai_clipping.py`)

```python
# ClippingJobRequest now includes:
job_request = ClippingJobRequest(
    video_url=video_source,
    job_id=request.external_job_id,
    external_job_id=request.external_job_id,  # For webhook identification
    owner_user_id=request.owner_user_id,
    callback_url=request.callback_url,
    # ... other fields
)
```

### 2. API Backend (viewcreator-api)

#### WebhooksModule (`src/modules/webhooks/`)

```
src/modules/webhooks/
├── dto/
│   └── ai-clipping-webhook.dto.ts   # Validation DTOs
├── webhooks.controller.ts            # POST /webhooks/ai-clipping
├── webhooks.service.ts               # Process webhooks, update jobs
└── webhooks.module.ts                # Module registration
```

#### WebhooksController

```typescript
@Controller('webhooks')
export class WebhooksController {
  @Post('ai-clipping')
  @HttpCode(HttpStatus.OK)
  async handleAiClippingWebhook(
    @Body() payload: AiClippingWebhookDto,
    @Headers('authorization') authorization?: string,
  ): Promise<{ received: boolean }> {
    // Validates signature (if WEBHOOK_SECRET configured)
    // Processes webhook via service
    // Returns 200 to prevent retries
  }
}
```

#### WebhooksService

```typescript
@Injectable()
export class WebhooksService {
  async processAiClippingWebhook(webhook: AiClippingWebhookDto): Promise<void> {
    // Finds job by external_job_id
    // Updates job record based on event type:
    //   - job.started: Set startedAt, status=RUNNING
    //   - job.progress: Update progressPercent, progressStage, clipsCompleted
    //   - job.completed: Save clips, calculate credits, status=SUCCEEDED
    //   - job.failed: Set errorMessage, status=FAILED
  }
}
```

#### AiClippingAgentJobsService Updates

```typescript
// New constructor properties:
private readonly webhookBaseUrl: string | null;  // From API_BASE_URL env
private readonly webhookSecret: string | null;   // From WEBHOOK_SECRET env

// New method:
private getWebhookCallbackUrl(): string | null {
  if (!this.webhookBaseUrl) return null;
  // Include /api prefix for NestJS global prefix
  return `${this.webhookBaseUrl}/api/webhooks/ai-clipping`;
}

// Modified submitJob call:
const callbackUrl = this.getWebhookCallbackUrl();
const workerResponse = await this.clippingWorkerClient.submitJob({
  // ... existing fields ...
  callback_url: callbackUrl || undefined,
  external_job_id: saved.id,
  owner_user_id: userId,
});

// Note: Polling was removed - webhooks are the sole update mechanism
```

---

## Webhook Payload Format

### All Events

```typescript
interface WebhookPayload {
  event: 'job.started' | 'job.progress' | 'job.completed' | 'job.failed';
  timestamp: string;  // ISO 8601
  job_id: string;     // Genesis worker job ID
  external_job_id?: string;  // API's AiClippingAgentJob.id
  owner_user_id?: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  progress_percent: number;
  current_step: string;
  clips_completed?: number;
  total_clips?: number;
  error?: string;
  output?: {
    total_clips: number;
    source_video_title?: string;
    processing_time_seconds?: number;
    clips: Array<{
      clip_index: number;
      s3_url: string;
      duration_ms: number;
      virality_score: number;
      summary?: string;
    }>;
    transcript_url?: string;
    plan_url?: string;
  };
}
```

### Event Examples

**job.started:**
```json
{
  "event": "job.started",
  "timestamp": "2024-11-28T10:30:00.000Z",
  "job_id": "abc-123",
  "external_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "owner_user_id": "user_789",
  "status": "running",
  "progress_percent": 5,
  "current_step": "Downloading video..."
}
```

**job.progress:**
```json
{
  "event": "job.progress",
  "timestamp": "2024-11-28T10:32:00.000Z",
  "job_id": "abc-123",
  "external_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress_percent": 65,
  "current_step": "Rendering clip 3/5...",
  "clips_completed": 2,
  "total_clips": 5
}
```

**job.completed:**
```json
{
  "event": "job.completed",
  "timestamp": "2024-11-28T10:38:00.000Z",
  "job_id": "abc-123",
  "external_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "succeeded",
  "progress_percent": 100,
  "current_step": "Processing complete!",
  "clips_completed": 5,
  "total_clips": 5,
  "output": {
    "total_clips": 5,
    "source_video_title": "My Amazing Video",
    "processing_time_seconds": 482.5,
    "clips": [
      {
        "clip_index": 0,
        "s3_url": "https://bucket.s3.amazonaws.com/clips/clip_00.mp4",
        "duration_ms": 45000,
        "virality_score": 0.92,
        "summary": "Key insight about..."
      }
    ],
    "transcript_url": "https://bucket.s3.amazonaws.com/transcript.json",
    "plan_url": "https://bucket.s3.amazonaws.com/plan.json"
  }
}
```

**job.failed:**
```json
{
  "event": "job.failed",
  "timestamp": "2024-11-28T10:32:00.000Z",
  "job_id": "abc-123",
  "external_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "progress_percent": 15,
  "current_step": "Processing failed",
  "error": "Failed to download video: Video is unavailable"
}
```

---

## Configuration

### Environment Variables

**viewcreator-api (.env):**
```env
# Required for webhook callback URL generation
# The callback URL will be: {API_BASE_URL}/api/webhooks/ai-clipping

# Production (AWS):
API_BASE_URL=https://api.viewcreator.ai

# Local development (API on host, genesis in Docker):
API_BASE_URL=http://host.docker.internal:3001

# Local development (both on host):
API_BASE_URL=http://localhost:3001

# Optional: Webhook authentication (future)
WEBHOOK_SECRET=your-secret-key-here
```

**viewcreator-genesis:**
No additional configuration needed. Callback URL is passed per-job.

### Docker Networking Notes

When genesis runs in Docker and the API runs on your host machine:
- Use `host.docker.internal` instead of `localhost` in `API_BASE_URL`
- This allows the Docker container to reach services on the host
- On Linux, you may need to add `--add-host=host.docker.internal:host-gateway` to docker run

---

## Testing

### Manual Testing with webhook.site

```bash
# Submit job with external callback URL for debugging
curl -X POST http://localhost:8000/ai-clipping/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "callback_url": "https://webhook.site/your-unique-id",
    "external_job_id": "test-123",
    "max_clips": 1
  }'
```

### End-to-End Testing

```bash
# 1. Start API (ensure API_BASE_URL is set)
cd viewcreator-api && npm run start:dev

# 2. Start Genesis worker
cd viewcreator-genesis && uvicorn app.main:app --reload

# 3. Submit job via API (webhooks will be sent automatically)
curl -X POST http://localhost:3000/ai-clipping-agent/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "youtubeUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "requestedClipCount": 1
  }'

# 4. Monitor API logs for webhook reception
# 5. Check job status in database
```

---

## Implementation Checklist

### Completed ✅

#### Genesis Worker (viewcreator-genesis)
- [x] Create `app/services/webhook_service.py`
- [x] Add WebhookPayload and WebhookResult dataclasses
- [x] Implement retry logic with exponential backoff
- [x] Add throttling for progress webhooks
- [x] Add payload builder helper
- [x] Integrate with `AIClippingPipeline._update_progress()`
- [x] Add `_send_webhook()` method to pipeline
- [x] Map JobStatus to webhook event types
- [x] Add `external_job_id` to ClippingJobRequest
- [x] Pass external_job_id through router

#### API Backend (viewcreator-api)
- [x] Create `src/modules/webhooks/` module
- [x] Create `WebhooksController` with `/api/webhooks/ai-clipping` endpoint
- [x] Create `AiClippingWebhookDto` for validation
- [x] Create `WebhooksService` to process events
- [x] Register module in `app.module.ts`
- [x] Update `AiClippingAgentJobsService` with webhook config
- [x] Add `getWebhookCallbackUrl()` method (includes `/api` prefix)
- [x] Pass `callback_url` in submitJob call
- [x] Remove polling system (webhooks only)

### Pending ⏳

#### Security (Phase 4)
- [ ] Add HMAC signature generation in genesis
- [ ] Add signature verification in API controller
- [ ] Configure internal-only networking in AWS (VPC)
- [ ] Add rate limiting on webhook endpoint

#### Real-time Notifications (Phase 5)
- [ ] Create `NotificationsService` for WebSocket support
- [ ] Send progress updates to connected clients
- [ ] Throttle notifications (every 10% progress)

#### Testing (Phase 6)
- [ ] Unit tests for `WebhookService` (Python)
- [ ] Unit tests for `WebhooksService` (NestJS)
- [ ] Integration tests with mock worker
- [ ] End-to-end test with real clipping job

---

## Benefits

| Aspect | Before (Polling) | After (Webhooks) |
|--------|------------------|------------------|
| HTTP requests per job | 50-100+ | 3-6 |
| Status update latency | 5-10 seconds | Instant |
| Server load | High | Low |
| User experience | Laggy progress | Real-time |
| Scalability | Poor | Excellent |

---

## File Changes Summary

### viewcreator-genesis

| File | Change |
|------|--------|
| `app/services/webhook_service.py` | **NEW** - Webhook delivery service |
| `app/services/ai_clipping_pipeline.py` | Modified - Added webhook integration |
| `app/routers/ai_clipping.py` | Modified - Pass external_job_id |

### viewcreator-api

| File | Change |
|------|--------|
| `src/modules/webhooks/webhooks.module.ts` | **NEW** - Module definition |
| `src/modules/webhooks/webhooks.controller.ts` | **NEW** - Webhook endpoint |
| `src/modules/webhooks/webhooks.service.ts` | **NEW** - Process webhooks |
| `src/modules/webhooks/dto/ai-clipping-webhook.dto.ts` | **NEW** - Validation DTOs |
| `src/modules/ai-clipping-agent/ai-clipping-agent-jobs.service.ts` | Modified - Add callback URL |
| `src/app.module.ts` | Modified - Import WebhooksModule |
