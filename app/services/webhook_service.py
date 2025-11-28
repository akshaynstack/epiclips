"""
Webhook Service - HTTP client for sending job status webhooks to callback URLs.

This service handles webhook delivery with:
- Automatic retries with exponential backoff
- Non-blocking async delivery
- Configurable timeouts
- HMAC-SHA256 signature for authenticity
- Comprehensive logging
"""

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class WebhookResult:
    """Result of a webhook delivery attempt."""

    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class WebhookPayload:
    """Standardized webhook payload structure."""

    event: str  # job.started, job.progress, job.completed, job.failed
    timestamp: str  # ISO 8601
    job_id: str
    status: str
    progress_percent: float
    current_step: str
    external_job_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    clips_completed: int = 0
    total_clips: int = 0
    error: Optional[str] = None
    output: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values for optional fields."""
        data = {
            "event": self.event,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
        }

        # Include optional fields only if set
        if self.external_job_id:
            data["external_job_id"] = self.external_job_id
        if self.owner_user_id:
            data["owner_user_id"] = self.owner_user_id
        if self.clips_completed or self.total_clips:
            data["clips_completed"] = self.clips_completed
            data["total_clips"] = self.total_clips
        if self.error:
            data["error"] = self.error
        if self.output:
            data["output"] = self.output

        return data


class WebhookService:
    """
    Service for delivering webhook notifications with retry logic.

    Features:
    - Automatic retries with exponential backoff
    - Non-blocking async delivery
    - Configurable timeouts and retry limits
    - Rate limiting to avoid overwhelming receivers
    - HMAC-SHA256 signatures for payload authenticity
    """

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        min_interval_seconds: float = 2.0,
    ):
        """
        Initialize the webhook service.

        Args:
            timeout_seconds: HTTP request timeout
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Base delay between retries (exponential backoff)
            min_interval_seconds: Minimum interval between webhooks for same job
        """
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self.min_interval = min_interval_seconds

        # Track last webhook time per job for throttling
        self._last_webhook_time: dict[str, float] = {}

        # Get webhook signing secret from config
        settings = get_settings()
        self._webhook_secret = settings.genesis_webhook_secret
        if self._webhook_secret:
            logger.info("Webhook HMAC signing enabled")
        else:
            logger.warning(
                "GENESIS_WEBHOOK_SECRET not configured - webhooks will not be signed"
            )

    def _sign_payload(self, payload_json: str) -> str:
        """
        Generate HMAC-SHA256 signature for a webhook payload.

        Args:
            payload_json: The JSON-encoded payload string

        Returns:
            Signature string in format: sha256=<hex-signature>
        """
        if not self._webhook_secret:
            return ""

        signature = hmac.new(
            self._webhook_secret.encode("utf-8"),
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    def should_send_progress(self, job_id: str) -> bool:
        """
        Check if enough time has passed since the last progress webhook.
        Used to throttle progress updates (completion/failure always sent).
        """
        import time

        now = time.time()
        last = self._last_webhook_time.get(job_id, 0)

        if now - last >= self.min_interval:
            self._last_webhook_time[job_id] = now
            return True
        return False

    def clear_job_tracking(self, job_id: str) -> None:
        """Clear tracking data for a completed job."""
        self._last_webhook_time.pop(job_id, None)

    async def send(
        self,
        url: str,
        payload: WebhookPayload,
        headers: Optional[dict[str, str]] = None,
    ) -> WebhookResult:
        """
        Send a webhook with automatic retries.

        Args:
            url: The callback URL to send the webhook to
            payload: The webhook payload to send
            headers: Optional additional headers

        Returns:
            WebhookResult with success status and details
        """
        if not url:
            return WebhookResult(success=False, error="No callback URL provided")

        payload_dict = payload.to_dict()
        # Serialize to JSON once for consistent signing
        payload_json = json.dumps(payload_dict, separators=(",", ":"), sort_keys=True)

        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "ViewCreator-Genesis/2.0",
            "X-Webhook-Event": payload.event,
            "X-Job-Id": payload.job_id,
        }

        # Add HMAC signature if secret is configured
        signature = self._sign_payload(payload_json)
        if signature:
            default_headers["X-Genesis-Webhook-Signature"] = signature

        if headers:
            default_headers.update(headers)

        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        content=payload_json,  # Use pre-serialized JSON for signature consistency
                        headers=default_headers,
                    )

                    if response.status_code < 300:
                        logger.info(
                            f"Webhook delivered: {payload.event} for job {payload.job_id} "
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

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.exception(f"Webhook unexpected error: {url} (attempt {attempt})")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        logger.error(
            f"Webhook failed after {self.max_retries} attempts: {url} "
            f"(job {payload.job_id}, event {payload.event})"
        )
        return WebhookResult(
            success=False,
            error=last_error,
            attempts=self.max_retries,
        )

    async def send_fire_and_forget(
        self,
        url: str,
        payload: WebhookPayload,
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
        output: Optional[dict[str, Any]] = None,
    ) -> WebhookPayload:
        """Build a standardized webhook payload."""
        return WebhookPayload(
            event=event,
            timestamp=datetime.now(timezone.utc).isoformat(),
            job_id=job_id,
            status=status,
            progress_percent=progress_percent,
            current_step=current_step,
            external_job_id=external_job_id,
            owner_user_id=owner_user_id,
            clips_completed=clips_completed,
            total_clips=total_clips,
            error=error,
            output=output,
        )


# Global singleton instance
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get or create the global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service
