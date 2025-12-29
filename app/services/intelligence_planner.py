"""
Intelligence Planner Service - Uses Gemini via OpenRouter to plan viral clips.
"""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from app.config import get_settings
from app.services.transcription_service import (
    TranscriptSegment,
    TranscriptionResult,
    find_sentence_end_boundary,
    find_sentence_start_boundary,
)

logger = logging.getLogger(__name__)


# Layout types for clips (internal representation)
# - screen_share: Split layout with screen on top, face on bottom
# - talking_head: Face-focused dynamic crop that follows the speaker
ClipLayoutType = Literal["talking_head", "screen_share"]

# Map user-facing layout IDs to internal layout types
LAYOUT_TYPE_MAP = {
    "split_screen": "screen_share",    # User-facing -> internal
    "talking_head": "talking_head",
    # Backwards compatibility for internal types used directly
    "screen_share": "screen_share",
}


@dataclass
class ClipPlanSegment:
    """A planned clip segment with timing and metadata."""
    
    start_time_ms: int
    end_time_ms: int
    virality_score: float
    layout_type: ClipLayoutType
    summary: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ClipPlanResponse:
    """Response from clip planning."""
    
    segments: list[ClipPlanSegment]  # Renamed from clips to match pipeline expectation
    total_clips: int = 0
    target_platform: str = "tiktok"
    insights: Optional[str] = None


@dataclass
class VisionFrame:
    """A video frame for vision analysis."""
    
    timestamp_ms: int
    file_path: str
    width: int
    height: int


class IntelligencePlannerService:
    """
    Service for planning viral clips using Gemini via OpenRouter.
    
    Features:
    - Multimodal analysis (transcript + vision frames)
    - Identifies viral-worthy segments with timestamps
    - Assigns layout types (talking_head, screen_share)
    - Scores clips for virality potential
    - Automatic clip count scaling based on video duration
    """

    def __init__(self):
        self.settings = get_settings()
        self._http_client: Optional[httpx.AsyncClient] = None
        
        if not self.settings.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set, intelligence planning will fail")

    def calculate_optimal_clip_count(
        self,
        video_duration_seconds: float,
        user_max_clips: Optional[int],
        auto_clip_count: bool = True,
    ) -> int:
        """
        Calculate optimal clip count based on video duration.
        
        Uses a scaled approach: longer videos get more clips, but with diminishing
        returns to prevent excessive clips on very long videos.
        
        Args:
            video_duration_seconds: Total video duration in seconds
            user_max_clips: User-requested maximum clips (upper bound). None = use config max_clips_absolute
            auto_clip_count: If True, auto-scale based on duration. If False, use user's max_clips directly.
            
        Returns:
            Optimal number of clips to generate
        """
        # Use config's max_clips_absolute as default when user_max_clips is None
        effective_max = user_max_clips if user_max_clips is not None else self.settings.max_clips_absolute
        
        if not auto_clip_count or not self.settings.clip_scaling_enabled:
            # Auto-scaling disabled by request or config, use user's requested count directly
            final_clips = min(effective_max, self.settings.max_clips_absolute)
            logger.info(
                f"Clip count (auto-scaling OFF): using max {effective_max} -> final: {final_clips}"
            )
            return final_clips
        
        video_duration_minutes = video_duration_seconds / 60.0
        
        # Calculate suggested clips based on duration
        # Using clips_per_minute_ratio (default 0.2 = 1 clip per 5 minutes)
        suggested_clips = int(video_duration_minutes * self.settings.clips_per_minute_ratio)
        
        # Apply bounds
        suggested_clips = max(suggested_clips, self.settings.min_clips)
        suggested_clips = min(suggested_clips, self.settings.max_clips_absolute)
        
        # Don't exceed user's requested maximum (if provided)
        final_clips = min(suggested_clips, effective_max)
        
        logger.info(
            f"Clip count scaling: {video_duration_minutes:.1f} min video -> "
            f"suggested {suggested_clips} clips (ratio: {self.settings.clips_per_minute_ratio}), "
            f"effective max: {effective_max}, final: {final_clips}"
        )
        
        return final_clips

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.openrouter_base_url,
                timeout=httpx.Timeout(300.0, connect=30.0),
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "HTTP-Referer": "https://epirium.ai",
                    "X-Title": "Epirium AI Clipping Agent",
                },
            )
        return self._http_client

    async def plan_clips(
        self,
        transcript_result: TranscriptionResult,
        video_metadata: Any = None,
        max_clips: int = 5,
        auto_clip_count: bool = True,
        min_duration_seconds: int = 15,
        max_duration_seconds: int = 90,
        duration_ranges: Optional[list[str]] = None,
        target_platform: str = "tiktok",
        frames: Optional[list[VisionFrame]] = None,
        layout_type: str = "split_screen",
        start_time_seconds: Optional[float] = None,
        end_time_seconds: Optional[float] = None,
    ) -> ClipPlanResponse:
        """
        Plan viral clips from video content.

        Args:
            transcript_result: TranscriptionResult from transcription service
            video_metadata: Optional video metadata
            max_clips: Maximum number of clips to generate
            auto_clip_count: If True, auto-scale clip count based on video duration
            min_duration_seconds: Minimum clip duration
            max_duration_seconds: Maximum clip duration
            duration_ranges: Optional list of selected duration ranges ('short', 'medium', 'long')
            target_platform: Target platform (tiktok, youtube_shorts, instagram_reels)
            frames: Optional sampled video frames for vision analysis
            layout_type: Layout type for clip rendering (split_screen, talking_head)
            start_time_seconds: Optional start of processing range (clips only from this point)
            end_time_seconds: Optional end of processing range (clips only until this point)

        Returns:
            ClipPlanResponse with identified clips
        """
        transcript = transcript_result.segments if transcript_result else []
        frames = frames or []

        if not transcript and not frames:
            logger.warning("No transcript or frames provided for clip planning")
            return ClipPlanResponse(segments=[], total_clips=0, target_platform=target_platform, insights="No content provided for analysis")

        # Store time range for validation
        self._start_time_seconds = start_time_seconds
        self._end_time_seconds = end_time_seconds
        
        # Filter transcript segments by time range if specified
        if start_time_seconds is not None or end_time_seconds is not None:
            start_ms = int((start_time_seconds or 0) * 1000)
            end_ms = int((end_time_seconds or float('inf')) * 1000)
            
            original_count = len(transcript)
            transcript = [
                seg for seg in transcript
                if seg.start_time_ms >= start_ms and seg.end_time_ms <= end_ms
            ]
            logger.info(
                f"Time range filter: {start_time_seconds}s - {end_time_seconds}s, "
                f"filtered {original_count} -> {len(transcript)} transcript segments"
            )
            
            # Also filter frames by time range
            if frames:
                original_frame_count = len(frames)
                frames = [
                    f for f in frames
                    if f.timestamp_ms >= start_ms and f.timestamp_ms <= end_ms
                ]
                logger.info(
                    f"Time range filter: filtered {original_frame_count} -> {len(frames)} frames"
                )

        # Calculate video duration from filtered transcript for clip count scaling
        video_duration_seconds = 0.0
        if transcript:
            # Use the range of the filtered transcript
            first_segment_start = transcript[0].start_time_ms / 1000.0
            last_segment_end = transcript[-1].end_time_ms / 1000.0
            video_duration_seconds = last_segment_end - first_segment_start
        
        # Apply clip count scaling algorithm
        user_max_clips = max_clips or self.settings.max_suggested_clips
        clip_count = self.calculate_optimal_clip_count(
            video_duration_seconds, 
            user_max_clips,
            auto_clip_count=auto_clip_count,
        )

        # Map user-facing layout type to internal type
        internal_layout_type = LAYOUT_TYPE_MAP.get(layout_type, "screen_share")

        logger.info(
            f"Planning clips: {len(transcript)} transcript segments, "
            f"{len(frames)} frames, requesting {clip_count} clips (scaled from user max {user_max_clips}), "
            f"duration_ranges={duration_ranges}, layout_type={layout_type} (internal: {internal_layout_type})"
        )

        # Store for use in response and sentence boundary snapping
        self._current_target_platform = target_platform
        self._current_min_duration = min_duration_seconds
        self._current_max_duration = max_duration_seconds
        self._current_duration_ranges = duration_ranges
        self._current_layout_type = internal_layout_type
        self._current_transcript = transcript  # Store for sentence boundary snapping
        
        # Build prompts
        system_prompt = self._build_system_prompt(clip_count, min_duration_seconds, max_duration_seconds, duration_ranges)
        transcript_text = self._build_transcript_text(transcript)
        
        logger.info(f"Transcript text length: {len(transcript_text)} chars, first 500 chars: {transcript_text[:500]}")
        
        # Load frames as base64 images (limit to max batch size)
        frames_to_send = frames[:self.settings.max_frames_per_vision_batch]
        frame_images = await self._load_frames_as_base64(frames_to_send)
        
        logger.debug(
            f"Loaded {len(frame_images)} frames as base64 images for vision analysis"
        )
        
        # Build multimodal message
        messages = self._build_vision_messages(
            system_prompt,
            transcript_text,
            frame_images,
            clip_count,
            transcript,
        )
        
        # Call LLM via OpenRouter with retry for JSON parsing failures
        logger.info(f"Calling {self.settings.openrouter_model} via OpenRouter...")
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            response = await self._call_openrouter(
                model=self.settings.openrouter_model,
                messages=messages,
                temperature=0.2,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            
            try:
                # Parse response
                return self._parse_clip_plan_response(response)
            except IntelligencePlanningError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Clip planning attempt {attempt + 1} failed (JSON parse error), retrying...")
                    import asyncio
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All {max_retries} clip planning attempts failed")
                    raise last_error

    def _build_system_prompt(
        self,
        clip_count: int,
        min_duration: int = 15,
        max_duration: int = 90,
        duration_ranges: Optional[list[str]] = None,
    ) -> str:
        """Build the system prompt for Gemini."""
        # Build strict duration guidance based on selected ranges
        duration_guidance = ""
        strict_bounds_text = f"STRICTLY between {min_duration} and {max_duration} seconds"
        
        if duration_ranges and len(duration_ranges) > 0:
            range_descriptions = {
                "short": {"text": "15-30 seconds (quick, punchy clips)", "min": 15, "max": 30},
                "medium": {"text": "30-60 seconds (moderate length clips)", "min": 30, "max": 60},
                "long": {"text": "60-120 seconds (longer, in-depth clips)", "min": 60, "max": 120},
            }
            selected_ranges = []
            actual_min = float('inf')
            actual_max = 0
            for r in duration_ranges:
                if r in range_descriptions:
                    selected_ranges.append(range_descriptions[r]["text"])
                    actual_min = min(actual_min, range_descriptions[r]["min"])
                    actual_max = max(actual_max, range_descriptions[r]["max"])
            
            if selected_ranges and actual_min != float('inf'):
                strict_bounds_text = f"STRICTLY between {int(actual_min)} and {int(actual_max)} seconds"
                duration_guidance = f"""
CRITICAL DURATION REQUIREMENTS:
The user has selected specific clip lengths. You MUST follow these EXACTLY:
{chr(10).join(f'- {r}' for r in selected_ranges)}

Each clip MUST be {strict_bounds_text}. Clips outside this range will be REJECTED.
Do NOT generate clips shorter than {int(actual_min)} seconds or longer than {int(actual_max)} seconds."""
        
        return f"""You are AI-Clipping-Agent, a virality analyst that identifies the most engaging segments from long-form videos for short-form content.

Your task:
1. Analyze the provided transcript and video frames to understand the content
2. Identify moments with high viral potential - strong hooks, emotional peaks, surprising reveals, or compelling visuals
3. Consider pacing, speaker energy, and visual interest when selecting clips
{duration_guidance}
Return exactly {clip_count} clips as a JSON object with this structure:
{{
  "clips": [
    {{
      "start_time": <number in seconds>,
      "end_time": <number in seconds>,
      "virality_score": <0.0 to 1.0>,
      "layout_type": "screen_share",
      "summary": "<brief description of why this clip is viral-worthy>",
      "tags": ["tag1", "tag2"]
    }}
  ],
  "insights": "<overall analysis of the video content>"
}}

STRICT RULES:
- DURATION: Each clip MUST be {strict_bounds_text}. This is NON-NEGOTIABLE.
- Verify: (end_time - start_time) >= {min_duration} AND (end_time - start_time) <= {max_duration}
- Prefer clips with strong opening hooks (first 3 seconds are critical)
- layout_type: ALWAYS use "screen_share" for optimal vertical video format
- Higher virality_score = higher confidence this clip will perform well
- Return times in SECONDS (not milliseconds)
- Clips that violate the duration requirements will be REJECTED"""

    def _build_transcript_text(self, transcript: list) -> str:
        """Build formatted transcript text."""
        if not transcript:
            return "[No transcript available]"
        
        # Debug: log segment structure (DEBUG level)
        if transcript and logger.isEnabledFor(logging.DEBUG):
            first_seg = transcript[0]
            logger.debug(f"First segment type: {type(first_seg)}")
            if hasattr(first_seg, 'text'):
                logger.debug(f"First segment text sample: '{first_seg.text[:50]}...'")
        
        lines = []
        for seg in transcript:
            time_str = f"[{seg.start_time_ms / 1000:.1f}s - {seg.end_time_ms / 1000:.1f}s]"
            speaker = f"({seg.speaker_label}) " if seg.speaker_label else ""
            lines.append(f"{time_str} {speaker}{seg.text}")
        
        return "\n".join(lines)

    async def _load_frames_as_base64(
        self,
        frames: list[VisionFrame],
    ) -> list[dict]:
        """Load frames as base64-encoded images."""
        results = []
        
        for frame in frames:
            try:
                with open(frame.file_path, "rb") as f:
                    image_data = f.read()
                
                base64_data = base64.b64encode(image_data).decode("utf-8")
                
                # Determine MIME type from extension
                ext = os.path.splitext(frame.file_path)[1].lower()
                mime_type = {
                    ".png": "image/png",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }.get(ext, "image/jpeg")
                
                results.append({
                    "base64": base64_data,
                    "mime_type": mime_type,
                    "timestamp_ms": frame.timestamp_ms,
                })
                
            except Exception as e:
                logger.warning(f"Failed to load frame {frame.file_path}: {e}")
        
        return results

    def _build_vision_messages(
        self,
        system_prompt: str,
        transcript_text: str,
        frame_images: list[dict],
        clip_count: int,
        transcript: list[TranscriptSegment],
    ) -> list[dict]:
        """Build multimodal messages for OpenRouter/Gemini."""
        # Build user content with text and images
        user_content = []
        
        # Add transcript as text
        video_duration = (
            transcript[-1].end_time_ms / 1000 if transcript else 0
        )
        
        user_content.append({
            "type": "text",
            "text": f"""Here is the transcript of the video:

{transcript_text}

The video is approximately {video_duration:.0f} seconds long.

Below are {len(frame_images)} sample frames from the video at regular intervals. Use these to understand the visual content and identify compelling moments.""",
        })
        
        # Add frames as images with timestamps
        for frame in frame_images:
            user_content.append({
                "type": "text",
                "text": f"Frame at {frame['timestamp_ms'] / 1000:.1f} seconds:",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{frame['mime_type']};base64,{frame['base64']}",
                    "detail": "low",  # Use low detail to reduce token count
                },
            })
        
        # Add final instruction
        user_content.append({
            "type": "text",
            "text": f"\nBased on the transcript and frames above, identify the {clip_count} most viral-worthy segments. Return your response as JSON.",
        })
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def _call_openrouter(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 8000,
        response_format: Optional[dict] = None,
    ) -> dict:
        """Call OpenRouter API."""
        client = await self._get_client()
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        response = await client.post(
            "/chat/completions",
            json=payload,
        )
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise IntelligencePlanningError(
                f"OpenRouter API error ({response.status_code}): {error_text}"
            )
        
        return response.json()

    def _parse_clip_plan_response(self, response: dict) -> ClipPlanResponse:
        """Parse OpenRouter response into ClipPlanResponse."""
        try:
            content = response["choices"][0]["message"]["content"]
            
            logger.debug(f"Raw Gemini response content: {content[:1000]}")
            
            # Try to parse JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'[\[{][\s\S]*[\]}]', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    logger.error(f"Failed to parse JSON from response: {content[:500]}")
                    raise IntelligencePlanningError(
                        f"Failed to parse JSON from response: {content[:200]}"
                    )
            
            logger.debug(f"Parsed response structure: {type(parsed).__name__}, keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
            
            # Handle array response (clips directly)
            if isinstance(parsed, list):
                clips_data = parsed
                insights = None
            else:
                clips_data = parsed.get("clips", [])
                insights = parsed.get("insights")
            
            logger.info(f"Found {len(clips_data)} clips in response before validation")
            
            # Get duration range bounds from stored values
            min_duration = getattr(self, '_current_min_duration', 15)
            max_duration = getattr(self, '_current_max_duration', 90)
            duration_ranges = getattr(self, '_current_duration_ranges', None)
            start_time_limit = getattr(self, '_start_time_seconds', None)
            end_time_limit = getattr(self, '_end_time_seconds', None)
            
            # Calculate strict bounds from duration_ranges if provided
            if duration_ranges and len(duration_ranges) > 0:
                range_config = {
                    "short": {"min": 15, "max": 30},
                    "medium": {"min": 30, "max": 60},
                    "long": {"min": 60, "max": 120},
                }
                strict_min = float('inf')
                strict_max = 0
                for r in duration_ranges:
                    if r in range_config:
                        strict_min = min(strict_min, range_config[r]["min"])
                        strict_max = max(strict_max, range_config[r]["max"])
                if strict_min != float('inf') and strict_max > 0:
                    min_duration = strict_min
                    max_duration = strict_max
                    logger.info(f"Duration range enforcement: {duration_ranges} -> {min_duration}-{max_duration}s strict bounds")
            
            # Pre-process clips (filter invalid ones and enforce duration bounds)
            valid_clips_data = []
            for clip in clips_data:
                start = clip.get("start_time", clip.get("startTime", clip.get("start", 0)))
                end = clip.get("end_time", clip.get("endTime", clip.get("end", 0)))
                
                # Convert to seconds if needed (check if values are too large for seconds)
                if start > 100000:  # Likely milliseconds
                    start = start / 1000
                if end > 100000:
                    end = end / 1000
                
                duration = end - start
                
                # Skip clips that are way too short (less than 5 seconds)
                if duration < 5:
                    logger.warning(f"Filtering clip with duration {duration}s - too short (< 5s)")
                    continue
                
                # Validate clip is within time range if specified
                if start_time_limit is not None and start < start_time_limit:
                    logger.warning(
                        f"Adjusting clip start from {start}s to {start_time_limit}s (before selected range)"
                    )
                    start = start_time_limit
                    duration = end - start
                
                if end_time_limit is not None and end > end_time_limit:
                    logger.warning(
                        f"Adjusting clip end from {end}s to {end_time_limit}s (after selected range)"
                    )
                    end = end_time_limit
                    duration = end - start
                
                # Enforce duration bounds with adjustment
                original_duration = duration
                adjusted = False
                
                # If clip is too short, try to extend it
                if duration < min_duration:
                    extension_needed = min_duration - duration
                    # Try to extend end time
                    new_end = end + extension_needed
                    # Respect end time limit if set
                    if end_time_limit is not None and new_end > end_time_limit:
                        new_end = end_time_limit
                    # Check if extension is sufficient
                    if new_end - start >= min_duration:
                        logger.info(
                            f"Extended short clip ({original_duration:.1f}s -> {new_end - start:.1f}s) "
                            f"to meet minimum duration {min_duration}s"
                        )
                        end = new_end
                        duration = end - start
                        adjusted = True
                    else:
                        logger.warning(
                            f"Filtering clip ({original_duration:.1f}s) - too short and cannot extend "
                            f"to minimum {min_duration}s"
                        )
                        continue
                
                # If clip is too long, trim it
                if duration > max_duration:
                    logger.info(
                        f"Trimmed long clip ({original_duration:.1f}s -> {max_duration}s) "
                        f"to meet maximum duration {max_duration}s"
                    )
                    end = start + max_duration
                    duration = max_duration
                    adjusted = True
                
                valid_clips_data.append({
                    **clip,
                    "start_time": start,
                    "end_time": end,
                })
                if adjusted:
                    logger.info(f"Adjusted clip: {start:.1f}s - {end:.1f}s (duration: {duration:.1f}s)")
                else:
                    logger.info(f"Valid clip found: {start:.1f}s - {end:.1f}s (duration: {duration:.1f}s)")
            
            # Build ClipPlanSegment objects using the user-selected layout type
            layout_type = getattr(self, '_current_layout_type', 'screen_share')
            transcript = getattr(self, '_current_transcript', [])
            max_duration = getattr(self, '_current_max_duration', 90)
            
            clips = []
            for clip in valid_clips_data:
                start_sec = clip.get("start_time", 0)
                end_sec = clip.get("end_time", 0)
                
                start_time_ms = int(start_sec * 1000)
                end_time_ms = int(end_sec * 1000)
                
                # Apply sentence/word boundary snapping to prevent cutting off mid-word
                if self.settings.sentence_snapping_enabled and transcript:
                    max_extension_ms = int(self.settings.sentence_extension_max_seconds * 1000)
                    
                    # 1. Snap START time to word/sentence boundary (prevent cutting mid-word)
                    original_start_ms = start_time_ms
                    adjusted_start_ms = find_sentence_start_boundary(
                        segments=transcript,
                        timestamp_ms=start_time_ms,
                        max_adjustment_ms=3000,  # Look up to 3 seconds back for clean start
                    )
                    
                    # Ensure we don't go negative
                    if adjusted_start_ms < 0:
                        adjusted_start_ms = 0
                    
                    if adjusted_start_ms != original_start_ms:
                        logger.info(
                            f"Clip start time adjusted for word boundary: "
                            f"{original_start_ms}ms -> {adjusted_start_ms}ms "
                            f"({adjusted_start_ms - original_start_ms:+d}ms)"
                        )
                        start_time_ms = adjusted_start_ms
                    
                    # 2. Snap END time to sentence boundary (prevent cutting mid-sentence)
                    original_end_ms = end_time_ms
                    adjusted_end_ms = find_sentence_end_boundary(
                        segments=transcript,
                        timestamp_ms=end_time_ms,
                        max_extension_ms=max_extension_ms,
                        search_direction="forward",
                    )
                    
                    # Ensure we don't exceed max clip duration
                    max_end_ms = start_time_ms + (max_duration * 1000)
                    if adjusted_end_ms > max_end_ms:
                        logger.debug(
                            f"Sentence boundary at {adjusted_end_ms}ms would exceed max duration, "
                            f"capping at {max_end_ms}ms"
                        )
                        adjusted_end_ms = max_end_ms
                    
                    if adjusted_end_ms != original_end_ms:
                        logger.info(
                            f"Clip end time adjusted for sentence boundary: "
                            f"{original_end_ms}ms -> {adjusted_end_ms}ms "
                            f"(+{adjusted_end_ms - original_end_ms}ms)"
                        )
                        end_time_ms = adjusted_end_ms

                clips.append(ClipPlanSegment(
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    virality_score=float(clip.get("virality_score", 0.5)),
                    layout_type=layout_type,
                    summary=clip.get("summary"),
                    tags=clip.get("tags", []),
                ))
            
            logger.info(f"Parsed {len(clips)} clips from Gemini response")
            
            return ClipPlanResponse(
                segments=clips,
                total_clips=len(clips),
                target_platform=getattr(self, '_current_target_platform', 'tiktok'),
                insights=insights,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse clip plan response: {e}")
            raise IntelligencePlanningError(f"Failed to parse clip plan: {e}")

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


class IntelligencePlanningError(Exception):
    """Exception raised when intelligence planning fails."""
    pass

