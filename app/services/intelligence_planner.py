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
from app.services.transcription_service import TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


# Layout types for clips
ClipLayoutType = Literal["talking_head", "screen_share"]


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
    """

    def __init__(self):
        self.settings = get_settings()
        self._http_client: Optional[httpx.AsyncClient] = None
        
        if not self.settings.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set, intelligence planning will fail")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.openrouter_base_url,
                timeout=httpx.Timeout(300.0, connect=30.0),
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "HTTP-Referer": "https://viewcreator.com",
                    "X-Title": "ViewCreator AI Clipping Agent",
                },
            )
        return self._http_client

    async def plan_clips(
        self,
        transcript_result: TranscriptionResult,
        video_metadata: Any = None,
        max_clips: int = 5,
        min_duration_seconds: int = 15,
        max_duration_seconds: int = 90,
        duration_ranges: Optional[list[str]] = None,
        target_platform: str = "tiktok",
        frames: Optional[list[VisionFrame]] = None,
    ) -> ClipPlanResponse:
        """
        Plan viral clips from video content.
        
        Args:
            transcript_result: TranscriptionResult from transcription service
            video_metadata: Optional video metadata
            max_clips: Maximum number of clips to generate
            min_duration_seconds: Minimum clip duration
            max_duration_seconds: Maximum clip duration
            duration_ranges: Optional list of selected duration ranges ('short', 'medium', 'long')
            target_platform: Target platform (tiktok, youtube_shorts, instagram_reels)
            frames: Optional sampled video frames for vision analysis
            
        Returns:
            ClipPlanResponse with identified clips
        """
        transcript = transcript_result.segments if transcript_result else []
        frames = frames or []
        
        if not transcript and not frames:
            logger.warning("No transcript or frames provided for clip planning")
            return ClipPlanResponse(segments=[], total_clips=0, target_platform=target_platform, insights="No content provided for analysis")
        
        clip_count = max_clips or self.settings.max_suggested_clips
        
        logger.info(
            f"Planning clips: {len(transcript)} transcript segments, "
            f"{len(frames)} frames, requesting {clip_count} clips, "
            f"duration_ranges={duration_ranges}"
        )
        
        # Store for use in response
        self._current_target_platform = target_platform
        self._current_min_duration = min_duration_seconds
        self._current_max_duration = max_duration_seconds
        self._current_duration_ranges = duration_ranges
        
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
        
        # Call Gemini via OpenRouter
        logger.info(f"Calling Gemini ({self.settings.gemini_model}) for clip planning...")
        
        response = await self._call_openrouter(
            model=self.settings.gemini_model,
            messages=messages,
            temperature=0.2,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )
        
        # Parse response
        return self._parse_clip_plan_response(response)

    def _build_system_prompt(
        self,
        clip_count: int,
        min_duration: int = 15,
        max_duration: int = 90,
        duration_ranges: Optional[list[str]] = None,
    ) -> str:
        """Build the system prompt for Gemini."""
        # Build duration guidance based on selected ranges
        duration_guidance = ""
        if duration_ranges and len(duration_ranges) > 0:
            range_descriptions = {
                "short": "15-30 seconds (quick, punchy clips)",
                "medium": "30-60 seconds (moderate length clips)",
                "long": "60-120 seconds (longer, more in-depth clips)",
            }
            selected_ranges = [range_descriptions.get(r, r) for r in duration_ranges if r in range_descriptions]
            if selected_ranges:
                duration_guidance = f"""
IMPORTANT: The user has selected the following clip length preferences:
{chr(10).join(f'- {r}' for r in selected_ranges)}

When possible, distribute clips across these selected length ranges. Prioritize clips that fit naturally within these durations while maintaining high virality potential."""
        
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

Rules:
- Each clip should be {min_duration}-{max_duration} seconds long for optimal short-form content
- Prefer clips with strong opening hooks (first 3 seconds are critical)
- layout_type: ALWAYS use "screen_share" for optimal vertical video format (split-screen with face close-up)
- Higher virality_score = higher confidence this clip will perform well
- Return times in SECONDS (not milliseconds)"""

    def _build_transcript_text(self, transcript: list) -> str:
        """Build formatted transcript text."""
        if not transcript:
            return "[No transcript available]"
        
        # Debug: log segment structure
        if transcript:
            first_seg = transcript[0]
            logger.info(f"First segment type: {type(first_seg)}, attrs: {dir(first_seg) if hasattr(first_seg, '__dict__') else 'N/A'}")
            if hasattr(first_seg, 'text'):
                logger.info(f"First segment text: '{first_seg.text[:100] if first_seg.text else 'EMPTY'}'")
            elif hasattr(first_seg, '__dict__'):
                logger.info(f"First segment dict: {first_seg.__dict__}")
        
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
            
            logger.info(f"Raw Gemini response content (first 1000 chars): {content[:1000]}")
            
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
            
            logger.info(f"Parsed response structure: {type(parsed).__name__}, keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
            
            # Handle array response (clips directly)
            if isinstance(parsed, list):
                clips_data = parsed
                insights = None
            else:
                clips_data = parsed.get("clips", [])
                insights = parsed.get("insights")
            
            logger.info(f"Found {len(clips_data)} clips in response before validation")
            
            # Pre-process clips (filter invalid ones)
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
                
                if duration >= 1:  # At least 1 second
                    valid_clips_data.append({
                        **clip,
                        "start_time": start,
                        "end_time": end,
                    })
                    logger.info(f"Valid clip found: {start}s - {end}s (duration: {duration}s)")
                else:
                    logger.warning(f"Filtering invalid clip with duration {duration}s (start={start}, end={end})")
            
            # Build ClipPlanSegment objects
            # NOTE: Always default to "screen_share" to match epiriumaiclips behavior
            # The OpusClip-style split-screen layout works best for most content
            clips = []
            for clip in valid_clips_data:
                start_sec = clip.get("start_time", 0)
                end_sec = clip.get("end_time", 0)

                clips.append(ClipPlanSegment(
                    start_time_ms=int(start_sec * 1000),
                    end_time_ms=int(end_sec * 1000),
                    virality_score=float(clip.get("virality_score", 0.5)),
                    layout_type="screen_share",  # Always use screen_share (epiriumaiclips style)
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

