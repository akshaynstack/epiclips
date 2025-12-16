"""
VLM-based Layout Detector - Uses Vision Language Models for accurate layout detection.

This service uses lightweight VLMs to analyze video frames and detect:
1. Whether there's a corner facecam/webcam overlay
2. The position of the facecam (top-left, top-right, bottom-left, bottom-right)
3. Whether the video is screen content or talking head

This is more reliable than pure face detection for complex scenarios like:
- Small webcam overlays that face detectors miss
- Webcam overlays with multiple faces on screen
- Screen recordings with face content in the main area

MODELS USED:
- Primary: google/gemini-2.0-flash-001 (fast, cheap, good vision)
- This runs via OpenRouter API, NOT locally
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VLMFrameResult:
    """Result from VLM analysis of a single frame."""
    timestamp_ms: int
    has_corner_webcam: bool
    webcam_position: Optional[str]
    layout_type: str  # "screen_share" or "talking_head"
    confidence: float
    reasoning: str = ""


@dataclass
class VLMLayoutResult:
    """Result from VLM layout analysis."""
    has_corner_webcam: bool
    webcam_position: Optional[str]  # "top-left", "top-right", "bottom-left", "bottom-right"
    layout_type: str  # "screen_share" or "talking_head"
    confidence: float
    reasoning: str
    webcam_bbox_estimate: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    
    # NEW: Per-frame results for transition detection
    frame_results: list[VLMFrameResult] = field(default_factory=list)
    has_transitions: bool = False  # True if different layouts detected across frames


class VLMLayoutDetector:
    """
    Detects video layout using Vision Language Models.
    
    Uses Gemini Flash vision for fast, accurate layout analysis.
    Falls back to pure heuristics if VLM is unavailable.
    
    API: Uses OpenRouter to call Gemini Flash (NOT running locally)
    Cost: ~$0.0001 per image (very cheap)
    Speed: ~1-2 seconds per image
    """
    
    def __init__(self):
        settings = get_settings()
        self.openrouter_api_key = settings.openrouter_api_key
        self._client: Optional[httpx.AsyncClient] = None
        
        if not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not configured - VLM detection will use fallback")
        else:
            logger.info("VLM Layout Detector initialized with OpenRouter API")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _encode_frame_to_base64(self, frame: np.ndarray, max_size: int = 512) -> str:
        """
        Encode a frame to base64 for VLM input.
        
        Resizes to max_size to reduce token usage while maintaining aspect ratio.
        """
        height, width = frame.shape[:2]
        
        # Resize to max dimension while maintaining aspect ratio
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    async def analyze_frame(
        self,
        frame: np.ndarray,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> VLMLayoutResult:
        """
        Analyze a single frame to detect layout type using VLM.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            frame_width: Original frame width for bbox calculation
            frame_height: Original frame height for bbox calculation
            
        Returns:
            VLMLayoutResult with layout detection results
        """
        if not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set, using fallback detection")
            return self._fallback_detection(frame, frame_width, frame_height)
        
        try:
            return await self._analyze_with_gemini_flash(frame, frame_width, frame_height)
        except Exception as e:
            logger.warning(f"VLM analysis failed: {e}, using fallback")
            return self._fallback_detection(frame, frame_width, frame_height)
    
    async def _analyze_with_gemini_flash(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> VLMLayoutResult:
        """Use Gemini Flash vision for layout detection."""
        
        # Encode frame to base64
        frame_b64 = self._encode_frame_to_base64(frame, max_size=768)  # Higher res for accuracy
        
        # Prompt for layout detection - optimized for accuracy
        prompt = """Analyze this video screenshot to determine the layout type.

CRITICAL RULES - FOLLOW EXACTLY:

1. **SCREEN CONTENT CHECK** - Look for code, browser, slides, applications, documents:
   - If you see ANY code editor, browser tabs, application UI, terminal, or document → "screen_recording"
   - Even if there's a person visible, if screen content exists → "screen_recording"

2. **WEBCAM OVERLAY DETECTION**:
   - Small face in a CORNER (top-left, top-right, bottom-left, bottom-right) → has_webcam_overlay: true
   - Face can be 10-30% of screen - as long as it's in a CORNER with screen content visible
   - Specify exact corner position: "bottom-right", "bottom-left", "top-right", "top-left"

3. **TALKING HEAD** - ONLY if ALL these are true:
   - NO code, NO browser, NO application UI, NO slides visible
   - Person is the ONLY content (just face/body and room background)
   - Main background is: plain wall, room, office, whiteboard - NOT a computer screen
   - Person is CENTERED or takes up majority of frame

DECISION TREE:
- See code/browser/apps/slides? → "screen_recording" (check for corner webcam)
- ONLY person + room/wall/whiteboard? → "talking_head"

Respond with ONLY this JSON:
{"has_webcam_overlay": boolean, "webcam_position": "bottom-right"|"bottom-left"|"top-right"|"top-left"|null, "main_content": "talking_head"|"screen_recording", "confidence": 0.9, "reasoning": "brief explanation"}"""

        client = await self._get_client()
        
        logger.info("Calling Gemini Flash Vision API for layout detection...")
        
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://viewcreator.ai",  # Required by OpenRouter
            },
            json={
                "model": "google/gemini-2.0-flash-001",  # Fast, cheap, excellent vision
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.0,  # Deterministic
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Parse response
        content = result["choices"][0]["message"]["content"]
        logger.info(f"VLM response: {content[:200]}...")
        
        # Extract JSON from response
        
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if not json_match:
            logger.warning(f"Could not parse VLM response: {content}")
            return self._fallback_detection(frame, frame_width, frame_height)
        
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in VLM response: {content}")
            return self._fallback_detection(frame, frame_width, frame_height)
        
        has_webcam = parsed.get("has_webcam_overlay", False)
        webcam_pos = parsed.get("webcam_position")
        main_content = parsed.get("main_content", "talking_head")
        confidence = parsed.get("confidence", 0.8)
        reasoning = parsed.get("reasoning", "")
        
        # Determine layout type
        if has_webcam and webcam_pos:
            layout_type = "screen_share"
            # Estimate webcam bbox based on position
            webcam_bbox = self._estimate_webcam_bbox(webcam_pos, frame_width, frame_height)
        else:
            layout_type = "talking_head" if main_content == "talking_head" else "screen_share"
            webcam_bbox = None
        
        logger.info(
            f"VLM layout detection: has_webcam={has_webcam}, position={webcam_pos}, "
            f"layout={layout_type}, confidence={confidence:.2f}"
        )
        
        return VLMLayoutResult(
            has_corner_webcam=has_webcam,
            webcam_position=webcam_pos,
            layout_type=layout_type,
            confidence=confidence,
            reasoning=reasoning,
            webcam_bbox_estimate=webcam_bbox,
        )
    
    def _estimate_webcam_bbox(
        self,
        position: str,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Estimate webcam bounding box based on corner position.
        
        Returns typical webcam size/position for split-screen rendering.
        """
        # Typical webcam overlay is about 15-20% of frame width
        webcam_width = int(frame_width * 0.18)
        webcam_height = int(frame_height * 0.18)
        
        # Add small margin from edges
        margin = 20
        
        if position == "top-left":
            x = margin
            y = margin
        elif position == "top-right":
            x = frame_width - webcam_width - margin
            y = margin
        elif position == "bottom-left":
            x = margin
            y = frame_height - webcam_height - margin
        elif position == "bottom-right":
            x = frame_width - webcam_width - margin
            y = frame_height - webcam_height - margin
        else:
            # Default to bottom-right
            x = frame_width - webcam_width - margin
            y = frame_height - webcam_height - margin
        
        return (x, y, webcam_width, webcam_height)
    
    def _fallback_detection(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> VLMLayoutResult:
        """
        Fallback detection using edge density heuristics.
        
        If VLM is unavailable, use simple heuristics to guess layout.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions
        h, w = edges.shape
        
        # Check corners for potential webcam (high edge density in small region)
        corner_size = int(min(w, h) * 0.25)
        corners = {
            "bottom-right": edges[h-corner_size:, w-corner_size:],
            "bottom-left": edges[h-corner_size:, :corner_size],
            "top-right": edges[:corner_size, w-corner_size:],
            "top-left": edges[:corner_size, :corner_size],
        }
        
        # Check main area edge density (indicates screen content)
        main_area = edges[corner_size:h-corner_size, corner_size:w-corner_size]
        main_density = np.mean(main_area) / 255
        
        # High main area density suggests screen content
        has_screen_content = main_density > 0.03
        
        # Check for potential webcam in corners
        corner_densities = {pos: np.mean(c) / 255 for pos, c in corners.items()}
        max_corner = max(corner_densities.items(), key=lambda x: x[1])
        
        # A corner with medium density might have a face
        # (faces have moderate edge density, not as high as text/UI)
        has_potential_webcam = 0.02 < max_corner[1] < 0.15
        
        if has_screen_content and has_potential_webcam:
            return VLMLayoutResult(
                has_corner_webcam=True,
                webcam_position=max_corner[0],
                layout_type="screen_share",
                confidence=0.5,
                reasoning="Fallback: edge density suggests screen content with corner webcam",
                webcam_bbox_estimate=self._estimate_webcam_bbox(max_corner[0], frame_width, frame_height),
            )
        elif has_screen_content:
            return VLMLayoutResult(
                has_corner_webcam=False,
                webcam_position=None,
                layout_type="screen_share",
                confidence=0.5,
                reasoning="Fallback: edge density suggests screen content without webcam",
            )
        else:
            return VLMLayoutResult(
                has_corner_webcam=False,
                webcam_position=None,
                layout_type="talking_head",
                confidence=0.5,
                reasoning="Fallback: low edge density suggests talking head",
            )
    
    async def analyze_clip_layout(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        sample_count: int = None,  # Auto-calculated based on duration
    ) -> VLMLayoutResult:
        """
        Analyze a video clip to determine its layout type and detect transitions.
        
        Samples multiple frames to detect layout transitions within a clip.
        For clips > 20 seconds, samples more frames to catch transitions.
        
        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            sample_count: Number of frames to sample (auto-calculated if None)
            
        Returns:
            VLMLayoutResult with aggregated results AND per-frame results for transition detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return VLMLayoutResult(
                has_corner_webcam=False,
                webcam_position=None,
                layout_type="talking_head",
                confidence=0.3,
                reasoning="Failed to open video",
            )
        
        try:
            analysis_start = time.monotonic()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration_ms = end_ms - start_ms
            duration_sec = duration_ms / 1000
            
            if sample_count is None:
                settings = get_settings()

                # VLM calls are network-bound; sampling at 1 FPS causes dozens of API calls per clip.
                # For typical "talking_head" vs "screen_share" detection + webcam position, a small,
                # evenly-spaced sample set is sufficient and dramatically faster.
                #
                # Fargate mode defaults to a smaller cap to keep latency predictable.
                max_samples = 5 if settings.fargate_mode else 12
                target_interval_sec = 10.0 if settings.fargate_mode else 5.0

                estimated = int(duration_sec / target_interval_sec) + 2  # include start/end
                sample_count = max(3, min(estimated, max_samples))
            
            logger.info(f"VLM sampling: {sample_count} frames for {duration_sec:.1f}s clip")
            
            # Sample evenly across the clip (avoid exact start/end to reduce black/transition frames)
            if duration_ms <= 0:
                sample_times = [start_ms]
            else:
                margin_ms = min(500, max(0, duration_ms // 10))
                sample_start = start_ms + margin_ms
                sample_end = end_ms - margin_ms

                if sample_end <= sample_start:
                    sample_times = [start_ms + duration_ms // 2]
                elif sample_count <= 1:
                    sample_times = [start_ms + duration_ms // 2]
                else:
                    step = (sample_end - sample_start) / (sample_count - 1)
                    sample_times = [int(sample_start + (step * i)) for i in range(sample_count)]

            # De-duplicate and clamp just in case
            sample_times = sorted({t for t in sample_times if start_ms <= t < end_ms})
            
            results_by_timestamp: dict[int, VLMLayoutResult] = {}
            frame_results_by_timestamp: dict[int, VLMFrameResult] = {}

            def _layout_label_from_result(result: VLMLayoutResult) -> str:
                return "screen_share" if result.has_corner_webcam else result.layout_type

            def _read_frame_at_timestamp(timestamp_ms: int) -> Optional[np.ndarray]:
                frame_pos = int((timestamp_ms / 1000) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret:
                    return None
                return frame

            def _heuristic_label_at_timestamp(timestamp_ms: int) -> Optional[str]:
                frame = _read_frame_at_timestamp(timestamp_ms)
                if frame is None:
                    return None
                fallback = self._fallback_detection(frame, frame_width, frame_height)
                return _layout_label_from_result(fallback)

            async def _analyze_vlm_at_timestamps(
                timestamps_ms: list[int],
            ) -> dict[int, VLMLayoutResult]:
                missing = [
                    t for t in timestamps_ms
                    if t not in results_by_timestamp and start_ms <= t < end_ms
                ]
                frames_to_analyze: list[tuple[int, np.ndarray]] = []
                for t in missing:
                    frame = _read_frame_at_timestamp(t)
                    if frame is not None:
                        frames_to_analyze.append((t, frame))

                if frames_to_analyze:
                    analyses = await asyncio.gather(
                        *[
                            self.analyze_frame(frame, frame_width, frame_height)
                            for _, frame in frames_to_analyze
                        ],
                        return_exceptions=True,
                    )
                    for (t, _frame), analysis in zip(frames_to_analyze, analyses):
                        if isinstance(analysis, Exception):
                            continue
                        results_by_timestamp[t] = analysis
                        frame_results_by_timestamp[t] = VLMFrameResult(
                            timestamp_ms=t,
                            has_corner_webcam=analysis.has_corner_webcam,
                            webcam_position=analysis.webcam_position,
                            layout_type=analysis.layout_type,
                            confidence=analysis.confidence,
                            reasoning=analysis.reasoning,
                        )

                return {t: results_by_timestamp[t] for t in timestamps_ms if t in results_by_timestamp}
            
            for sample_time in sample_times:
                frame = _read_frame_at_timestamp(sample_time)
                if frame is None:
                    continue
                
                result = await self.analyze_frame(frame, frame_width, frame_height)
                results_by_timestamp[sample_time] = result
                frame_results_by_timestamp[sample_time] = VLMFrameResult(
                    timestamp_ms=sample_time,
                    has_corner_webcam=result.has_corner_webcam,
                    webcam_position=result.webcam_position,
                    layout_type=result.layout_type,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                )
                
                logger.info(
                    f"VLM frame {sample_time}ms: webcam={result.has_corner_webcam}, "
                    f"layout={result.layout_type}, confidence={result.confidence:.2f}"
                )

            if not results_by_timestamp:
                return VLMLayoutResult(
                    has_corner_webcam=False,
                    webcam_position=None,
                    layout_type="talking_head",
                    confidence=0.3,
                    reasoning="No frames could be analyzed",
                )

            settings = get_settings()
            target_precision_ms = 250 if settings.fargate_mode else 200
            max_windows_to_refine = 2 if settings.fargate_mode else 4
            max_heuristic_steps = 10
            max_extra_vlm_frames = 6 if settings.fargate_mode else 12
            extra_vlm_frames_added = 0

            # ------------------------------------------------------------
            # Detect transitions from coarse samples
            # ------------------------------------------------------------
            sorted_timestamps = sorted(results_by_timestamp.keys())
            sorted_results = [results_by_timestamp[t] for t in sorted_timestamps]
            labels_by_frame = [_layout_label_from_result(r) for r in sorted_results]
            unique_layouts = set(labels_by_frame)
            has_transitions = len(unique_layouts) > 1

            # ------------------------------------------------------------
            # Adaptive refinement: only when transitions are suspected.
            # Use cheap local heuristics to bracket the boundary, then add a
            # couple VLM frames around the boundary for precise segmentation.
            # ------------------------------------------------------------
            if has_transitions:
                windows = []
                for idx in range(1, len(sorted_timestamps)):
                    left_ts = sorted_timestamps[idx - 1]
                    right_ts = sorted_timestamps[idx]
                    left_label = labels_by_frame[idx - 1]
                    right_label = labels_by_frame[idx]
                    if left_label == right_label:
                        continue

                    # Skip low-confidence edges; they tend to be noise.
                    left_conf = results_by_timestamp[left_ts].confidence
                    right_conf = results_by_timestamp[right_ts].confidence
                    if left_conf < 0.7 or right_conf < 0.7:
                        continue

                    windows.append((left_ts, right_ts, left_label, right_label))

                windows = windows[:max_windows_to_refine]

                if windows:
                    logger.info(
                        f"VLM transitions detected; refining {len(windows)} window(s) to ~{target_precision_ms}ms precision"
                    )

                for left_ts, right_ts, left_label, right_label in windows:
                    if extra_vlm_frames_added >= max_extra_vlm_frames:
                        break

                    lo = left_ts
                    hi = right_ts

                    # Heuristic bisection (no network).
                    for _ in range(max_heuristic_steps):
                        if hi - lo <= target_precision_ms:
                            break

                        mid = int((lo + hi) / 2)
                        mid_label = _heuristic_label_at_timestamp(mid)
                        if mid_label is None:
                            break
                        if mid_label == left_label:
                            lo = mid
                        elif mid_label == right_label:
                            hi = mid
                        else:
                            break

                    # Add VLM samples near the refined boundary (bounded network calls).
                    # Use the bracket endpoints so the segmenter can interpolate.
                    lo_missing = lo not in results_by_timestamp
                    hi_missing = hi not in results_by_timestamp
                    needed = int(lo_missing) + int(hi_missing)
                    if extra_vlm_frames_added + needed > max_extra_vlm_frames:
                        break
                    await _analyze_vlm_at_timestamps([lo, hi])
                    if lo_missing and lo in results_by_timestamp:
                        extra_vlm_frames_added += 1
                    if hi_missing and hi in results_by_timestamp:
                        extra_vlm_frames_added += 1

                    # If heuristic bracketing didn't actually bracket in VLM space,
                    # fall back to a small VLM bisection to find a real boundary.
                    lo_label_vlm = _layout_label_from_result(results_by_timestamp[lo])
                    hi_label_vlm = _layout_label_from_result(results_by_timestamp[hi])

                    if lo_label_vlm == hi_label_vlm:
                        lo = left_ts
                        hi = right_ts
                        lo_label_vlm = left_label
                        hi_label_vlm = right_label

                    max_vlm_steps = 4
                    for _ in range(max_vlm_steps):
                        if extra_vlm_frames_added >= max_extra_vlm_frames:
                            break
                        if hi - lo <= target_precision_ms:
                            break

                        mid = int((lo + hi) / 2)
                        if mid in results_by_timestamp:
                            mid_label_vlm = _layout_label_from_result(results_by_timestamp[mid])
                        else:
                            await _analyze_vlm_at_timestamps([mid])
                            if mid not in results_by_timestamp:
                                break
                            extra_vlm_frames_added += 1
                            mid_label_vlm = _layout_label_from_result(results_by_timestamp[mid])

                        if mid_label_vlm == lo_label_vlm:
                            lo = mid
                            lo_label_vlm = mid_label_vlm
                        else:
                            hi = mid
                            hi_label_vlm = mid_label_vlm

                # Recompute with refined samples.
                sorted_timestamps = sorted(results_by_timestamp.keys())
                sorted_results = [results_by_timestamp[t] for t in sorted_timestamps]
                labels_by_frame = [_layout_label_from_result(r) for r in sorted_results]
                unique_layouts = set(labels_by_frame)
                has_transitions = len(unique_layouts) > 1
                if has_transitions:
                    logger.info(
                        f"VLM refinement complete: frames={len(sorted_results)}, layouts={unique_layouts}, added={extra_vlm_frames_added}"
                    )
            
            # Count webcam frames
            webcam_votes = sum(1 for r in sorted_results if r.has_corner_webcam)
            
            # Get most common position from frames that detected webcam
            positions = [r.webcam_position for r in sorted_results if r.webcam_position]
            if positions:
                from collections import Counter
                webcam_position = Counter(positions).most_common(1)[0][0]
            else:
                webcam_position = None
            
            # Average confidence
            avg_confidence = sum(r.confidence for r in sorted_results) / len(sorted_results)
            
            # Determine DOMINANT layout (for the whole clip when not using transitions)
            # If transitions exist, each segment will use its own layout
            if webcam_votes > 0:
                # If ANY frame has webcam, dominant is screen_share
                # But if there are transitions, individual segments may differ
                dominant_layout = "screen_share"
                webcam_bbox = self._estimate_webcam_bbox(webcam_position, frame_width, frame_height)
            else:
                # No webcam in any frame - check majority
                screen_share_votes = sum(1 for r in vlm_results if r.layout_type == "screen_share")
                dominant_layout = "screen_share" if screen_share_votes > len(vlm_results) / 2 else "talking_head"
                webcam_bbox = None
            
            logger.info(
                f"VLM clip analysis: {len(sorted_results)} frames, webcam={webcam_votes > 0} ({webcam_votes}/{len(sorted_results)}), "
                f"position={webcam_position}, dominant_layout={dominant_layout}, "
                f"has_transitions={has_transitions}, confidence={avg_confidence:.2f}, "
                f"time={time.monotonic() - analysis_start:.1f}s"
            )
            
            return VLMLayoutResult(
                has_corner_webcam=webcam_votes > 0,
                webcam_position=webcam_position,
                layout_type=dominant_layout,
                confidence=avg_confidence,
                reasoning=f"Analyzed {len(sorted_results)} frames: {webcam_votes}/{len(sorted_results)} detected webcam, transitions={has_transitions}",
                webcam_bbox_estimate=webcam_bbox,
                frame_results=[frame_results_by_timestamp[t] for t in sorted_timestamps if t in frame_results_by_timestamp],
                has_transitions=has_transitions,
            )
            
        finally:
            cap.release()


# Singleton instance
_vlm_detector: Optional[VLMLayoutDetector] = None


def get_vlm_detector() -> VLMLayoutDetector:
    """Get or create the VLM layout detector singleton."""
    global _vlm_detector
    if _vlm_detector is None:
        _vlm_detector = VLMLayoutDetector()
    return _vlm_detector
