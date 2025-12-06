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

CRITICAL DISTINCTION - READ CAREFULLY:

**TALKING HEAD** = A person's face/body is the MAIN SUBJECT of the frame:
- Person takes up >30% of the frame
- Person is centered or near-center
- Background may show room, whiteboard, wall, office - but person is the FOCUS
- NO screen recording, NO slides, NO code, NO browser visible
- Face is FULL SIZE, not a tiny overlay

**SCREEN SHARE WITH WEBCAM** = Screen content is primary, with a SMALL person overlay:
- Main content is: code, browser, slides, pricing tables, software UI, documents
- There is a TINY webcam overlay (10-25% of screen) in a CORNER
- The webcam shows just a face in a small box, NOT the main subject
- Person is clearly IN A CORNER (top-left, top-right, bottom-left, bottom-right)

**SCREEN SHARE WITHOUT WEBCAM** = Only screen content, no person visible:
- Just code, browser, slides, documents - no face at all

KEY RULE: If the person's face/body is the MAIN FOCUS of the frame (not in a corner), 
that is ALWAYS "talking_head" even if there's a whiteboard behind them!

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
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration_ms = end_ms - start_ms
            duration_sec = duration_ms / 1000
            
            # AUTO-CALCULATE sample count based on duration
            # More samples for longer clips to catch transitions
            if sample_count is None:
                if duration_sec <= 15:
                    sample_count = 3  # Short clips: 3 frames
                elif duration_sec <= 30:
                    sample_count = 5  # Medium clips: 5 frames (every ~6 seconds)
                else:
                    sample_count = 7  # Long clips: 7 frames (every ~5-8 seconds)
            
            logger.info(f"VLM sampling {sample_count} frames for {duration_sec:.1f}s clip")
            
            # Sample frames INCLUDING start and end to catch boundary transitions
            # First sample: 0.5s into clip (to avoid black/transition frames)
            # Last sample: 0.5s before end
            # Middle samples: evenly distributed
            if sample_count == 1:
                sample_times = [start_ms + duration_ms // 2]
            else:
                # First sample at ~500ms into clip, last sample at ~500ms before end
                margin_ms = min(500, duration_ms // (sample_count * 2))
                usable_duration = duration_ms - (2 * margin_ms)
                sample_times = [
                    start_ms + margin_ms + int(usable_duration * i / (sample_count - 1))
                    for i in range(sample_count)
                ]
            
            frame_results: list[VLMFrameResult] = []
            vlm_results = []
            
            for sample_time in sample_times:
                frame_pos = int((sample_time / 1000) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                result = await self.analyze_frame(frame, frame_width, frame_height)
                vlm_results.append(result)
                
                # Store per-frame result for transition detection
                frame_result = VLMFrameResult(
                    timestamp_ms=sample_time,
                    has_corner_webcam=result.has_corner_webcam,
                    webcam_position=result.webcam_position,
                    layout_type=result.layout_type,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                )
                frame_results.append(frame_result)
                
                logger.info(
                    f"VLM frame {sample_time}ms: webcam={result.has_corner_webcam}, "
                    f"layout={result.layout_type}, confidence={result.confidence:.2f}"
                )
            
            if not vlm_results:
                return VLMLayoutResult(
                    has_corner_webcam=False,
                    webcam_position=None,
                    layout_type="talking_head",
                    confidence=0.3,
                    reasoning="No frames could be analyzed",
                )
            
            # DETECT TRANSITIONS: Check if layout changes between frames
            # A transition occurs when we go from webcam -> no webcam or vice versa
            layouts_by_frame = [
                "screen_share" if r.has_corner_webcam else r.layout_type 
                for r in vlm_results
            ]
            unique_layouts = set(layouts_by_frame)
            has_transitions = len(unique_layouts) > 1
            
            if has_transitions:
                logger.info(
                    f"TRANSITION DETECTED: {len(unique_layouts)} different layouts found: {unique_layouts}"
                )
            
            # Count webcam frames
            webcam_votes = sum(1 for r in vlm_results if r.has_corner_webcam)
            
            # Get most common position from frames that detected webcam
            positions = [r.webcam_position for r in vlm_results if r.webcam_position]
            if positions:
                from collections import Counter
                webcam_position = Counter(positions).most_common(1)[0][0]
            else:
                webcam_position = None
            
            # Average confidence
            avg_confidence = sum(r.confidence for r in vlm_results) / len(vlm_results)
            
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
                f"VLM clip analysis: {len(vlm_results)} frames, webcam={webcam_votes > 0} ({webcam_votes}/{len(vlm_results)}), "
                f"position={webcam_position}, dominant_layout={dominant_layout}, "
                f"has_transitions={has_transitions}, confidence={avg_confidence:.2f}"
            )
            
            return VLMLayoutResult(
                has_corner_webcam=webcam_votes > 0,
                webcam_position=webcam_position,
                layout_type=dominant_layout,
                confidence=avg_confidence,
                reasoning=f"Analyzed {len(vlm_results)} frames: {webcam_votes}/{len(vlm_results)} detected webcam, transitions={has_transitions}",
                webcam_bbox_estimate=webcam_bbox,
                frame_results=frame_results,
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
