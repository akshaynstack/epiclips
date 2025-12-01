"""
Content Region Detector - Intelligently detects screen content vs webcam regions.

This service analyzes video frames to identify:
1. Screen content regions (slides, code, UI, text)
2. Webcam overlay regions (typically small rectangle with face)
3. Main speaker/face position for smart cropping
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContentRegion:
    """A detected content region in the frame."""
    x: int
    y: int
    width: int
    height: int
    region_type: str  # "screen_content", "webcam_overlay", "face", "unknown"
    confidence: float


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_width: int
    frame_height: int
    primary_face_center: Optional[Tuple[int, int]]  # (x, y) center of primary face
    webcam_region: Optional[ContentRegion]  # Detected webcam overlay if any
    screen_content_center: Tuple[int, int]  # Best center point for screen content
    is_screen_share_layout: bool  # True if frame appears to be screen share


class ContentRegionDetector:
    """
    Detects and analyzes content regions in video frames for smart cropping.
    
    Uses multiple signals:
    - Face detection results
    - Edge density (high for UI/text, low for webcam backgrounds)
    - Color variance analysis
    - Common webcam overlay positions (corners)
    """

    def __init__(self):
        self._webcam_positions = [
            "bottom_right",
            "bottom_left", 
            "top_right",
            "top_left",
        ]

    def analyze_frame(
        self,
        frame: Optional[np.ndarray],
        face_detections: list[dict],
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> FrameAnalysis:
        """
        Analyze a frame to detect content regions.
        
        Args:
            frame: Video frame as numpy array (BGR), can be None if only using face data
            face_detections: List of face detection dicts with 'bbox' and 'confidence'
            frame_width: Frame width (used when frame is None)
            frame_height: Frame height (used when frame is None)
            
        Returns:
            FrameAnalysis with detected regions
        """
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            width, height = frame_width, frame_height
        
        # Find primary face
        primary_face_center = None
        if face_detections:
            # Use largest/most confident face
            best_face = max(face_detections, key=lambda f: f.get("confidence", 0))
            bbox = best_face.get("bbox", {})
            if bbox:
                center_x = int(bbox.get("x", 0) + bbox.get("width", 0) / 2)
                center_y = int(bbox.get("y", 0) + bbox.get("height", 0) / 2)
                primary_face_center = (center_x, center_y)
        
        # Detect webcam overlay region
        webcam_region = self._detect_webcam_overlay(frame, face_detections)
        
        # Determine screen content center (avoiding webcam region)
        screen_content_center = self._calculate_screen_content_center(
            width, height, webcam_region
        )
        
        # Determine if this looks like a screen share
        is_screen_share = self._is_screen_share_layout(
            frame, face_detections, webcam_region
        )
        
        return FrameAnalysis(
            frame_width=width,
            frame_height=height,
            primary_face_center=primary_face_center,
            webcam_region=webcam_region,
            screen_content_center=screen_content_center,
            is_screen_share_layout=is_screen_share,
        )

    def _detect_webcam_overlay(
        self,
        frame: np.ndarray,
        face_detections: list[dict],
    ) -> Optional[ContentRegion]:
        """
        Detect webcam overlay region in the frame.
        
        Webcam overlays are typically:
        - Small (< 30% of frame area)
        - In a corner
        - Contain a face
        - Have different color/texture than surrounding content
        """
        height, width = frame.shape[:2]
        
        if not face_detections:
            return None
        
        # Check if any face is in a corner region (typical webcam placement)
        corner_size = 0.35  # Consider corners as 35% from edges
        
        for face in face_detections:
            bbox = face.get("bbox", {})
            if not bbox:
                continue
            
            face_center_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
            face_center_y = bbox.get("y", 0) + bbox.get("height", 0) / 2
            face_area = bbox.get("width", 0) * bbox.get("height", 0)
            
            # Relative position
            rel_x = face_center_x / width
            rel_y = face_center_y / height
            
            # Check if face is in a corner
            in_corner = False
            corner_position = None
            
            if rel_x < corner_size and rel_y < corner_size:
                in_corner = True
                corner_position = "top_left"
            elif rel_x > (1 - corner_size) and rel_y < corner_size:
                in_corner = True
                corner_position = "top_right"
            elif rel_x < corner_size and rel_y > (1 - corner_size):
                in_corner = True
                corner_position = "bottom_left"
            elif rel_x > (1 - corner_size) and rel_y > (1 - corner_size):
                in_corner = True
                corner_position = "bottom_right"
            
            # Check if face is small relative to frame (typical webcam size)
            frame_area = width * height
            face_ratio = face_area / frame_area
            
            if in_corner and face_ratio < 0.15:  # Face is less than 15% of frame
                # This looks like a webcam overlay
                # Estimate webcam region as expanded face bbox
                margin = 1.5  # Expand bbox by 50%
                webcam_x = max(0, int(bbox.get("x", 0) - bbox.get("width", 0) * (margin - 1) / 2))
                webcam_y = max(0, int(bbox.get("y", 0) - bbox.get("height", 0) * (margin - 1) / 2))
                webcam_w = min(int(bbox.get("width", 0) * margin), width - webcam_x)
                webcam_h = min(int(bbox.get("height", 0) * margin), height - webcam_y)
                
                return ContentRegion(
                    x=webcam_x,
                    y=webcam_y,
                    width=webcam_w,
                    height=webcam_h,
                    region_type="webcam_overlay",
                    confidence=face.get("confidence", 0.5),
                )
        
        return None

    def _calculate_screen_content_center(
        self,
        width: int,
        height: int,
        webcam_region: Optional[ContentRegion],
    ) -> Tuple[int, int]:
        """
        Calculate the optimal center point for screen content.
        
        Avoids the webcam overlay region if detected.
        """
        if not webcam_region:
            # No webcam detected, use frame center
            return (width // 2, height // 2)
        
        # Calculate center of the non-webcam area
        webcam_center_x = webcam_region.x + webcam_region.width / 2
        webcam_center_y = webcam_region.y + webcam_region.height / 2
        
        # Move screen center away from webcam
        if webcam_center_x < width / 2:
            # Webcam is on left, center content to right
            screen_x = int(width * 0.55)
        else:
            # Webcam is on right, center content to left
            screen_x = int(width * 0.45)
        
        if webcam_center_y < height / 2:
            # Webcam is on top, center content lower
            screen_y = int(height * 0.55)
        else:
            # Webcam is on bottom, center content higher
            screen_y = int(height * 0.40)
        
        return (screen_x, screen_y)

    def _is_screen_share_layout(
        self,
        frame: Optional[np.ndarray],
        face_detections: list[dict],
        webcam_region: Optional[ContentRegion],
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> bool:
        """
        Determine if the frame appears to be a screen share layout.
        
        Screen share characteristics:
        - High edge density (lots of UI elements, text)
        - Face in corner (webcam overlay)
        - OR no face visible (pure screen share)
        """
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            width, height = frame_width, frame_height
        
        # If there's a detected webcam overlay, likely screen share
        if webcam_region:
            return True
        
        # If we have the actual frame, analyze edge density
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # High edge density suggests UI/text content (screen share)
            if edge_density > 0.08:
                if face_detections:
                    largest_face = max(
                        face_detections,
                        key=lambda f: f.get("bbox", {}).get("width", 0) * f.get("bbox", {}).get("height", 0)
                    )
                    bbox = largest_face.get("bbox", {})
                    face_area = bbox.get("width", 0) * bbox.get("height", 0)
                    frame_area = width * height
                    
                    if face_area / frame_area < 0.10:
                        return True
                else:
                    return True
        else:
            # Without the frame, use face position/size heuristics
            if face_detections:
                largest_face = max(
                    face_detections,
                    key=lambda f: f.get("bbox", {}).get("width", 0) * f.get("bbox", {}).get("height", 0)
                )
                bbox = largest_face.get("bbox", {})
                face_area = bbox.get("width", 0) * bbox.get("height", 0)
                frame_area = width * height
                
                # Small face in corner suggests screen share with webcam
                if frame_area > 0 and face_area / frame_area < 0.12:
                    face_center_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
                    face_center_y = bbox.get("y", 0) + bbox.get("height", 0) / 2
                    rel_x = face_center_x / width
                    rel_y = face_center_y / height
                    
                    # Face in corner = likely webcam overlay
                    if (rel_x < 0.35 or rel_x > 0.65) and (rel_y < 0.35 or rel_y > 0.65):
                        return True
        
        return False

    def calculate_smart_crop_center(
        self,
        frame_analysis: FrameAnalysis,
        layout_type: str,
        is_face_region: bool = True,
    ) -> Tuple[int, int]:
        """
        Calculate optimal crop center based on analysis and layout type.
        
        Args:
            frame_analysis: Analysis results for the frame
            layout_type: "talking_head" or "screen_share"
            is_face_region: If True, return center for face; if False, for screen content
            
        Returns:
            (x, y) center coordinates for cropping
        """
        if layout_type == "talking_head":
            # For talking head, always follow the face
            if frame_analysis.primary_face_center:
                return frame_analysis.primary_face_center
            else:
                # Fallback to center
                return (
                    frame_analysis.frame_width // 2,
                    frame_analysis.frame_height // 2,
                )
        
        elif layout_type == "screen_share":
            if is_face_region:
                # For face portion of screen share
                if frame_analysis.primary_face_center:
                    return frame_analysis.primary_face_center
                elif frame_analysis.webcam_region:
                    # Use webcam region center
                    return (
                        frame_analysis.webcam_region.x + frame_analysis.webcam_region.width // 2,
                        frame_analysis.webcam_region.y + frame_analysis.webcam_region.height // 2,
                    )
                else:
                    # Assume webcam in bottom corner
                    return (
                        frame_analysis.frame_width // 4,
                        int(frame_analysis.frame_height * 0.75),
                    )
            else:
                # For screen content portion
                return frame_analysis.screen_content_center
        
        # Default fallback
        return (
            frame_analysis.frame_width // 2,
            frame_analysis.frame_height // 2,
        )

    def analyze_for_split_layout(
        self,
        frame: Optional[np.ndarray],
        face_detections: list[dict],
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> dict:
        """
        Analyze frame specifically for split layout (screen top, face bottom).

        Split Layout:
        - Screen content: 50% (TOP)
        - Speaker face: 50% (BOTTOM)
        - Captions: Overlaid on combined video

        This method identifies:
        1. Whether the frame is suitable for split layout
        2. Primary speaker face position for tight face cropping
        3. Optimal screen content center avoiding webcam overlay
        4. Webcam overlay position if present

        Args:
            frame: Video frame as numpy array (BGR), can be None
            face_detections: List of face detection dicts with 'bbox' and 'confidence'
            frame_width: Frame width (used when frame is None)
            frame_height: Frame height (used when frame is None)

        Returns:
            dict with analysis results for split layout rendering
        """
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            width, height = frame_width, frame_height
        
        frame_area = width * height
        
        # Results dict
        result = {
            "is_split_suitable": False,
            "webcam_position": None,
            "primary_face_center": None,
            "primary_face_bbox": None,
            "screen_content_center": (width // 2, int(height * 0.40)),  # Default upper center
            "face_size_ratio": 0.0,
            "layout_confidence": 0.0,
        }

        if not face_detections:
            # No faces detected - might be pure screen share
            result["is_split_suitable"] = True
            result["layout_confidence"] = 0.5
            return result
        
        # Find primary face (largest with highest confidence)
        weighted_faces = []
        for face in face_detections:
            bbox = face.get("bbox", {})
            confidence = face.get("confidence", 0.5)
            face_w = bbox.get("width", 0)
            face_h = bbox.get("height", 0)
            face_area_px = face_w * face_h
            
            # Weight by confidence × area
            weight = confidence * face_area_px
            weighted_faces.append((face, weight, face_area_px))
        
        weighted_faces.sort(key=lambda x: x[1], reverse=True)
        
        if weighted_faces:
            primary_face, _, face_area_px = weighted_faces[0]
            bbox = primary_face.get("bbox", {})
            
            center_x = int(bbox.get("x", 0) + bbox.get("width", 0) / 2)
            center_y = int(bbox.get("y", 0) + bbox.get("height", 0) / 2)
            
            result["primary_face_center"] = (center_x, center_y)
            result["primary_face_bbox"] = bbox
            result["face_size_ratio"] = face_area_px / frame_area if frame_area > 0 else 0
            
            # Determine webcam position (small face in corner)
            rel_x = center_x / width
            rel_y = center_y / height
            face_ratio = face_area_px / frame_area if frame_area > 0 else 0
            
            # Small face in corner indicates webcam overlay
            if face_ratio < 0.12:
                if rel_x < 0.35:
                    if rel_y < 0.35:
                        result["webcam_position"] = "top_left"
                    elif rel_y > 0.65:
                        result["webcam_position"] = "bottom_left"
                elif rel_x > 0.65:
                    if rel_y < 0.35:
                        result["webcam_position"] = "top_right"
                    elif rel_y > 0.65:
                        result["webcam_position"] = "bottom_right"
            
            # Calculate optimal screen content center avoiding webcam
            if result["webcam_position"]:
                result["is_split_suitable"] = True
                result["layout_confidence"] = 0.9

                # Shift screen center away from webcam
                if "left" in result["webcam_position"]:
                    screen_x = int(width * 0.55)
                else:
                    screen_x = int(width * 0.45)

                if "top" in result["webcam_position"]:
                    screen_y = int(height * 0.55)
                else:
                    screen_y = int(height * 0.40)

                result["screen_content_center"] = (screen_x, screen_y)
            else:
                # Face is centered/large - could still be screen share with speaker overlay
                # Or could be talking head
                if face_ratio > 0.15:
                    # Large face - likely talking head, split layout not ideal
                    result["is_split_suitable"] = False
                    result["layout_confidence"] = 0.3
                else:
                    # Medium face - could work for split layout
                    result["is_split_suitable"] = True
                    result["layout_confidence"] = 0.6

        return result

