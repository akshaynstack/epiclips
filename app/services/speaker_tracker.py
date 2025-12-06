"""
Speaker Tracker Service - Active speaker detection and face-to-speaker mapping.

This service provides:
1. CPU-only speaker diarization using Resemblyzer + clustering
2. Speaker-to-face track mapping based on speech timing
3. Active speaker detection at any timestamp
4. Speaker importance scoring for layout decisions
5. Background face filtering

Key insight: Layout decisions should be driven by ACTIVE SPEAKERS, not face count.
- SINGLE layout when only one mapped speaker is active
- SPLIT layout only when two or more mapped speakers are currently active
- Non-speaking faces should NEVER trigger split-screen
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Minimum speaking time percentage to be considered a "real" speaker (not background)
MIN_SPEAKING_TIME_PERCENT = 2.0  # 2% - faces below this are background

# Hysteresis for layout switching - layout must be stable for this duration
LAYOUT_SWITCH_HYSTERESIS_MS = 1000  # 1 second

# Time window for smoothing layout decisions
LAYOUT_SMOOTHING_WINDOW_MS = 500  # 0.5 second

# ============================================================================
# CAMERA REGION FILTERING
# ============================================================================
# In videos with screen share + camera, the camera is typically in the bottom portion.
# Faces in the top portion are likely from screen content (thumbnails, photos, etc.)
# and should be IGNORED for speaker tracking.

# Minimum Y position (as fraction of frame height) for a face to be considered
# in the camera region. Faces above this line are filtered out.
# 0.25 means faces must be in the bottom 75% of the frame (more lenient)
CAMERA_REGION_MIN_Y = 0.25

# Minimum face area (as fraction of frame area) to be considered a real person
# Filters out small faces from thumbnails, photos in presentations, etc.
# 0.005 = 0.5% of frame - filters out very tiny faces but keeps small webcams
MIN_FACE_AREA_RATIO = 0.005  # 0.5% of frame

# Speaker embedding model parameters (Resemblyzer)
EMBEDDING_SAMPLE_RATE = 16000
EMBEDDING_HOP_SIZE = 160  # 10ms at 16kHz
EMBEDDING_WINDOW_SIZE = 400  # 25ms at 16kHz

# Clustering parameters
MIN_CLUSTER_SIZE = 3  # Minimum segments to form a speaker cluster
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold for same speaker


@dataclass
class SpeakerSegment:
    """A segment of speech attributed to a speaker."""
    
    speaker_id: int
    start_time_ms: int
    end_time_ms: int
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    
    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms


@dataclass
class FaceTrack:
    """Represents a tracked face across frames."""
    
    track_id: int
    timestamps_ms: list[int] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    confidences: list[float] = field(default_factory=list)
    
    # Computed properties
    total_visible_ms: float = 0.0
    avg_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    avg_center: tuple[float, float] = (0.5, 0.5)  # Normalized center position
    
    # Speaker mapping
    mapped_speaker_id: Optional[int] = None
    speaking_time_ms: float = 0.0
    speaking_time_percent: float = 0.0
    is_background: bool = False
    
    @property
    def importance_score(self) -> float:
        """
        Calculate importance score for this face track.
        
        Score = 0.6 * speaking_time + 0.2 * bbox_area + 0.2 * centrality
        """
        # Normalize speaking time (assuming max 100% = 1.0)
        speaking_score = min(self.speaking_time_percent / 50.0, 1.0)  # 50% speaking = max
        
        # Calculate bbox area score (larger = more important)
        if self.bboxes:
            avg_area = sum(b[2] * b[3] for b in self.bboxes) / len(self.bboxes)
            # Assume max face area is 40% of frame (0.4 * frame_area)
            # Normalize assuming 1920x1080 frame
            max_area = 0.4 * 1920 * 1080
            area_score = min(avg_area / max_area, 1.0)
        else:
            area_score = 0.0
        
        # Calculate centrality score (center of frame = 1.0, corners = 0.0)
        center_x, center_y = self.avg_center
        # Distance from center (0.5, 0.5), normalized to [0, 1]
        dist_from_center = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
        max_dist = (0.5 ** 2 + 0.5 ** 2) ** 0.5  # Corner distance
        centrality_score = 1.0 - (dist_from_center / max_dist)
        
        return 0.6 * speaking_score + 0.2 * area_score + 0.2 * centrality_score


@dataclass
class SpeakerToFaceMapping:
    """Maps a speaker to a face track."""
    
    speaker_id: int
    face_track_id: int
    confidence: float
    overlap_ms: float  # Total milliseconds where speaker was active and face was visible


@dataclass 
class ActiveSpeakerInfo:
    """Information about active speakers at a given timestamp."""
    
    timestamp_ms: int
    active_speaker_ids: list[int]
    active_face_track_ids: list[int]
    recommended_layout: str  # "talking_head" or "screen_share"
    confidence: float


@dataclass
class SpeakerAnalysis:
    """Complete speaker analysis result."""
    
    speaker_segments: list[SpeakerSegment]
    face_tracks: list[FaceTrack]
    speaker_to_face_mappings: list[SpeakerToFaceMapping]
    total_speakers: int
    total_speaking_time_ms: float
    background_face_ids: list[int]
    
    def get_active_speakers_at(self, timestamp_ms: int) -> list[int]:
        """Get list of speaker IDs active at given timestamp."""
        active = []
        for seg in self.speaker_segments:
            if seg.start_time_ms <= timestamp_ms < seg.end_time_ms:
                if seg.speaker_id not in active:
                    active.append(seg.speaker_id)
        return active
    
    def get_active_faces_at(self, timestamp_ms: int) -> list[int]:
        """Get list of face track IDs active (speaking) at given timestamp."""
        active_speakers = self.get_active_speakers_at(timestamp_ms)
        
        face_ids = []
        for mapping in self.speaker_to_face_mappings:
            if mapping.speaker_id in active_speakers:
                if mapping.face_track_id not in face_ids:
                    face_ids.append(mapping.face_track_id)
        
        return face_ids
    
    def get_layout_at(self, timestamp_ms: int) -> tuple[str, float]:
        """
        Get recommended layout at given timestamp based on active speakers.
        
        Returns:
            Tuple of (layout_type, confidence)
        """
        active_faces = self.get_active_faces_at(timestamp_ms)
        
        # Filter out background faces
        non_background_faces = [
            fid for fid in active_faces 
            if fid not in self.background_face_ids
        ]
        
        if len(non_background_faces) == 0:
            # No active speakers - default to talking_head
            return ("talking_head", 0.5)
        elif len(non_background_faces) == 1:
            # Single active speaker - talking_head
            return ("talking_head", 0.9)
        else:
            # Multiple active speakers - split screen
            return ("screen_share", 0.9)


class SpeakerTracker:
    """
    Tracks speakers and maps them to face tracks for intelligent layout decisions.
    
    This service runs entirely on CPU using:
    - Resemblyzer for speaker embeddings
    - Agglomerative clustering for speaker identification
    - Overlap analysis for speaker-to-face mapping
    """
    
    def __init__(self):
        self._encoder = None
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize speaker encoder (Resemblyzer)."""
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
            logger.info("Resemblyzer speaker encoder initialized (CPU)")
        except ImportError:
            logger.warning(
                "Resemblyzer not installed. Speaker diarization will use fallback. "
                "Install with: pip install resemblyzer"
            )
            self._encoder = None
        except Exception as e:
            logger.warning(f"Failed to initialize Resemblyzer: {e}")
            self._encoder = None
    
    async def analyze_speakers(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        face_detections: list[dict],
        transcript_segments: Optional[list] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> SpeakerAnalysis:
        """
        Analyze speakers in a video clip and map them to face tracks.
        
        Args:
            video_path: Path to the video file
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            face_detections: List of face detection frames from detection pipeline
            transcript_segments: Optional transcript segments with timing
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            SpeakerAnalysis with speaker segments, face tracks, and mappings
        """
        logger.info(f"Analyzing speakers: {start_ms}ms - {end_ms}ms")
        
        # Step 1: Build face tracks from detections
        face_tracks = self._build_face_tracks(
            face_detections, frame_width, frame_height
        )
        logger.info(f"Built {len(face_tracks)} face tracks")
        
        # Step 2: Get speaker segments (diarization)
        speaker_segments = await self._get_speaker_segments(
            video_path, start_ms, end_ms, transcript_segments
        )
        logger.info(f"Found {len(speaker_segments)} speaker segments")
        
        # Step 3: Map speakers to face tracks
        mappings = self._map_speakers_to_faces(
            speaker_segments, face_tracks, start_ms, end_ms
        )
        logger.info(f"Created {len(mappings)} speaker-to-face mappings")
        
        # Step 4: Calculate speaking time for each face track
        self._calculate_speaking_times(face_tracks, speaker_segments, mappings, end_ms - start_ms)
        
        # Step 5: Identify background faces
        background_face_ids = [
            track.track_id for track in face_tracks
            if track.speaking_time_percent < MIN_SPEAKING_TIME_PERCENT
        ]
        
        for track in face_tracks:
            track.is_background = track.track_id in background_face_ids
        
        logger.info(f"Identified {len(background_face_ids)} background faces")
        
        # Count unique speakers
        unique_speakers = set(seg.speaker_id for seg in speaker_segments)
        total_speaking_ms = sum(seg.duration_ms for seg in speaker_segments)
        
        return SpeakerAnalysis(
            speaker_segments=speaker_segments,
            face_tracks=face_tracks,
            speaker_to_face_mappings=mappings,
            total_speakers=len(unique_speakers),
            total_speaking_time_ms=total_speaking_ms,
            background_face_ids=background_face_ids,
        )
    
    def _is_face_in_camera_region(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Check if a face is in the camera region (bottom portion of frame).
        
        Faces in the top portion of the frame are likely from screen content
        (thumbnails, photos, UI elements) and should be filtered out.
        
        Args:
            bbox: (x, y, width, height) of the face
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            True if face is in camera region, False if in screen region
        """
        x, y, w, h = bbox
        frame_area = frame_width * frame_height
        
        # Check 1: Face area - filter out tiny faces (thumbnails, photos)
        face_area = w * h
        face_area_ratio = face_area / frame_area if frame_area > 0 else 0
        if face_area_ratio < MIN_FACE_AREA_RATIO:
            return False
        
        # Check 2: Y position - face center should be in bottom portion
        face_center_y = (y + h / 2) / frame_height
        if face_center_y < CAMERA_REGION_MIN_Y:
            # Face is in top portion (likely screen content)
            return False
        
        return True
    
    def _filter_frame_faces(
        self,
        detections: list[dict],
        frame_width: int,
        frame_height: int
    ) -> list[dict]:
        """
        Filter faces in a single frame based on relative size and position.
        Keeps the likely reactor(s) and removes screen content faces.
        """
        if not detections:
            return []
            
        valid_detections = []
        
        # 1. Calculate areas and centers
        face_props = []
        for det in detections:
            bbox = det.get("bbox", {})
            x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
            area = w * h
            center_y = y + h/2
            face_props.append({
                "det": det,
                "area": area,
                "center_y": center_y,
                "y": y,
                "h": h
            })
            
        if not face_props:
            return []
            
        # 2. Find dominant face (largest area with position bias)
        # Score = Area * PositionWeight
        # PositionWeight is higher for bottom region.
        
        for prop in face_props:
            # Normalize area (0-1 relative to frame)
            norm_area = prop["area"] / (frame_width * frame_height) if frame_width * frame_height > 0 else 0
            
            # Normalize Y (0-1, 0 is top, 1 is bottom)
            norm_y = prop["center_y"] / frame_height if frame_height > 0 else 0
            
            # Position weight: favor bottom
            # If y < 0.3 (top 30%), weight is low.
            if norm_y < 0.3:
                pos_weight = 0.2
            elif norm_y < 0.5:
                pos_weight = 0.5
            else:
                pos_weight = 1.0
                
            prop["score"] = norm_area * pos_weight
            
        # Sort by score descending
        face_props.sort(key=lambda p: p["score"], reverse=True)
        
        dominant_face = face_props[0]
        
        # 3. Filter based on dominant face
        for prop in face_props:
            # Always keep the dominant face (if it meets min absolute requirements)
            if prop == dominant_face:
                # Check absolute min requirements (tiny faces are noise)
                if prop["area"] / (frame_width * frame_height) < MIN_FACE_AREA_RATIO:
                    continue
                valid_detections.append(prop["det"])
                continue
                
            # For other faces:
            # - Must be comparable in size (e.g., > 30% of dominant face)
            # - Must not be significantly higher than dominant face (if dominant is at bottom)
            
            size_ratio = prop["area"] / dominant_face["area"] if dominant_face["area"] > 0 else 0
            
            # If it's tiny compared to dominant, drop it
            if size_ratio < 0.3:
                continue
                
            # If it's significantly higher (screen content), drop it
            # "Significantly higher" = center_y is > 20% of screen height above dominant
            if (dominant_face["center_y"] - prop["center_y"]) / frame_height > 0.2:
                continue
                
            valid_detections.append(prop["det"])
            
        return valid_detections

    def _build_face_tracks(
        self,
        face_detections: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> list[FaceTrack]:
        """
        Build face tracks from detection frames.
        
        Groups detections by track_id and calculates aggregate properties.
        
        IMPORTANT: Filters out faces in the screen region (top portion of frame)
        to avoid tracking faces from thumbnails, photos, or UI content.
        """
        tracks_dict: dict[int, FaceTrack] = {}
        filtered_count = 0
        
        for frame_data in face_detections:
            timestamp_ms = frame_data.get("timestamp_ms", 0)
            # Handle both "detections" (new format) and direct detection list
            detections = frame_data.get("detections", [])
            if not detections and isinstance(frame_data, dict):
                # Try timestamp_sec format from detection_pipeline
                timestamp_sec = frame_data.get("timestamp_sec", 0)
                if timestamp_sec:
                    timestamp_ms = int(timestamp_sec * 1000)
            
            # Apply dynamic filtering per frame
            valid_detections = self._filter_frame_faces(detections, frame_width, frame_height)
            
            # Count how many were filtered
            filtered_count += len(detections) - len(valid_detections)
            
            for det in valid_detections:
                track_id = det.get("track_id", 0)
                bbox = det.get("bbox", {})
                confidence = det.get("confidence", 0.5)
                
                x = bbox.get("x", 0)
                y = bbox.get("y", 0)
                w = bbox.get("width", 0)
                h = bbox.get("height", 0)
                
                # Legacy check just in case (though _filter_frame_faces handles it)
                # if not self._is_face_in_camera_region((x, y, w, h), frame_width, frame_height):
                #    continue
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = FaceTrack(track_id=track_id)
                
                track = tracks_dict[track_id]
                track.timestamps_ms.append(timestamp_ms)
                track.bboxes.append((x, y, w, h))
                track.confidences.append(confidence)
        
        # Calculate aggregate properties for each track
        tracks = []
        for track in tracks_dict.values():
            if track.timestamps_ms:
                # Calculate total visible time
                track.timestamps_ms.sort()
                if len(track.timestamps_ms) > 1:
                    # Estimate visibility duration from timestamps
                    track.total_visible_ms = track.timestamps_ms[-1] - track.timestamps_ms[0]
                else:
                    track.total_visible_ms = 500  # Default single frame visibility
                
                # Calculate average bbox
                avg_x = sum(b[0] for b in track.bboxes) / len(track.bboxes)
                avg_y = sum(b[1] for b in track.bboxes) / len(track.bboxes)
                avg_w = sum(b[2] for b in track.bboxes) / len(track.bboxes)
                avg_h = sum(b[3] for b in track.bboxes) / len(track.bboxes)
                track.avg_bbox = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
                
                # Calculate average center (normalized)
                center_x = (avg_x + avg_w / 2) / frame_width
                center_y = (avg_y + avg_h / 2) / frame_height
                track.avg_center = (center_x, center_y)
                
                tracks.append(track)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} faces in screen region (top of frame)")
        
        return tracks
    
    async def _get_speaker_segments(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
        transcript_segments: Optional[list] = None,
    ) -> list[SpeakerSegment]:
        """
        Get speaker segments via diarization.
        
        Uses Resemblyzer + clustering if available, otherwise falls back to
        transcript-based or simple VAD-based segmentation.
        """
        # If we have transcript segments with speaker labels, use those
        if transcript_segments:
            segments = self._segments_from_transcript(transcript_segments, start_ms, end_ms)
            if segments:
                return segments
        
        # Try Resemblyzer-based diarization
        if self._encoder is not None:
            try:
                return await self._diarize_with_resemblyzer(video_path, start_ms, end_ms)
            except Exception as e:
                logger.warning(f"Resemblyzer diarization failed: {e}")
        
        # Fallback: Simple energy-based segmentation (single speaker)
        return await self._simple_vad_segments(video_path, start_ms, end_ms)
    
    def _segments_from_transcript(
        self,
        transcript_segments: list,
        start_ms: int,
        end_ms: int,
    ) -> list[SpeakerSegment]:
        """Extract speaker segments from transcript if speaker labels exist."""
        segments = []
        
        for seg in transcript_segments:
            seg_start = getattr(seg, 'start_time_ms', 0)
            seg_end = getattr(seg, 'end_time_ms', 0)
            speaker_label = getattr(seg, 'speaker_label', None)
            
            # Filter to clip range
            if seg_end < start_ms or seg_start > end_ms:
                continue
            
            # Clamp to clip bounds
            seg_start = max(seg_start, start_ms)
            seg_end = min(seg_end, end_ms)
            
            # Convert speaker label to ID
            if speaker_label:
                try:
                    speaker_id = int(speaker_label.replace("SPEAKER_", "").replace("speaker_", ""))
                except ValueError:
                    speaker_id = hash(speaker_label) % 100
            else:
                speaker_id = 0  # Default single speaker
            
            segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                start_time_ms=seg_start,
                end_time_ms=seg_end,
                confidence=0.8,
            ))
        
        return segments
    
    async def _diarize_with_resemblyzer(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
    ) -> list[SpeakerSegment]:
        """
        Perform speaker diarization using Resemblyzer embeddings + clustering.
        
        This is CPU-only and runs in a thread pool to avoid blocking.
        """
        from resemblyzer import preprocess_wav
        
        loop = asyncio.get_event_loop()
        
        def _sync_diarize() -> list[SpeakerSegment]:
            # Extract audio for the clip
            audio_path = video_path.replace(".mp4", "_temp_audio.wav")
            try:
                # Extract audio using ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_ms / 1000),
                    "-t", str((end_ms - start_ms) / 1000),
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(EMBEDDING_SAMPLE_RATE),
                    "-ac", "1",
                    audio_path,
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                
                # Load and preprocess audio
                wav = preprocess_wav(audio_path)
                
                # Segment audio into windows and compute embeddings
                segment_duration_ms = 2000  # 2 second segments
                segment_samples = int(segment_duration_ms * EMBEDDING_SAMPLE_RATE / 1000)
                
                embeddings = []
                segment_times = []
                
                for i in range(0, len(wav) - segment_samples, segment_samples // 2):
                    segment = wav[i:i + segment_samples]
                    
                    # Skip silent segments
                    if np.abs(segment).mean() < 0.01:
                        continue
                    
                    emb = self._encoder.embed_utterance(segment)
                    embeddings.append(emb)
                    
                    seg_start = start_ms + int(i * 1000 / EMBEDDING_SAMPLE_RATE)
                    seg_end = seg_start + segment_duration_ms
                    segment_times.append((seg_start, seg_end))
                
                if not embeddings:
                    return [SpeakerSegment(
                        speaker_id=0,
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        confidence=0.5,
                    )]
                
                # Cluster embeddings
                embeddings_array = np.array(embeddings)
                speaker_ids = self._cluster_embeddings(embeddings_array)
                
                # Create speaker segments
                segments = []
                for (seg_start, seg_end), speaker_id in zip(segment_times, speaker_ids):
                    segments.append(SpeakerSegment(
                        speaker_id=speaker_id,
                        start_time_ms=seg_start,
                        end_time_ms=seg_end,
                        confidence=0.8,
                    ))
                
                # Merge consecutive segments with same speaker
                merged = self._merge_speaker_segments(segments)
                
                return merged
                
            finally:
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
        
        return await loop.run_in_executor(None, _sync_diarize)
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        """
        Cluster speaker embeddings using agglomerative clustering.
        
        Returns list of speaker IDs for each embedding.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            if len(embeddings) < 2:
                return [0] * len(embeddings)
            
            # Use cosine distance for speaker embeddings
            # Agglomerative clustering with distance threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - SIMILARITY_THRESHOLD,  # Convert similarity to distance
                metric="cosine",
                linkage="average",
            )
            
            labels = clustering.fit_predict(embeddings)
            return labels.tolist()
            
        except ImportError:
            logger.warning("sklearn not available, using single speaker fallback")
            return [0] * len(embeddings)
    
    def _merge_speaker_segments(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        """Merge consecutive segments from the same speaker."""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            # Merge if same speaker and close in time (within 500ms)
            if seg.speaker_id == last.speaker_id and seg.start_time_ms - last.end_time_ms < 500:
                last.end_time_ms = seg.end_time_ms
            else:
                merged.append(seg)
        
        return merged
    
    async def _simple_vad_segments(
        self,
        video_path: str,
        start_ms: int,
        end_ms: int,
    ) -> list[SpeakerSegment]:
        """
        Simple VAD-based segmentation as fallback.
        
        Assumes single speaker, uses energy-based voice activity detection.
        """
        # Simple fallback: assume single speaker for entire duration
        return [SpeakerSegment(
            speaker_id=0,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            confidence=0.5,
        )]
    
    def _map_speakers_to_faces(
        self,
        speaker_segments: list[SpeakerSegment],
        face_tracks: list[FaceTrack],
        start_ms: int,
        end_ms: int,
    ) -> list[SpeakerToFaceMapping]:
        """
        Map speakers to face tracks based on temporal overlap.
        
        For each speaker, find the face track that appears most often
        during their speech segments.
        """
        if not speaker_segments or not face_tracks:
            return []
        
        # Group speaker segments by speaker ID
        speaker_to_segments: dict[int, list[SpeakerSegment]] = {}
        for seg in speaker_segments:
            if seg.speaker_id not in speaker_to_segments:
                speaker_to_segments[seg.speaker_id] = []
            speaker_to_segments[seg.speaker_id].append(seg)
        
        mappings = []
        
        for speaker_id, segs in speaker_to_segments.items():
            # Calculate overlap with each face track
            best_track_id = None
            best_overlap = 0.0
            
            for track in face_tracks:
                overlap = self._calculate_overlap(segs, track)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_track_id = track.track_id
            
            if best_track_id is not None and best_overlap > 0:
                # Calculate confidence based on overlap ratio
                total_speaking = sum(s.duration_ms for s in segs)
                confidence = min(best_overlap / total_speaking, 1.0) if total_speaking > 0 else 0.5
                
                mappings.append(SpeakerToFaceMapping(
                    speaker_id=speaker_id,
                    face_track_id=best_track_id,
                    confidence=confidence,
                    overlap_ms=best_overlap,
                ))
                
                # Update face track with speaker mapping
                for track in face_tracks:
                    if track.track_id == best_track_id:
                        track.mapped_speaker_id = speaker_id
                        break
        
        return mappings
    
    def _calculate_overlap(
        self,
        speaker_segments: list[SpeakerSegment],
        face_track: FaceTrack,
    ) -> float:
        """Calculate total overlap time between speaker segments and face visibility."""
        total_overlap = 0.0
        
        for seg in speaker_segments:
            for ts in face_track.timestamps_ms:
                # Check if face is visible during this speaker segment
                # Assume face detection has some temporal extent (~500ms)
                face_start = ts - 250
                face_end = ts + 250
                
                # Calculate overlap
                overlap_start = max(seg.start_time_ms, face_start)
                overlap_end = min(seg.end_time_ms, face_end)
                
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start
        
        return total_overlap
    
    def _calculate_speaking_times(
        self,
        face_tracks: list[FaceTrack],
        speaker_segments: list[SpeakerSegment],
        mappings: list[SpeakerToFaceMapping],
        total_duration_ms: float,
    ):
        """Calculate speaking time for each face track based on mapped speakers."""
        # Create speaker to face mapping lookup
        speaker_to_face = {m.speaker_id: m.face_track_id for m in mappings}
        
        # Calculate speaking time per face track
        face_speaking_time: dict[int, float] = {t.track_id: 0.0 for t in face_tracks}
        
        for seg in speaker_segments:
            face_id = speaker_to_face.get(seg.speaker_id)
            if face_id is not None:
                face_speaking_time[face_id] += seg.duration_ms
        
        # Update face tracks
        for track in face_tracks:
            track.speaking_time_ms = face_speaking_time.get(track.track_id, 0.0)
            track.speaking_time_percent = (
                (track.speaking_time_ms / total_duration_ms * 100)
                if total_duration_ms > 0 else 0.0
            )
    
    def get_layout_timeline(
        self,
        speaker_analysis: SpeakerAnalysis,
        start_ms: int,
        end_ms: int,
        sample_interval_ms: int = 500,
    ) -> list[dict]:
        """
        Generate a layout timeline based on active speakers.
        
        Applies hysteresis to prevent rapid layout switching.
        
        Returns list of layout decisions with timestamps.
        """
        timeline = []
        current_layout = None
        layout_start_ms = start_ms
        pending_layout = None
        pending_start_ms = None
        
        for t in range(start_ms, end_ms, sample_interval_ms):
            layout, confidence = speaker_analysis.get_layout_at(t)
            
            if current_layout is None:
                # First layout decision
                current_layout = layout
                layout_start_ms = t
                continue
            
            if layout != current_layout:
                # Layout change detected
                if pending_layout == layout:
                    # Same pending layout - check hysteresis
                    if t - pending_start_ms >= LAYOUT_SWITCH_HYSTERESIS_MS:
                        # Hysteresis passed - commit layout change
                        timeline.append({
                            "start_ms": layout_start_ms,
                            "end_ms": t,
                            "layout": current_layout,
                        })
                        current_layout = layout
                        layout_start_ms = t
                        pending_layout = None
                        pending_start_ms = None
                else:
                    # New pending layout
                    pending_layout = layout
                    pending_start_ms = t
            else:
                # Layout stable - reset pending
                pending_layout = None
                pending_start_ms = None
        
        # Add final segment
        if current_layout:
            timeline.append({
                "start_ms": layout_start_ms,
                "end_ms": end_ms,
                "layout": current_layout,
            })
        
        return timeline
    
    def export_debug_info(
        self,
        speaker_analysis: SpeakerAnalysis,
        layout_timeline: list[dict],
        output_path: str,
    ):
        """
        Export debug information to JSON file.
        
        Includes:
        - Speaker segments with timestamps
        - Face track timelines
        - Speaker-to-track mapping
        - Final layout timeline
        """
        import json
        
        debug_info = {
            "speaker_segments": [
                {
                    "speaker_id": seg.speaker_id,
                    "start_ms": seg.start_time_ms,
                    "end_ms": seg.end_time_ms,
                    "duration_ms": seg.duration_ms,
                    "confidence": seg.confidence,
                }
                for seg in speaker_analysis.speaker_segments
            ],
            "face_tracks": [
                {
                    "track_id": track.track_id,
                    "total_visible_ms": track.total_visible_ms,
                    "avg_bbox": track.avg_bbox,
                    "avg_center": track.avg_center,
                    "mapped_speaker_id": track.mapped_speaker_id,
                    "speaking_time_ms": track.speaking_time_ms,
                    "speaking_time_percent": track.speaking_time_percent,
                    "is_background": track.is_background,
                    "importance_score": track.importance_score,
                    "timestamp_count": len(track.timestamps_ms),
                }
                for track in speaker_analysis.face_tracks
            ],
            "speaker_to_face_mappings": [
                {
                    "speaker_id": m.speaker_id,
                    "face_track_id": m.face_track_id,
                    "confidence": m.confidence,
                    "overlap_ms": m.overlap_ms,
                }
                for m in speaker_analysis.speaker_to_face_mappings
            ],
            "layout_timeline": layout_timeline,
            "summary": {
                "total_speakers": speaker_analysis.total_speakers,
                "total_face_tracks": len(speaker_analysis.face_tracks),
                "total_speaking_time_ms": speaker_analysis.total_speaking_time_ms,
                "background_face_count": len(speaker_analysis.background_face_ids),
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(debug_info, f, indent=2)
        
        logger.info(f"Debug info exported to: {output_path}")


# ============================================================================
# SMOOTH TRACKING UTILITIES
# ============================================================================

class SmoothBBoxTracker:
    """
    Provides smooth bounding box tracking with weighted average smoothing.
    
    Prevents jumpy camera movements by:
    1. Using exponential moving average for bbox center
    2. Applying hysteresis for significant movements
    3. Clamping to video bounds
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.3,  # Lower = smoother but more lag
        min_movement_threshold: float = 0.02,  # Minimum movement to trigger update (fraction of frame)
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        self.smoothing_factor = smoothing_factor
        self.min_movement_threshold = min_movement_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Current smoothed state
        self._current_center: Optional[tuple[float, float]] = None
        self._current_size: Optional[tuple[float, float]] = None
        self._history: list[tuple[float, float, float, float]] = []  # (cx, cy, w, h)
    
    def update(
        self,
        bbox: tuple[int, int, int, int],  # (x, y, w, h)
        confidence: float = 1.0,
    ) -> tuple[int, int, int, int]:
        """
        Update tracker with new detection and return smoothed bbox.
        
        Args:
            bbox: New detection (x, y, width, height)
            confidence: Detection confidence (higher = trust more)
            
        Returns:
            Smoothed bbox (x, y, width, height)
        """
        x, y, w, h = bbox
        
        # Calculate center
        new_cx = x + w / 2
        new_cy = y + h / 2
        
        if self._current_center is None:
            # First detection - use directly
            self._current_center = (new_cx, new_cy)
            self._current_size = (w, h)
        else:
            # Calculate movement
            old_cx, old_cy = self._current_center
            movement = (
                ((new_cx - old_cx) / self.frame_width) ** 2 +
                ((new_cy - old_cy) / self.frame_height) ** 2
            ) ** 0.5
            
            # Only update if movement exceeds threshold
            if movement > self.min_movement_threshold:
                # Apply exponential moving average
                # Adjust smoothing based on confidence
                effective_smoothing = self.smoothing_factor * confidence
                
                smoothed_cx = old_cx + effective_smoothing * (new_cx - old_cx)
                smoothed_cy = old_cy + effective_smoothing * (new_cy - old_cy)
                
                self._current_center = (smoothed_cx, smoothed_cy)
            
            # Always smooth size changes
            old_w, old_h = self._current_size
            smoothed_w = old_w + self.smoothing_factor * (w - old_w)
            smoothed_h = old_h + self.smoothing_factor * (h - old_h)
            self._current_size = (smoothed_w, smoothed_h)
        
        # Add to history for analysis
        cx, cy = self._current_center
        sw, sh = self._current_size
        self._history.append((cx, cy, sw, sh))
        if len(self._history) > 100:
            self._history = self._history[-50:]
        
        # Convert back to bbox and clamp to bounds
        smoothed_x = int(max(0, cx - sw / 2))
        smoothed_y = int(max(0, cy - sh / 2))
        smoothed_w = int(min(sw, self.frame_width - smoothed_x))
        smoothed_h = int(min(sh, self.frame_height - smoothed_y))
        
        return (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
    
    def get_stable_bbox(self, window_size: int = 5) -> tuple[int, int, int, int]:
        """
        Get a stable bbox using median of recent history.
        
        Useful for determining a single crop for an entire segment.
        """
        if not self._history:
            return (0, 0, self.frame_width, self.frame_height)
        
        recent = self._history[-window_size:] if len(self._history) >= window_size else self._history
        
        # Use median for stability
        cx = sorted([h[0] for h in recent])[len(recent) // 2]
        cy = sorted([h[1] for h in recent])[len(recent) // 2]
        w = sorted([h[2] for h in recent])[len(recent) // 2]
        h = sorted([h[3] for h in recent])[len(recent) // 2]
        
        x = int(max(0, cx - w / 2))
        y = int(max(0, cy - h / 2))
        
        return (x, y, int(w), int(h))
    
    def reset(self):
        """Reset tracker state."""
        self._current_center = None
        self._current_size = None
        self._history = []


def smooth_crop_timeline(
    crop_keyframes: list[dict],
    frame_width: int,
    frame_height: int,
    smoothing_factor: float = 0.3,
) -> list[dict]:
    """
    Apply smoothing to a crop timeline to prevent jumpy camera movements.
    
    Args:
        crop_keyframes: List of keyframes with 'timestamp_ms' and 'crop' fields
        frame_width: Video frame width
        frame_height: Video frame height
        smoothing_factor: How much to smooth (0 = no smoothing, 1 = no movement)
        
    Returns:
        Smoothed crop keyframes
    """
    if not crop_keyframes:
        return crop_keyframes
    
    tracker = SmoothBBoxTracker(
        smoothing_factor=smoothing_factor,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    
    smoothed = []
    for kf in sorted(crop_keyframes, key=lambda k: k.get("timestamp_ms", 0)):
        crop = kf.get("crop", {})
        bbox = (
            crop.get("x", 0),
            crop.get("y", 0),
            crop.get("width", frame_width),
            crop.get("height", frame_height),
        )
        confidence = kf.get("confidence", 1.0)
        
        smoothed_bbox = tracker.update(bbox, confidence)
        
        smoothed.append({
            "timestamp_ms": kf.get("timestamp_ms", 0),
            "crop": {
                "x": smoothed_bbox[0],
                "y": smoothed_bbox[1],
                "width": smoothed_bbox[2],
                "height": smoothed_bbox[3],
            },
            "confidence": confidence,
        })
    
    return smoothed


def calculate_stable_crop_for_segment(
    face_detections: list[dict],
    segment_start_ms: int,
    segment_end_ms: int,
    frame_width: int,
    frame_height: int,
    target_aspect_ratio: float = 9 / 16,  # 9:16 for vertical video
) -> dict:
    """
    Calculate a single stable crop for a video segment based on face positions.
    
    Uses median face position to determine crop, ensuring stability.
    
    Args:
        face_detections: List of face detection frames
        segment_start_ms: Segment start time
        segment_end_ms: Segment end time
        frame_width: Video frame width
        frame_height: Video frame height
        target_aspect_ratio: Target aspect ratio (width/height)
        
    Returns:
        Crop region dict with x, y, width, height
    """
    # Filter detections to segment time range
    segment_faces = []
    for frame in face_detections:
        ts = frame.get("timestamp_ms", 0)
        if segment_start_ms <= ts <= segment_end_ms:
            for det in frame.get("detections", []):
                bbox = det.get("bbox", {})
                segment_faces.append({
                    "x": bbox.get("x", 0),
                    "y": bbox.get("y", 0),
                    "width": bbox.get("width", 0),
                    "height": bbox.get("height", 0),
                })
    
    if not segment_faces:
        # No faces - center crop
        target_width = int(frame_height * target_aspect_ratio)
        return {
            "x": (frame_width - target_width) // 2,
            "y": 0,
            "width": target_width,
            "height": frame_height,
        }
    
    # Calculate median face center
    centers_x = [f["x"] + f["width"] / 2 for f in segment_faces]
    centers_y = [f["y"] + f["height"] / 2 for f in segment_faces]
    
    median_cx = sorted(centers_x)[len(centers_x) // 2]
    median_cy = sorted(centers_y)[len(centers_y) // 2]
    
    # Calculate crop dimensions
    target_width = int(frame_height * target_aspect_ratio)
    
    # Center crop on face
    crop_x = int(median_cx - target_width / 2)
    
    # Clamp to bounds
    crop_x = max(0, min(crop_x, frame_width - target_width))
    
    return {
        "x": crop_x,
        "y": 0,
        "width": target_width,
        "height": frame_height,
    }

