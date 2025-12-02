"""
Transcription Service - Audio transcription using Whisper via Groq.
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptWord:
    """Word-level timing for precise caption display."""
    
    word: str
    start_time_ms: int
    end_time_ms: int


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    
    start_time_ms: int
    end_time_ms: int
    text: str
    speaker_label: Optional[str] = None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Result of transcription operation."""
    
    segments: list[TranscriptSegment]
    full_text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"


class TranscriptionService:
    """
    Service for transcribing audio using Whisper models via Groq.
    
    Groq provides the fastest Whisper inference (216x realtime) and is the most
    cost-effective option for audio transcription.
    
    Features:
    - Word-level timestamps for caption highlighting
    - Automatic chunking for long audio files (30 min chunks)
    - Language detection and specification
    """

    def __init__(self):
        self.settings = get_settings()
        self._groq_client = None
        self._init_client()

    def _init_client(self):
        """Initialize Groq client."""
        if not self.settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set, transcription will fail")
            return
        
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.settings.groq_api_key)
            logger.info("Groq client initialized for transcription")
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")

    async def transcribe(
        self,
        video_path: str,
        work_dir: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe a video file by extracting audio first.
        
        Args:
            video_path: Path to video file
            work_dir: Working directory for temporary files
            language: Optional language code (auto-detected if not specified)
            translate_to_english: Whether to translate to English
            
        Returns:
            TranscriptionResult with segments and word-level timing
        """
        if not os.path.isfile(video_path):
            raise TranscriptionError(f"Video file not found: {video_path}")
        
        # Extract audio from video
        audio_path = os.path.join(work_dir, "audio_extracted.mp3")
        await self._extract_audio_from_video(video_path, audio_path)
        
        try:
            return await self.transcribe_audio(
                audio_path=audio_path,
                language=language,
                translate_to_english=translate_to_english,
            )
        finally:
            # Cleanup extracted audio
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

    async def _extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        """Extract audio track from video file using ffmpeg."""
        logger.info(f"Extracting audio from video: {video_path}")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ab", "64k",  # Reasonable quality for speech
            "-ar", "16000",  # 16kHz sample rate for Whisper
            "-ac", "1",  # Mono
            audio_path,
        ]
        
        # Use run_in_executor for Windows compatibility
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            raise TranscriptionError(f"Failed to extract audio from video: {error_msg}")
        
        if not os.path.exists(audio_path):
            raise TranscriptionError("Audio extraction produced no output file")
        
        logger.info(f"Audio extracted to: {audio_path}")

    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            language: Optional language code (auto-detected if not specified)
            translate_to_english: Whether to translate to English
            
        Returns:
            TranscriptionResult with segments and word-level timing
        """
        if not os.path.isfile(audio_path):
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        # Get audio duration to decide if we need chunking
        duration_seconds = await self._get_audio_duration(audio_path)
        
        # Groq supports up to 30 minute chunks
        chunk_duration = self.settings.groq_chunk_duration_seconds  # 30 min
        needs_chunking = duration_seconds > chunk_duration
        
        logger.info(
            f"Transcribing audio: {audio_path} "
            f"({duration_seconds:.1f}s, provider=groq, chunking={needs_chunking})"
        )
        
        if needs_chunking:
            return await self._transcribe_chunked(
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                chunk_duration_seconds=chunk_duration,
                language=language,
                translate_to_english=translate_to_english,
            )
        else:
            return await self._transcribe_single(
                audio_path=audio_path,
                language=language,
                translate_to_english=translate_to_english,
            )

    async def _transcribe_single(
        self,
        audio_path: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ) -> TranscriptionResult:
        """Transcribe a single audio file (no chunking)."""
        if not self._groq_client:
            raise TranscriptionError("Groq client not initialized. Set GROQ_API_KEY environment variable.")
        
        model = self.settings.transcription_model
        return await self._transcribe_with_groq(
            audio_path, model, language, translate_to_english
        )

    async def _transcribe_chunked(
        self,
        audio_path: str,
        duration_seconds: float,
        chunk_duration_seconds: int,
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ) -> TranscriptionResult:
        """Transcribe long audio by splitting into chunks."""
        # Create temp directory for chunks
        audio_dir = os.path.dirname(audio_path)
        chunks_dir = os.path.join(audio_dir, "audio_chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        try:
            # Split audio into chunks
            chunks = await self._split_audio_into_chunks(
                audio_path, chunks_dir, duration_seconds, chunk_duration_seconds
            )
            
            logger.info(f"Split audio into {len(chunks)} chunks for transcription")
            
            # Groq supports high parallelism
            max_parallel = 5
            
            all_segments: list[TranscriptSegment] = []
            detected_language: Optional[str] = None
            
            # Process chunks in parallel batches
            for batch_start in range(0, len(chunks), max_parallel):
                batch = chunks[batch_start:batch_start + max_parallel]
                
                # Create tasks for parallel transcription
                tasks = []
                for chunk_info in batch:
                    task = self._transcribe_single(
                        chunk_info["path"],
                        language=language or detected_language,
                        translate_to_english=translate_to_english,
                    )
                    tasks.append((chunk_info, task))
                
                # Execute batch
                results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
                
                # Process results
                for (chunk_info, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.error(f"Chunk {chunk_info['index']} transcription failed: {result}")
                        continue
                    
                    # Capture language from first successful result
                    if detected_language is None and result.language:
                        detected_language = result.language
                    
                    # Adjust timestamps by chunk offset
                    offset_ms = chunk_info["start_ms"]
                    for segment in result.segments:
                        all_segments.append(TranscriptSegment(
                            start_time_ms=segment.start_time_ms + offset_ms,
                            end_time_ms=segment.end_time_ms + offset_ms,
                            text=segment.text,
                            speaker_label=segment.speaker_label,
                            words=[
                                TranscriptWord(
                                    word=w.word,
                                    start_time_ms=w.start_time_ms + offset_ms,
                                    end_time_ms=w.end_time_ms + offset_ms,
                                )
                                for w in segment.words
                            ],
                        ))
            
            # Sort segments by start time
            all_segments.sort(key=lambda s: s.start_time_ms)
            
            # Build full text
            full_text = " ".join(s.text for s in all_segments)
            
            return TranscriptionResult(
                segments=all_segments,
                full_text=full_text,
                language=detected_language,
                duration_seconds=duration_seconds,
                provider="groq",
                model=self.settings.transcription_model,
            )
            
        finally:
            # Cleanup chunk files
            import shutil
            if os.path.exists(chunks_dir):
                shutil.rmtree(chunks_dir, ignore_errors=True)

    async def _transcribe_with_groq(
        self,
        audio_path: str,
        model: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ) -> TranscriptionResult:
        """Transcribe using Groq API."""
        if not self._groq_client:
            raise TranscriptionError("Groq client not initialized")
        
        # Run in thread pool since Groq client is sync
        loop = asyncio.get_event_loop()
        
        def _sync_transcribe():
            with open(audio_path, "rb") as audio_file:
                kwargs = {
                    "file": (os.path.basename(audio_path), audio_file),
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["word", "segment"],
                }
                
                if language and language != "auto":
                    kwargs["language"] = language
                
                if translate_to_english:
                    # Groq uses translation endpoint
                    response = self._groq_client.audio.translations.create(**kwargs)
                else:
                    response = self._groq_client.audio.transcriptions.create(**kwargs)
                
                return response
        
        response = await loop.run_in_executor(None, _sync_transcribe)
        
        return self._parse_whisper_response(response, "groq", model)

    def _get_value(self, obj, key: str, default=None):
        """Get value from object (handles both dict and object attributes)."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_whisper_response(
        self,
        response,
        provider: str,
        model: str,
    ) -> TranscriptionResult:
        """Parse Whisper API response into TranscriptionResult."""
        segments: list[TranscriptSegment] = []
        
        # Log response type for debugging
        logger.info(f"Parsing Whisper response type: {type(response)}")
        
        # Get segments from response - handle both dict and object
        response_segments = self._get_value(response, "segments") or []
        response_words = self._get_value(response, "words") or []
        
        logger.info(f"Response has {len(response_segments)} segments, {len(response_words)} words")
        
        # Debug first segment structure
        if response_segments:
            first_seg = response_segments[0]
            logger.info(f"First segment type: {type(first_seg)}, content: {first_seg}")
        
        # Build word lookup by time for matching to segments
        word_lookup: dict[int, list[TranscriptWord]] = {}
        for word_data in response_words:
            start = self._get_value(word_data, "start", 0)
            end = self._get_value(word_data, "end", 0)
            word_text = self._get_value(word_data, "word", "")
            
            start_ms = int(float(start) * 1000)
            word_lookup.setdefault(start_ms, []).append(TranscriptWord(
                word=str(word_text).strip(),
                start_time_ms=start_ms,
                end_time_ms=int(float(end) * 1000),
            ))
        
        for seg in response_segments:
            start = self._get_value(seg, "start", 0)
            end = self._get_value(seg, "end", 0)
            text = self._get_value(seg, "text", "")
            
            start_ms = int(float(start) * 1000)
            end_ms = int(float(end) * 1000)
            text = str(text).strip()
            
            # Find words that belong to this segment
            segment_words: list[TranscriptWord] = []
            for word_start_ms in sorted(word_lookup.keys()):
                if start_ms <= word_start_ms < end_ms:
                    segment_words.extend(word_lookup[word_start_ms])
            
            segments.append(TranscriptSegment(
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                text=text,
                words=segment_words,
            ))
        
        # Fallback if no segments but we have text
        if not segments:
            full_text_raw = self._get_value(response, "text", "")
            if full_text_raw:
                full_text = str(full_text_raw).strip()
                duration = self._get_value(response, "duration", 0)
                
                logger.info(f"No segments found, using fallback with full_text length: {len(full_text)}")
                
                segments.append(TranscriptSegment(
                    start_time_ms=0,
                    end_time_ms=int(float(duration) * 1000),
                    text=full_text,
                    words=[],
                ))
        
        full_text = " ".join(s.text for s in segments)
        
        logger.info(f"Parsed {len(segments)} segments, full_text length: {len(full_text)}")
        
        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=self._get_value(response, "language"),
            duration_seconds=self._get_value(response, "duration"),
            provider=provider,
            model=model,
        )

    async def _split_audio_into_chunks(
        self,
        audio_path: str,
        output_dir: str,
        total_duration_seconds: float,
        chunk_duration_seconds: int,
    ) -> list[dict]:
        """Split audio file into chunks using ffmpeg."""
        chunks = []
        chunk_duration_ms = chunk_duration_seconds * 1000
        total_duration_ms = int(total_duration_seconds * 1000)
        
        num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
        
        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
            
            chunk_path = os.path.join(output_dir, f"chunk-{i:03d}.mp3")
            
            start_seconds = start_ms / 1000
            duration_seconds = (end_ms - start_ms) / 1000
            
            # Use ffmpeg to extract chunk
            cmd = [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-ss", str(start_seconds),
                "-t", str(duration_seconds),
                "-acodec", "libmp3lame",
                "-ab", "32k",  # Lower bitrate for speech
                "-ar", "16000",  # Lower sample rate
                "-ac", "1",  # Mono
                chunk_path,
            ]
            
            # Use run_in_executor for Windows compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True)
            )
            
            if result.returncode != 0:
                raise TranscriptionError(f"Failed to extract audio chunk {i}")
            
            chunks.append({
                "path": chunk_path,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "index": i,
            })
        
        return chunks

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        
        # Use run_in_executor for Windows compatibility
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True)
        )
        
        try:
            return float(result.stdout.decode().strip())
        except ValueError:
            raise TranscriptionError(f"Failed to get audio duration for: {audio_path}")


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    pass

