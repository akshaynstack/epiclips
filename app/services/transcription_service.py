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


# Sentence-ending punctuation marks
SENTENCE_END_PUNCTUATION = {'.', '!', '?', '。', '！', '？'}


def find_sentence_end_boundary(
    segments: list[TranscriptSegment],
    timestamp_ms: int,
    max_extension_ms: int = 5000,
    search_direction: str = "forward",
) -> int:
    """
    Find the nearest sentence boundary after a given timestamp.
    
    Uses word-level timing and punctuation detection to find natural
    sentence endings, preventing clips from cutting off mid-sentence.
    
    Args:
        segments: List of TranscriptSegment with word-level timing
        timestamp_ms: The timestamp to search from (in milliseconds)
        max_extension_ms: Maximum milliseconds to extend beyond timestamp (default 5000ms = 5s)
        search_direction: "forward" to find end after timestamp, "backward" to find before
        
    Returns:
        Adjusted timestamp in milliseconds at sentence boundary,
        or original timestamp if no boundary found within max_extension
    """
    if not segments:
        return timestamp_ms
    
    # Collect all words with their timing from segments that might contain our timestamp
    candidate_words: list[tuple[str, int]] = []  # (word, end_time_ms)
    
    for segment in segments:
        # Only consider segments that could contain words near our timestamp
        segment_start = segment.start_time_ms
        segment_end = segment.end_time_ms
        
        if search_direction == "forward":
            # For forward search, look at segments from timestamp onwards
            if segment_end < timestamp_ms:
                continue
            if segment_start > timestamp_ms + max_extension_ms:
                break
        else:
            # For backward search, look at segments before timestamp
            if segment_start > timestamp_ms:
                continue
            if segment_end < timestamp_ms - max_extension_ms:
                continue
        
        # If segment has word-level timing, use it
        if segment.words:
            for word in segment.words:
                candidate_words.append((word.word, word.end_time_ms))
        else:
            # Fall back to segment-level: treat segment text as ending at segment end
            # Check if segment text ends with sentence punctuation
            text = segment.text.strip()
            if text:
                candidate_words.append((text, segment_end))
    
    if not candidate_words:
        return timestamp_ms
    
    if search_direction == "forward":
        # Find first sentence boundary after timestamp within max_extension
        for word, end_time in candidate_words:
            if end_time < timestamp_ms:
                continue
            if end_time > timestamp_ms + max_extension_ms:
                # Exceeded max extension, return original
                logger.debug(
                    f"Sentence boundary: no boundary found within {max_extension_ms}ms of {timestamp_ms}ms, "
                    f"using original timestamp"
                )
                return timestamp_ms
            
            # Check if word ends with sentence punctuation
            word_stripped = word.rstrip()
            if word_stripped and word_stripped[-1] in SENTENCE_END_PUNCTUATION:
                logger.debug(
                    f"Sentence boundary: found end at {end_time}ms (word: '{word_stripped}'), "
                    f"extended from {timestamp_ms}ms by {end_time - timestamp_ms}ms"
                )
                return end_time
    else:
        # For backward search, find last sentence boundary before timestamp
        last_boundary = None
        for word, end_time in candidate_words:
            if end_time > timestamp_ms:
                continue
            if end_time < timestamp_ms - max_extension_ms:
                continue
            
            word_stripped = word.rstrip()
            if word_stripped and word_stripped[-1] in SENTENCE_END_PUNCTUATION:
                last_boundary = end_time
        
        if last_boundary is not None:
            logger.debug(
                f"Sentence boundary (backward): found end at {last_boundary}ms, "
                f"adjusted from {timestamp_ms}ms"
            )
            return last_boundary
    
    # No sentence boundary found, return original
    return timestamp_ms


def find_sentence_start_boundary(
    segments: list[TranscriptSegment],
    timestamp_ms: int,
    max_adjustment_ms: int = 3000,
) -> int:
    """
    Find the nearest sentence/word start boundary before a given timestamp.
    
    This prevents clips from starting mid-word by snapping the start time
    to the beginning of the word or sentence that contains the timestamp.
    
    Args:
        segments: List of TranscriptSegment with word-level timing
        timestamp_ms: The timestamp to search from (in milliseconds)
        max_adjustment_ms: Maximum milliseconds to adjust backward (default 3000ms = 3s)
        
    Returns:
        Adjusted timestamp in milliseconds at word/sentence start,
        or original timestamp if no suitable boundary found
    """
    if not segments:
        return timestamp_ms
    
    # Collect all words with their timing from relevant segments
    all_words: list[tuple[str, int, int]] = []  # (word, start_time_ms, end_time_ms)
    
    for segment in segments:
        segment_start = segment.start_time_ms
        segment_end = segment.end_time_ms
        
        # Only consider segments near our timestamp
        if segment_end < timestamp_ms - max_adjustment_ms:
            continue
        if segment_start > timestamp_ms + 1000:  # Small buffer to catch current word
            break
        
        if segment.words:
            for word in segment.words:
                all_words.append((word.word, word.start_time_ms, word.end_time_ms))
        else:
            # Fall back to segment-level
            all_words.append((segment.text, segment_start, segment_end))
    
    if not all_words:
        return timestamp_ms
    
    # Find the word that contains or is closest before the timestamp
    best_start = None
    best_is_sentence_start = False
    
    for i, (word, start_time, end_time) in enumerate(all_words):
        # Check if this word contains our timestamp (we're cutting mid-word)
        if start_time <= timestamp_ms <= end_time:
            # We're in the middle of this word - snap to its start
            logger.debug(
                f"Start boundary: timestamp {timestamp_ms}ms is mid-word '{word}', "
                f"snapping to word start at {start_time}ms"
            )
            return start_time
        
        # Check if this is just before our timestamp
        if start_time < timestamp_ms and end_time <= timestamp_ms:
            # Check if previous word ended a sentence (so this is a sentence start)
            if i > 0:
                prev_word = all_words[i - 1][0].rstrip()
                if prev_word and prev_word[-1] in SENTENCE_END_PUNCTUATION:
                    # This word starts a new sentence
                    if timestamp_ms - start_time <= max_adjustment_ms:
                        best_start = start_time
                        best_is_sentence_start = True
            
            # If not a sentence start but within range, consider it
            if not best_is_sentence_start and timestamp_ms - start_time <= max_adjustment_ms:
                best_start = start_time
    
    # Also check if timestamp falls between words
    for i, (word, start_time, end_time) in enumerate(all_words):
        if start_time > timestamp_ms:
            # This word starts after our timestamp - we should start at this word
            if start_time - timestamp_ms <= 500:  # Within 500ms after
                logger.debug(
                    f"Start boundary: timestamp {timestamp_ms}ms is between words, "
                    f"snapping to next word start at {start_time}ms"
                )
                return start_time
            break
    
    if best_start is not None:
        if best_is_sentence_start:
            logger.debug(
                f"Start boundary: found sentence start at {best_start}ms, "
                f"adjusted from {timestamp_ms}ms (-{timestamp_ms - best_start}ms)"
            )
        else:
            logger.debug(
                f"Start boundary: snapped to word start at {best_start}ms, "
                f"adjusted from {timestamp_ms}ms (-{timestamp_ms - best_start}ms)"
            )
        return best_start
    
    return timestamp_ms


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
        self._local_model = None
        self._init_client()

    def _init_client(self):
        """Initialize Groq client or local model."""
        if self.settings.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.settings.groq_api_key)
                logger.info("Groq client initialized for transcription")
            except ImportError:
                logger.warning("groq package not installed. Falling back to local Whisper if needed.")
        else:
            logger.info("GROQ_API_KEY not set. Using local Whisper by default.")

    def _get_local_model(self):
        """Lazy load the local faster-whisper model."""
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
                model_size = os.getenv("WHISPER_MODEL", "tiny") # use tiny for speed
                logger.info(f"Loading local Whisper model: {model_size}...")
                self._local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info("Local Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local Whisper model: {e}")
                raise TranscriptionError(f"Local Whisper initialization failed: {e}")
        return self._local_model

    async def transcribe(
        self,
        video_path: str,
        work_dir: str,
        language: Optional[str] = None,
        translate_to_english: bool = False,
        use_local: bool = False,
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
                use_local=use_local,
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
        use_local: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            language: Optional language code (auto-detected if not specified)
            translate_to_english: Whether to translate to English
            use_local: Force use of local faster-whisper
            
        Returns:
            TranscriptionResult with segments and word-level timing
        """
        if not os.path.isfile(audio_path):
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        if use_local or not self._groq_client:
            return await self._transcribe_local(audio_path, language)

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

    async def _transcribe_local(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe using local faster-whisper."""
        model = self._get_local_model()
        
        logger.info(f"Starting local transcription for: {audio_path}")
        
        # Run in executor to not block async loop
        def _sync_local_transcribe():
            segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True, language=language)
            
            all_segments = []
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
                
                transcript_words = []
                if segment.words:
                    for word in segment.words:
                        transcript_words.append(TranscriptWord(
                            word=word.word.strip(),
                            start_time_ms=int(word.start * 1000),
                            end_time_ms=int(word.end * 1000),
                        ))
                
                all_segments.append(TranscriptSegment(
                    start_time_ms=int(segment.start * 1000),
                    end_time_ms=int(segment.end * 1000),
                    text=segment.text.strip(),
                    words=transcript_words
                ))
            return all_segments, full_text.strip(), info

        loop = asyncio.get_event_loop()
        segments, full_text, info = await loop.run_in_executor(None, _sync_local_transcribe)
        
        logger.info(f"Local transcription complete: {len(segments)} segments")
        
        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=info.language,
            duration_seconds=info.duration,
            provider="local",
            model=os.getenv("WHISPER_MODEL", "tiny")
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

