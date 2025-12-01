"""
Caption Generator Service - Generates viral-style ASS subtitles with word-by-word highlighting.
"""

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

from app.config import CaptionStyle, get_settings
from app.services.transcription_service import TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


# Vertical positions in ASS format
POSITION_MAP = {
    "top": 8,
    "center": 5,
    "bottom": 2,
}

# ASS alignment values (numpad layout)
# 7 8 9 (top)
# 4 5 6 (middle)
# 1 2 3 (bottom)
ALIGNMENT_MAP = {
    "left": {"top": 7, "center": 4, "bottom": 1},
    "center": {"top": 8, "center": 5, "bottom": 2},
    "right": {"top": 9, "center": 6, "bottom": 3},
}


@dataclass
class WordGroup:
    """A group of words to display together."""
    
    words: list[TranscriptWord]
    start_time_ms: int
    end_time_ms: int
    text: str


class CaptionGeneratorService:
    """
    Service for generating viral-style ASS captions.
    
    Features:
    - Word-by-word highlighting (gold highlight on current word)
    - Configurable styling (font, color, position)
    - Support for transcript segments and word-level timing
    - Modern viral caption aesthetic
    """

    def __init__(self):
        self.settings = get_settings()

    async def generate_captions(
        self,
        transcript_segments: list[TranscriptSegment],
        clip_start_ms: int,
        clip_end_ms: int,
        output_path: str,
        caption_style: Optional[CaptionStyle] = None,
    ) -> Optional[str]:
        """
        Generate ASS captions for a clip.
        
        Args:
            transcript_segments: Transcript segments with timing
            clip_start_ms: Clip start time in milliseconds
            clip_end_ms: Clip end time in milliseconds
            output_path: Path to save the .ass file
            caption_style: Optional custom caption styling
            
        Returns:
            Path to generated .ass file, or None if no captions generated
        """
        style = caption_style or self.settings.get_caption_style()
        
        # Filter segments that overlap with this clip
        relevant_segments = [
            seg for seg in transcript_segments
            if seg.end_time_ms > clip_start_ms and seg.start_time_ms < clip_end_ms
        ]
        
        if not relevant_segments:
            logger.debug(f"No transcript segments found for clip {clip_start_ms}-{clip_end_ms}")
            return None
        
        # Check if we have word-level timing
        has_word_timing = any(
            seg.words and len(seg.words) > 0
            for seg in relevant_segments
        )
        
        if has_word_timing and style.word_by_word_highlight:
            # Generate word-by-word highlighted captions
            ass_content = self._generate_word_by_word_ass(
                relevant_segments, clip_start_ms, clip_end_ms, style
            )
        else:
            # Fallback to segment-based captions
            ass_content = self._generate_segment_ass(
                relevant_segments, clip_start_ms, clip_end_ms, style
            )
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        
        logger.debug(
            f"Generated ASS captions: {output_path} "
            f"(word_timing={has_word_timing}, segments={len(relevant_segments)})"
        )
        
        return output_path

    def _generate_word_by_word_ass(
        self,
        segments: list[TranscriptSegment],
        clip_start_ms: int,
        clip_end_ms: int,
        style: CaptionStyle,
    ) -> str:
        """Generate word-by-word highlighted ASS captions (viral style)."""
        header = self._generate_ass_header(style)
        events: list[str] = []
        
        # Collect all words from segments
        all_words: list[TranscriptWord] = []
        
        for segment in segments:
            if segment.words and len(segment.words) > 0:
                all_words.extend(segment.words)
            else:
                # Fallback: split segment text into pseudo-words with estimated timing
                words = self._split_segment_into_words(segment)
                all_words.extend(words)
        
        # Filter words that fall within clip bounds
        clip_words = [
            w for w in all_words
            if w.end_time_ms > clip_start_ms and w.start_time_ms < clip_end_ms
        ]
        
        if not clip_words:
            return header + self._generate_events_header()
        
        # Group words into display groups
        word_groups = self._group_words(clip_words, style.max_words_per_line)
        
        # Generate events for each word group with highlighting
        for group in word_groups:
            group_events = self._generate_word_group_events(
                group, clip_start_ms, style
            )
            events.extend(group_events)
        
        return header + self._generate_events_header() + "\n".join(events)

    def _generate_segment_ass(
        self,
        segments: list[TranscriptSegment],
        clip_start_ms: int,
        clip_end_ms: int,
        style: CaptionStyle,
    ) -> str:
        """Generate segment-based ASS captions (fallback)."""
        header = self._generate_ass_header(style)
        events: list[str] = []
        
        for segment in segments:
            # Clamp segment timing to clip bounds
            start_ms = max(0, segment.start_time_ms - clip_start_ms)
            end_ms = min(clip_end_ms - clip_start_ms, segment.end_time_ms - clip_start_ms)
            
            if end_ms <= start_ms:
                continue
            
            text = segment.text.strip()
            if style.uppercase:
                text = text.upper()
            
            # Split long text into lines
            lines = self._wrap_text(text, style.max_words_per_line)
            display_text = "\\N".join(lines)
            
            start_time = self._format_ass_time(start_ms)
            end_time = self._format_ass_time(end_ms)
            
            events.append(
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{display_text}"
            )
        
        return header + self._generate_events_header() + "\n".join(events)

    def _generate_ass_header(self, style: CaptionStyle) -> str:
        """Generate ASS header with style definitions."""
        primary_color = self._hex_to_ass(style.primary_color)
        highlight_color = self._hex_to_ass(style.highlight_color)
        outline_color = self._hex_to_ass(style.outline_color)
        shadow_color = self._hex_to_ass(style.shadow_color)
        
        alignment = ALIGNMENT_MAP[style.alignment][style.position]
        bold = -1 if style.bold else 0
        
        # Calculate vertical margin based on position (for 1080x1920)
        margin_v = {
            "top": 100,
            "center": 450,
            "bottom": 100,
        }[style.position]
        
        return f"""[Script Info]
Title: ViewCreator Viral Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{primary_color},{highlight_color},{outline_color},{shadow_color},{bold},0,0,0,100,100,0,0,1,{style.outline_width},2,{alignment},50,50,{margin_v},1
Style: Highlight,{style.font_name},{style.font_size},{highlight_color},{primary_color},{outline_color},{shadow_color},{bold},0,0,0,100,100,0,0,1,{style.outline_width},2,{alignment},50,50,{margin_v},1

"""

    def _generate_events_header(self) -> str:
        """Generate ASS events section header."""
        return "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    def _generate_word_group_events(
        self,
        group: WordGroup,
        clip_start_ms: int,
        style: CaptionStyle,
    ) -> list[str]:
        """Generate events for a word group with word-by-word highlighting."""
        events: list[str] = []
        words = group.words

        # For each word in the group, create an event showing all words
        # but with the current word highlighted
        for i, current_word in enumerate(words):
            word_start = max(0, current_word.start_time_ms - clip_start_ms)

            # FIX: To prevent overlapping dialogue events (double lines),
            # each word's end time should be the start of the next word.
            # This ensures seamless transitions without overlap.
            if i < len(words) - 1:
                # End this word's event when the next word starts
                next_word_start = max(0, words[i + 1].start_time_ms - clip_start_ms)
                word_end = next_word_start
            else:
                # Last word in group - use its actual end time
                word_end = max(0, current_word.end_time_ms - clip_start_ms)

            if word_end <= word_start:
                continue

            # Build display text with current word highlighted
            display_parts: list[str] = []

            for j, word in enumerate(words):
                word_text = word.word.strip()
                if style.uppercase:
                    word_text = word_text.upper()

                if j == i:
                    # Current word - use highlight style
                    display_parts.append(f"{{\\rHighlight}}{word_text}{{\\rDefault}}")
                else:
                    display_parts.append(word_text)

            # Wrap into lines
            lines = self._wrap_words(display_parts, style.max_words_per_line)
            display_text = "\\N".join(lines)

            start_time = self._format_ass_time(word_start)
            end_time = self._format_ass_time(word_end)

            events.append(
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{display_text}"
            )

        return events

    def _group_words(
        self,
        words: list[TranscriptWord],
        max_per_group: int,
    ) -> list[WordGroup]:
        """Group words into display groups."""
        groups: list[WordGroup] = []
        
        for i in range(0, len(words), max_per_group):
            group_words = words[i:min(i + max_per_group, len(words))]
            
            if not group_words:
                continue
            
            groups.append(WordGroup(
                words=group_words,
                start_time_ms=group_words[0].start_time_ms,
                end_time_ms=group_words[-1].end_time_ms,
                text=" ".join(w.word for w in group_words),
            ))
        
        return groups

    def _split_segment_into_words(
        self,
        segment: TranscriptSegment,
    ) -> list[TranscriptWord]:
        """Split segment text into pseudo-words with estimated timing."""
        text = segment.text.strip()
        word_strings = text.split()
        
        if not word_strings:
            return []
        
        duration = segment.end_time_ms - segment.start_time_ms
        word_duration = duration / len(word_strings)
        
        return [
            TranscriptWord(
                word=word,
                start_time_ms=int(segment.start_time_ms + i * word_duration),
                end_time_ms=int(segment.start_time_ms + (i + 1) * word_duration),
            )
            for i, word in enumerate(word_strings)
        ]

    def _wrap_text(self, text: str, max_words_per_line: int) -> list[str]:
        """Wrap text into lines by word count."""
        words = text.split()
        return self._wrap_words(words, max_words_per_line)

    def _wrap_words(self, words: list[str], max_words_per_line: int) -> list[str]:
        """Wrap word array into lines."""
        lines: list[str] = []
        
        for i in range(0, len(words), max_words_per_line):
            line_words = words[i:min(i + max_words_per_line, len(words))]
            lines.append(" ".join(line_words))
        
        return lines

    def _hex_to_ass(self, hex_color: str) -> str:
        """Convert hex color to ASS format (&HAABBGGRR)."""
        # Remove # if present
        clean = hex_color.lstrip("#")
        
        # Parse RGB
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
        
        # ASS uses &HAABBGGRR format (alpha, blue, green, red)
        return f"&H00{b:02X}{g:02X}{r:02X}"

    def _format_ass_time(self, ms: int) -> str:
        """Format milliseconds to ASS time format (H:MM:SS.CC)."""
        total_seconds = ms // 1000
        centiseconds = (ms % 1000) // 10
        seconds = total_seconds % 60
        minutes = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

