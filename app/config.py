"""
Configuration module using Pydantic Settings for environment variable management.

Only essential environment variables are exposed. All other settings are hardcoded
for consistency and simplicity.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class CaptionStyle:
    """Caption styling configuration (hardcoded)."""

    font_name: str = "Arial Black"
    font_size: int = 72
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"  # Gold
    outline_color: str = "#000000"
    outline_width: int = 4
    shadow_color: str = "#000000"
    position: Literal["top", "center", "bottom"] = "center"
    max_words_per_line: int = 4
    word_by_word_highlight: bool = True
    alignment: Literal["left", "center", "right"] = "center"
    bold: bool = True
    uppercase: bool = True


# ============================================================
# LAYOUT PRESETS
# ============================================================

class LayoutType:
    """
    Available layout type identifiers for clip rendering.

    Users can select from these layouts to control how clips are composed.
    """
    AUTO = "auto"                    # AI-powered dynamic layout detection - switches mid-clip
    SPLIT_SCREEN = "split_screen"    # Screen content on top, face on bottom (50/50 split)
    TALKING_HEAD = "talking_head"    # Face-focused single crop that follows the speaker


def get_layout_preset(layout_id: str) -> dict:
    """
    Get layout configuration for a given layout ID.

    Args:
        layout_id: One of the LayoutType constants

    Returns:
        Layout configuration dict

    Raises:
        ValueError: If layout_id is not recognized
    """
    presets = {
        LayoutType.AUTO: {
            "id": LayoutType.AUTO,
            "name": "Auto (Recommended)",
            "description": "AI detects layout changes and switches dynamically mid-clip - best for mixed content",
            "screen_ratio": 0.50,  # Used when screen_share detected
            "face_ratio": 0.50,    # Used when screen_share detected
            "requires_face": False,
            "requires_screen": False,
            "dynamic": True,  # Indicates this uses SmartLayoutDetector
        },
        LayoutType.SPLIT_SCREEN: {
            "id": LayoutType.SPLIT_SCREEN,
            "name": "Split Screen",
            "description": "Screen content on top, face close-up on bottom - optimal for tutorials and screen recordings",
            "screen_ratio": 0.50,
            "face_ratio": 0.50,
            "requires_face": True,
            "requires_screen": True,
        },
        LayoutType.TALKING_HEAD: {
            "id": LayoutType.TALKING_HEAD,
            "name": "Talking Head",
            "description": "Dynamic face-focused crop that follows the speaker - perfect for podcasts and vlogs",
            "screen_ratio": 0.0,
            "face_ratio": 1.0,
            "requires_face": True,
            "requires_screen": False,
        },
    }

    if layout_id not in presets:
        valid_layouts = list(presets.keys())
        raise ValueError(f"Unknown layout type: {layout_id}. Valid layouts: {valid_layouts}")

    return presets[layout_id]


def get_available_layouts() -> list[dict]:
    """
    Get list of available layout presets with metadata.

    Returns:
        List of layout info dicts with id, name, description, and preview info
    """
    return [
        {
            "id": LayoutType.AUTO,
            "name": "Auto (Recommended)",
            "description": "AI detects layout changes and switches dynamically mid-clip - best for mixed content",
            "icon": "sparkles",  # lucide icon name for frontend
            "preview_layout": {"dynamic": True},
        },
        {
            "id": LayoutType.SPLIT_SCREEN,
            "name": "Split Screen",
            "description": "Screen content on top, face close-up on bottom - optimal for tutorials and screen recordings",
            "icon": "layout-grid",  # lucide icon name for frontend
            "preview_layout": {"top": "screen", "bottom": "face"},
        },
        {
            "id": LayoutType.TALKING_HEAD,
            "name": "Talking Head",
            "description": "Dynamic face-focused crop that follows the speaker - perfect for podcasts and vlogs",
            "icon": "user",
            "preview_layout": {"full": "face"},
        },
    ]


# ============================================================
# CAPTION PRESETS
# ============================================================

class CaptionPreset:
    """
    Available caption preset identifiers.

    Users can select from these presets instead of manually configuring styles.
    """
    VIRAL_GOLD = "viral_gold"
    CLEAN_WHITE = "clean_white"
    NEON_POP = "neon_pop"
    BOLD_BOXED = "bold_boxed"
    GRADIENT_GLOW = "gradient_glow"


def get_caption_preset(preset_id: str) -> CaptionStyle:
    """
    Get a CaptionStyle configuration for a given preset ID.

    Args:
        preset_id: One of the CaptionPreset constants

    Returns:
        Configured CaptionStyle for the preset

    Raises:
        ValueError: If preset_id is not recognized
    """
    presets = {
        CaptionPreset.VIRAL_GOLD: _create_viral_gold_style(),
        CaptionPreset.CLEAN_WHITE: _create_clean_white_style(),
        CaptionPreset.NEON_POP: _create_neon_pop_style(),
        CaptionPreset.BOLD_BOXED: _create_bold_boxed_style(),
        CaptionPreset.GRADIENT_GLOW: _create_gradient_glow_style(),
    }

    if preset_id not in presets:
        valid_presets = list(presets.keys())
        raise ValueError(f"Unknown caption preset: {preset_id}. Valid presets: {valid_presets}")

    return presets[preset_id]


def get_available_presets() -> list[dict]:
    """
    Get list of available caption presets with metadata.

    Returns:
        List of preset info dicts with id, name, description
    """
    return [
        {
            "id": CaptionPreset.VIRAL_GOLD,
            "name": "Viral Gold",
            "description": "Bold white text with gold word highlighting - classic viral TikTok style",
            "preview_colors": {"primary": "#FFFFFF", "highlight": "#FFD700"},
        },
        {
            "id": CaptionPreset.CLEAN_WHITE,
            "name": "Clean White",
            "description": "Minimal white text with subtle styling - professional and clean",
            "preview_colors": {"primary": "#FFFFFF", "highlight": "#E0E0E0"},
        },
        {
            "id": CaptionPreset.NEON_POP,
            "name": "Neon Pop",
            "description": "Vibrant cyan and magenta neon colors - gaming and entertainment",
            "preview_colors": {"primary": "#00FFFF", "highlight": "#FF00FF"},
        },
        {
            "id": CaptionPreset.BOLD_BOXED,
            "name": "Bold Boxed",
            "description": "High contrast white with yellow highlights - news and podcasts",
            "preview_colors": {"primary": "#FFFFFF", "highlight": "#FFFF00"},
        },
        {
            "id": CaptionPreset.GRADIENT_GLOW,
            "name": "Gradient Glow",
            "description": "Modern white with coral pink accents - trendy lifestyle content",
            "preview_colors": {"primary": "#FFFFFF", "highlight": "#FF6B6B"},
        },
    ]


def _create_viral_gold_style() -> CaptionStyle:
    """Viral Gold: Bold white + gold highlighting - classic viral TikTok style."""
    style = CaptionStyle()
    style.font_name = "Arial Black"
    style.font_size = 72
    style.primary_color = "#FFFFFF"
    style.highlight_color = "#FFD700"  # Gold
    style.outline_color = "#000000"
    style.outline_width = 4
    style.shadow_color = "#000000"
    style.position = "center"
    style.max_words_per_line = 4
    style.word_by_word_highlight = True
    style.alignment = "center"
    style.bold = True
    style.uppercase = True
    return style


def _create_clean_white_style() -> CaptionStyle:
    """Clean White: Minimal professional look with subtle highlighting."""
    style = CaptionStyle()
    style.font_name = "Arial"
    style.font_size = 64
    style.primary_color = "#FFFFFF"
    style.highlight_color = "#E0E0E0"  # Light gray (subtle)
    style.outline_color = "#333333"  # Dark gray
    style.outline_width = 2
    style.shadow_color = "#222222"
    style.position = "center"
    style.max_words_per_line = 5
    style.word_by_word_highlight = True
    style.alignment = "center"
    style.bold = True
    style.uppercase = False  # Mixed case for professional look
    return style


def _create_neon_pop_style() -> CaptionStyle:
    """Neon Pop: Vibrant cyan + magenta for gaming/entertainment."""
    style = CaptionStyle()
    style.font_name = "Arial Black"
    style.font_size = 76
    style.primary_color = "#00FFFF"  # Cyan
    style.highlight_color = "#FF00FF"  # Magenta
    style.outline_color = "#000000"
    style.outline_width = 5
    style.shadow_color = "#000066"  # Dark blue shadow for glow effect
    style.position = "center"
    style.max_words_per_line = 4
    style.word_by_word_highlight = True
    style.alignment = "center"
    style.bold = True
    style.uppercase = True
    return style


def _create_bold_boxed_style() -> CaptionStyle:
    """Bold Boxed: High contrast with yellow highlights for news/podcasts."""
    style = CaptionStyle()
    style.font_name = "Helvetica"
    style.font_size = 68
    style.primary_color = "#FFFFFF"
    style.highlight_color = "#FFFF00"  # Yellow
    style.outline_color = "#000000"
    style.outline_width = 6  # Thicker outline for boxed look
    style.shadow_color = "#000000"
    style.position = "bottom"  # News style typically at bottom
    style.max_words_per_line = 6
    style.word_by_word_highlight = True
    style.alignment = "center"
    style.bold = True
    style.uppercase = True
    return style


def _create_gradient_glow_style() -> CaptionStyle:
    """Gradient Glow: Modern trendy style with coral pink accents."""
    style = CaptionStyle()
    style.font_name = "Arial Black"
    style.font_size = 70
    style.primary_color = "#FFFFFF"
    style.highlight_color = "#FF6B6B"  # Coral pink
    style.outline_color = "#4A0080"  # Dark purple
    style.outline_width = 3
    style.shadow_color = "#2D004D"  # Deep purple shadow
    style.position = "center"
    style.max_words_per_line = 4
    style.word_by_word_highlight = True
    style.alignment = "center"
    style.bold = True
    style.uppercase = True
    return style


class Settings(BaseSettings):
    """
    Application settings.

    Only essential configuration is loaded from environment variables.
    All processing/rendering settings are hardcoded for consistency.
    """

    # ============================================================
    # ENVIRONMENT VARIABLES (minimal set)
    # ============================================================

    # Application
    app_name: str = "viewcreator-genesis"
    debug: bool = False
    log_level: str = "INFO"

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: str = "viewcreator-media"

    # API Keys (required)
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Security - API authentication
    genesis_api_key: Optional[str] = None  # API key for authenticating incoming requests
    genesis_webhook_secret: Optional[str] = None  # Secret for signing outgoing webhooks

    # yt-dlp Configuration
    ytdlp_proxy: Optional[str] = None  # Proxy URL (e.g., http://user:pass@host:port)

    # Performance tuning (configurable for ECS scaling)
    max_workers: int = 4  # Max concurrent jobs (set to vCPU count for optimal performance)
    max_render_workers: int = 3  # Max concurrent FFmpeg render processes

    # ============================================================
    # HARDCODED SETTINGS (not configurable via env vars)
    # ============================================================

    # Processing settings
    @property
    def frame_interval_seconds(self) -> float:
        return 2.0

    @property
    def max_concurrent_jobs(self) -> int:
        return self.max_workers  # Use configurable env var

    @property
    def max_concurrent_renders(self) -> int:
        return self.max_render_workers  # Use configurable env var

    @property
    def temp_directory(self) -> str:
        return "/tmp/genesis"

    @property
    def workspace_root(self) -> str:
        return "/tmp/ai-clipping-agent"

    # Detection settings
    @property
    def face_confidence_threshold(self) -> float:
        return 0.5

    @property
    def pose_confidence_threshold(self) -> float:
        return 0.5

    @property
    def tracking_max_age(self) -> int:
        return 30  # Frames before a track is deleted

    # API settings
    @property
    def api_timeout_seconds(self) -> int:
        return 300

    @property
    def max_video_duration_seconds(self) -> int:
        return 14400  # 4 hours max (credit-guarded in API)

    # yt-dlp Configuration
    @property
    def ytdlp_path(self) -> str:
        return "yt-dlp"

    @property
    def ytdlp_cookies_from_browser(self) -> Optional[str]:
        return None

    @property
    def max_download_duration_seconds(self) -> int:
        return 14400  # 4 hours max (credit-guarded in API)

    # Transcription Configuration (Groq Whisper)
    @property
    def transcription_model(self) -> str:
        return "whisper-large-v3-turbo"

    @property
    def groq_chunk_duration_seconds(self) -> int:
        return 1800  # 30 minutes

    # OpenRouter / LLM Configuration
    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def gemini_model(self) -> str:
        return "google/gemini-2.5-flash"

    # Clip Planning Configuration
    @property
    def max_suggested_clips(self) -> int:
        return 5

    @property
    def max_frames_per_vision_batch(self) -> int:
        return 48

    # Rendering Configuration
    @property
    def target_output_width(self) -> int:
        return 1080

    @property
    def target_output_height(self) -> int:
        return 1920

    @property
    def ffmpeg_preset(self) -> str:
        return "veryfast"

    @property
    def ffmpeg_crf(self) -> int:
        return 20

    # Split Layout (screen top, face bottom, captions overlaid)
    @property
    def split_screen_ratio(self) -> float:
        return 0.50  # Screen content: 50% (top)

    @property
    def split_face_ratio(self) -> float:
        return 0.50  # Speaker face: 50% (bottom)

    @property
    def use_split_layout(self) -> bool:
        return True  # Enable split layout for screen_share (screen top, face bottom)

    # Legacy Stack Layout (kept for fallback)
    @property
    def stack_screen_height_ratio(self) -> float:
        return 0.55

    @property
    def stack_face_height_ratio(self) -> float:
        return 0.45

    # Camera Physics (mass-spring-damper smoothing)
    @property
    def camera_mass(self) -> float:
        return 1.0

    @property
    def camera_stiffness(self) -> float:
        return 150.0

    @property
    def camera_damping(self) -> float:
        return 20.0

    # Caption styling
    @property
    def caption_font_name(self) -> str:
        return "Arial Black"

    @property
    def caption_font_size(self) -> int:
        return 72

    @property
    def caption_primary_color(self) -> str:
        return "#FFFFFF"

    @property
    def caption_highlight_color(self) -> str:
        return "#FFD700"

    @property
    def caption_position(self) -> Literal["top", "center", "bottom"]:
        return "center"

    @property
    def caption_max_words_per_line(self) -> int:
        return 4

    @property
    def caption_word_by_word_highlight(self) -> bool:
        return True

    @property
    def caption_uppercase(self) -> bool:
        return True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_caption_style(self) -> CaptionStyle:
        """Build CaptionStyle from settings."""
        style = CaptionStyle()
        style.font_name = self.caption_font_name
        style.font_size = self.caption_font_size
        style.primary_color = self.caption_primary_color
        style.highlight_color = self.caption_highlight_color
        style.position = self.caption_position
        style.max_words_per_line = self.caption_max_words_per_line
        style.word_by_word_highlight = self.caption_word_by_word_highlight
        style.uppercase = self.caption_uppercase
        return style

    def get_ytdlp_extra_args(self) -> list[str]:
        """Parse yt-dlp extra arguments (none by default)."""
        return []


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
