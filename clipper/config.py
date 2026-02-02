"""Global configuration management."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import yaml


# Available models by provider
AVAILABLE_MODELS = {
    "antigravity": {
        "claude-sonnet-4-5-thinking": "Claude Sonnet 4.5 with extended thinking",
        "claude-opus-4-5-thinking": "Claude Opus 4.5 with extended thinking", 
        "claude-sonnet-4-5": "Claude Sonnet 4.5 without thinking",
        "gemini-3-flash": "Gemini 3 Flash with thinking",
        "gemini-3-pro-low": "Gemini 3 Pro Low",
        "gemini-3-pro-high": "Gemini 3 Pro High",
    },
    "gemini": {
        "gemini-2.5-flash": "Fast Gemini model",
        "gemini-2.5-pro": "Pro Gemini model",
        "gemma-3-12b-it": "Gemma 3 12B Instruct",
    },
    "openai": {
        "gpt-4o-mini": "Fast GPT-4o",
        "gpt-4o": "Full GPT-4o",
        "gpt-4-turbo": "GPT-4 Turbo",
    }
}


@dataclass
class CaptionStyle:
    """Caption styling configuration."""
    font: str = "Komika Axis"
    fontsize: int = 72
    color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline_width: int = 4
    shadow_depth: int = 2
    margin_v: int = 400


@dataclass
class Config:
    """Global configuration for ClipperVX."""
    
    # Directories
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    temp_dir: Path = field(default_factory=lambda: Path("./temp"))
    
    # Video settings
    target_width: int = 1080
    target_height: int = 1920
    
    # Clip settings
    default_clips: int = 3
    min_clip_length: int = 15
    max_clip_length: int = 60
    
    # LLM settings
    llm_provider: Literal["gemini", "openai", "antigravity"] = "gemini"
    llm_model: str = "gemma-3-12b-it"
    openai_model: str = "gpt-4o-mini"
    
    # Whisper settings
    whisper_model: str = "base"
    
    # Caption styling
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    
    # FFmpeg settings
    ffmpeg_preset: str = "medium"
    ffmpeg_crf: int = 18
    audio_bitrate: str = "192k"
    
    # API Keys (from environment)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Extract caption style if present
        caption_data = {}
        for key in list(data.keys()):
            if key.startswith("caption_"):
                caption_key = key.replace("caption_", "")
                caption_data[caption_key] = data.pop(key)
        
        if caption_data:
            data["caption_style"] = CaptionStyle(**caption_data)
        
        # Filter out unknown keys
        valid_keys = {
            "output_dir", "temp_dir", "target_width", "target_height",
            "default_clips", "min_clip_length", "max_clip_length",
            "llm_provider", "llm_model", "openai_model", "whisper_model",
            "caption_style", "ffmpeg_preset", "ffmpeg_crf", "audio_bitrate"
        }
        data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**data)
    
    @classmethod
    def load_default(cls) -> "Config":
        """Load default configuration."""
        default_path = Path(__file__).parent.parent / "configs" / "defaults.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()
    
    def get_api_key(self) -> str:
        """Get API key based on provider."""
        if self.llm_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            return self.gemini_api_key
        else:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return self.openai_api_key
