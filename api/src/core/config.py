from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)
from pathlib import Path
from typing import Annotated

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[3] / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    try:
        return _pkg_version("kokoro-fastapi")
    except PackageNotFoundError:
        return "0.0.0"


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = _read_version()
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )

    # Model lifecycle
    model_ttl: int = 300  # Seconds of idle before unloading model from GPU. -1 = never unload, 0 = unload immediately after each request

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    # CORS origins allowed to call the API from a browser. Default "*" works
    # only when cors_allow_credentials is False (the wildcard+credentials combo
    # is forbidden by the CORS spec). NoDecode lets us accept a friendly
    # comma-separated string (CORS_ORIGINS=https://a.com,http://b:5173) in
    # addition to a JSON list, rather than pydantic's default JSON-only parsing.
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    cors_enabled: bool = True  # Whether to enable CORS
    cors_allow_credentials: bool = (
        True  # Send Access-Control-Allow-Credentials. Requires explicit origins (not "*").
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v):
        """Accept a comma-separated string or a JSON/list value for CORS_ORIGINS."""
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):  # JSON list form, let pydantic handle it
                import json

                return json.loads(s)
            return [o.strip() for o in s.split(",") if o.strip()]
        return v

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    class Config:
        env_file = ".env"

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
