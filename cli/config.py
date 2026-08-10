"""Configuration loading, validation, and frozen dataclass."""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments (// and /* */) and trailing commas so json.load can parse."""
    # Remove single-line comments
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # Remove multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


@dataclass(frozen=True)
class Config:
    """Frozen pipeline configuration."""

    whisper_model: str = "medium"
    demucs_model: str = "htdemucs"
    sample_rate: int = 44100
    pitch_min_hz: float = 65.41
    pitch_max_hz: float = 1046.5
    crepe_hop_ms: int = 10
    pause_min_silence_ms: int = 400
    pause_threshold_pct: float = 5.0
    gap_lead_in_ms: int = 500
    linebreak_beat_offset: int = 4
    ffmpeg_audio_bitrate: str = "128k"
    output_dir: str = "./output"
    temp_dir: str = "./tmp"
    debug_alignment: bool = False
    bpm_use_accompaniment: bool = False

    # Derived paths
    _config_path: Path = field(default=None, repr=False)  # type: ignore[assignment]

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir).resolve()


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from a JSONC file, falling back to defaults."""
    if config_path is None:
        # Default: look next to this module
        default_path = Path(__file__).parent / "config.jsonc"
    else:
        default_path = Path(config_path)

    if not default_path.exists():
        return Config()

    text = default_path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(_strip_jsonc(text))

    # Map JSONC keys to dataclass fields
    field_map = {
        "whisper_model": str,
        "demucs_model": str,
        "sample_rate": int,
        "pitch_min_hz": float,
        "pitch_max_hz": float,
        "crepe_hop_ms": int,
        "pause_min_silence_ms": int,
        "pause_threshold_pct": float,
        "gap_lead_in_ms": int,
        "linebreak_beat_offset": int,
        "ffmpeg_audio_bitrate": str,
        "output_dir": str,
        "temp_dir": str,
        "debug_alignment": bool,
        "bpm_use_accompaniment": bool,
    }

    kwargs: dict[str, Any] = {"_config_path": default_path}
    for key, typ in field_map.items():
        if key in data:
            try:
                kwargs[key] = typ(data[key])
            except (TypeError, ValueError) as exc:
                print(f"[config] Warning: invalid value for {key}: {data[key]!r} — using default", file=sys.stderr)

    return Config(**kwargs)
