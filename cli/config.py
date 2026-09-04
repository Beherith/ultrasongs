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

    device: str = "auto"
    transcription_backend: str = "faster-whisper"
    whisper_model: str = "medium"
    whisper_language: str = "en"
    faster_whisper_compute_type: str = "auto"
    whisperx_batch_size: int = 8
    whisperx_compute_type: str = "default"
    whisperx_align_model: str = ""
    whisperx_interpolate_method: str = "nearest"
    whisperx_chunk_pause_ms: int = 1000
    whisperx_align_runs: int = 3
    transcribe_runs: int = 3
    demucs_model: str = "htdemucs"
    sample_rate: int = 44100
    pitch_min_hz: float = 65.41
    pitch_max_hz: float = 1046.5
    crepe_hop_ms: int = 10
    band_energy_min_hz: float = 60.0
    band_energy_max_hz: float = 4000.0
    pause_min_silence_ms: int = 400
    pause_threshold_pct: float = 5.0
    gap_lead_in_ms: int = 500
    linebreak_beat_offset: int = 4
    beat_resolution_multiplier: int = 2
    activity_quiet_confidence: float = 0.2
    activity_voiced_confidence: float = 0.5
    activity_noise_percentile: float = 0.9
    activity_noise_fallback_percentile: float = 0.1
    activity_signal_percentile: float = 0.5
    activity_signal_fallback_percentile: float = 0.75
    activity_threshold_ratio: float = 0.2
    note_min_confidence: float = 0.3
    note_fallback_confidence: float = 0.5
    note_dropout_gap_ms: int = 50
    note_smooth_window: int = 5
    note_pitch_tolerance: int = 1
    note_min_duration_ms: int = 60
    note_frame_step_ms: int = 10
    note_segment_plots: bool = False
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
        "device": str,
        "transcription_backend": str,
        "whisper_model": str,
        "whisper_language": str,
        "faster_whisper_compute_type": str,
        "whisperx_batch_size": int,
        "whisperx_compute_type": str,
        "whisperx_align_model": str,
        "whisperx_interpolate_method": str,
        "whisperx_chunk_pause_ms": int,
        "whisperx_align_runs": int,
        "transcribe_runs": int,
        "demucs_model": str,
        "sample_rate": int,
        "pitch_min_hz": float,
        "pitch_max_hz": float,
        "crepe_hop_ms": int,
        "band_energy_min_hz": float,
        "band_energy_max_hz": float,
        "pause_min_silence_ms": int,
        "pause_threshold_pct": float,
        "gap_lead_in_ms": int,
        "linebreak_beat_offset": int,
        "beat_resolution_multiplier": int,
        "activity_quiet_confidence": float,
        "activity_voiced_confidence": float,
        "activity_noise_percentile": float,
        "activity_noise_fallback_percentile": float,
        "activity_signal_percentile": float,
        "activity_signal_fallback_percentile": float,
        "activity_threshold_ratio": float,
        "note_min_confidence": float,
        "note_fallback_confidence": float,
        "note_dropout_gap_ms": int,
        "note_smooth_window": int,
        "note_pitch_tolerance": int,
        "note_min_duration_ms": int,
        "note_frame_step_ms": int,
        "note_segment_plots": bool,
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

    defaults = Config()
    if kwargs.get("transcription_backend", defaults.transcription_backend) not in {
        "faster-whisper", "whisperx",
    }:
        print("[config] Warning: invalid transcription_backend — using default", file=sys.stderr)
        kwargs.pop("transcription_backend", None)
    if kwargs.get("faster_whisper_compute_type", defaults.faster_whisper_compute_type) not in {
        "auto", "default", "float16", "float32", "int8", "int8_float16",
    }:
        print("[config] Warning: invalid faster_whisper_compute_type — using default", file=sys.stderr)
        kwargs.pop("faster_whisper_compute_type", None)
    if kwargs.get("whisperx_batch_size", defaults.whisperx_batch_size) < 1:
        print("[config] Warning: whisperx_batch_size must be at least 1 — using default", file=sys.stderr)
        kwargs.pop("whisperx_batch_size", None)
    if kwargs.get("whisperx_compute_type", defaults.whisperx_compute_type) not in {
        "default", "float16", "float32", "int8",
    }:
        print("[config] Warning: invalid whisperx_compute_type — using default", file=sys.stderr)
        kwargs.pop("whisperx_compute_type", None)
    if kwargs.get("whisperx_interpolate_method", defaults.whisperx_interpolate_method) not in {
        "nearest", "linear", "ignore",
    }:
        print("[config] Warning: invalid whisperx_interpolate_method — using default", file=sys.stderr)
        kwargs.pop("whisperx_interpolate_method", None)
    if kwargs.get("whisperx_chunk_pause_ms", defaults.whisperx_chunk_pause_ms) < 0:
        print("[config] Warning: whisperx_chunk_pause_ms must be non-negative — using default", file=sys.stderr)
        kwargs.pop("whisperx_chunk_pause_ms", None)
    whisperx_align_runs = kwargs.get("whisperx_align_runs", defaults.whisperx_align_runs)
    if whisperx_align_runs < 1 or whisperx_align_runs % 2 == 0:
        print("[config] Warning: whisperx_align_runs must be a positive odd number — using default", file=sys.stderr)
        kwargs.pop("whisperx_align_runs", None)

    return Config(**kwargs)
