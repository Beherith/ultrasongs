"""Central, typed configuration for every UltraSongs component.

Configuration is resolved in the following order (last value wins): built-in
defaults, a JSON/TOML configuration file, environment variables, and validated
per-run UI overrides. Only fields explicitly marked as UI-editable below may be
overridden by a browser request.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

CONFIG_ENV_VAR = "ULTRASONGS_CONFIG"
ENV_PREFIX = "ULTRASONGS_"
SNAPSHOT_SCHEMA_VERSION = 1

def _ui_field(default: Any, *, title: str, description: str, **constraints: Any) -> Any:
    """Declare a safe per-run UI setting and retain its presentation metadata."""
    return Field(
        default,
        title=title,
        description=description,
        json_schema_extra={"ui_override": True},
        **constraints,
    )


class SettingsModel(BaseModel):
    """Immutable base model; unknown configuration is always an error."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ServerSettings(SettingsModel):
    host: str = "127.0.0.1"
    port: int = Field(8050, ge=1, le=65535)
    debug: bool = False
    worker_backend: Literal["threadpool"] = "threadpool"

    @field_validator("host")
    @classmethod
    def host_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("server host cannot be empty")
        return value.strip()


class PathSettings(SettingsModel):
    temp_dir: Path = Field(default="tmp")
    drafts_dir: Path = Field(default="drafts")
    projects_dir: Path = Field(default="projects")
    reports_dir: Path = Field(default="reports")
    exports_dir: Path = Field(default="exports")


class SecuritySettings(SettingsModel):
    max_upload_megabytes: int = Field(1024, ge=1, le=102_400)
    max_concurrent_jobs: int = Field(1, ge=1, le=64)
    allowed_audio_extensions: tuple[str, ...] = (".mp3", ".wav", ".flac", ".m4a", ".ogg")
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".webm", ".mov", ".avi")

    @field_validator("allowed_audio_extensions", "allowed_video_extensions")
    @classmethod
    def normalized_extensions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("at least one upload extension is required")
        normalized = tuple(
            value.lower() if value.startswith(".") else f".{value.lower()}" for value in values
        )
        if any(value == "." for value in normalized):
            raise ValueError("upload extensions cannot be empty")
        return normalized


class FfmpegSettings(SettingsModel):
    executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    audio_codec: str = "libmp3lame"
    # Compatibility defaults from the current Next.js upload pipeline.
    audio_bitrate_kbps: int = Field(128, ge=32, le=512)
    sample_rate_hz: int = Field(44_100, ge=8_000, le=192_000)
    channels: Literal[1, 2] = 1
    timeout_seconds: int = Field(600, ge=1, le=86_400)

    @field_validator("executable", "ffprobe_executable", "audio_codec")
    @classmethod
    def command_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("FFmpeg commands and codecs cannot be empty")
        return value.strip()


WhisperModelName = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
]
DeviceName = Literal["auto", "cpu", "cuda"]
ComputeType = Literal["auto", "int8", "int8_float16", "float16", "float32"]


class TranscriptionSettings(SettingsModel):
    engine: Literal["faster-whisper", "whisperx"] = _ui_field(
        "faster-whisper",
        title="Transcription engine",
        description="Word-timestamp engine used for this run.",
    )
    model: WhisperModelName = _ui_field(
        "medium", title="Whisper model", description="Whisper model size used for transcription."
    )
    language: str | None = _ui_field(
        None,
        title="Language override",
        description="BCP-47/Whisper language code, or blank for automatic detection.",
    )
    beam_size: int = _ui_field(
        5, title="Beam size", description="Whisper decoding beam size.", ge=1, le=20
    )
    vad_filter: bool = _ui_field(
        True,
        title="Voice activity filter",
        description="Apply Whisper's voice activity filter before decoding.",
    )
    device: DeviceName = "auto"
    compute_type: ComputeType = "auto"

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower() or None

    @model_validator(mode="after")
    def validate_compute_device(self) -> TranscriptionSettings:
        if self.device == "cpu" and self.compute_type in {"float16", "int8_float16"}:
            raise ValueError(f"compute type {self.compute_type!r} requires a CUDA device")
        return self


class WhisperXSettings(SettingsModel):
    model: WhisperModelName = _ui_field(
        "small", title="WhisperX model", description="Model used when WhisperX is selected."
    )
    batch_size: int = _ui_field(
        4,
        title="WhisperX batch size",
        description="Number of chunks aligned per batch.",
        ge=1,
        le=64,
    )
    device: DeviceName = "cpu"
    compute_type: ComputeType = "int8"
    python_executable: str = ".venv-whisperx/Scripts/python.exe"

    @field_validator("python_executable")
    @classmethod
    def python_command_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("WhisperX Python executable cannot be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_compute_device(self) -> WhisperXSettings:
        if self.device == "cpu" and self.compute_type in {"float16", "int8_float16"}:
            raise ValueError(f"compute type {self.compute_type!r} requires a CUDA device")
        return self


class SeparationSettings(SettingsModel):
    model: Literal["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"] = _ui_field(
        "htdemucs", title="Separation model", description="Demucs source-separation model."
    )
    shifts: int = _ui_field(
        1,
        title="Separation shifts",
        description="Number of random equivariant shifts.",
        ge=0,
        le=10,
    )
    overlap: float = _ui_field(
        0.25,
        title="Segment overlap",
        description="Overlap between separated chunks.",
        ge=0.0,
        lt=1.0,
    )
    device: DeviceName = "auto"


class PitchSettings(SettingsModel):
    model: Literal["tiny", "full"] = _ui_field(
        "full", title="Pitch model", description="torchcrepe model used for pitch estimation."
    )
    sample_rate_hz: int = Field(16_000, ge=8_000, le=96_000)
    hop_length: int = Field(160, ge=1, le=8192)
    # C2-C6, matching the legacy torchcrepe pass.
    min_frequency_hz: float = Field(65.41, gt=0, le=2_000)
    max_frequency_hz: float = Field(1_046.5, gt=0, le=8_000)
    batch_size: int = Field(2048, ge=1, le=65_536)
    confidence_thresholds: tuple[float, ...] = _ui_field(
        (0.5, 0.3, 0.1),
        title="Pitch confidence thresholds",
        description="Confidence fallbacks, tried from strictest to loosest.",
    )
    device: DeviceName = "auto"

    @field_validator("confidence_thresholds")
    @classmethod
    def validate_thresholds(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("at least one pitch confidence threshold is required")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("pitch confidence thresholds must be between 0 and 1")
        if any(left <= right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("pitch confidence thresholds must be strictly descending")
        return values

    @model_validator(mode="after")
    def validate_frequency_range(self) -> PitchSettings:
        if self.min_frequency_hz >= self.max_frequency_hz:
            raise ValueError("minimum pitch frequency must be below maximum pitch frequency")
        return self


class PauseSettings(SettingsModel):
    frame_milliseconds: int = Field(25, ge=5, le=1_000)
    hop_milliseconds: int = Field(10, ge=1, le=1_000)
    threshold_ratio: float = _ui_field(
        0.05,
        title="Pause threshold",
        description="RMS level relative to the track's 95th percentile.",
        gt=0,
        lt=1,
    )
    minimum_duration_milliseconds: int = _ui_field(
        400,
        title="Minimum pause duration",
        description="Shortest silence retained as a pause.",
        ge=0,
        le=10_000,
    )


class AlignmentSettings(SettingsModel):
    engine: Literal["smith-waterman"] = _ui_field(
        "smith-waterman",
        title="Lyrics alignment engine",
        description="Deterministic algorithm used to align supplied lyrics.",
    )
    match_score: float = Field(4.0, gt=0)
    gap_open_penalty: float = Field(4.0, gt=0)
    gap_extend_penalty: float = Field(0.5, gt=0)


class TempoSettings(SettingsModel):
    fallback_bpm: float = _ui_field(
        120.0,
        title="Fallback BPM",
        description="Tempo used when automatic detection cannot produce a result.",
        ge=20,
        le=400,
    )
    minimum_bpm: float = Field(40.0, ge=20, le=400)
    maximum_bpm: float = Field(240.0, ge=20, le=400)

    @model_validator(mode="after")
    def validate_bpm_range(self) -> TempoSettings:
        if self.minimum_bpm >= self.maximum_bpm:
            raise ValueError("minimum BPM must be below maximum BPM")
        return self


class ExportSettings(SettingsModel):
    text_encoding: Literal["utf-8", "utf-8-sig"] = "utf-8"
    include_audio_in_zip: bool = True
    bpm_decimal_places: int = Field(2, ge=0, le=6)


class ValidationSettings(SettingsModel):
    minimum_matched_notes: int = _ui_field(
        1,
        title="Minimum matched notes",
        description="Fewest matched notes required for a valid comparison.",
        ge=1,
    )
    minimum_match_ratio: float = _ui_field(
        0.5,
        title="Minimum match ratio",
        description="Minimum fraction of reference notes that must match.",
        ge=0,
        le=1,
    )
    maximum_timing_rmse_ms: float = _ui_field(
        500.0,
        title="Maximum timing RMSE",
        description="Largest accepted note-start RMSE in milliseconds.",
        ge=0,
    )
    maximum_duration_rmse_ms: float = _ui_field(
        500.0,
        title="Maximum duration RMSE",
        description="Largest accepted duration RMSE in milliseconds.",
        ge=0,
    )
    maximum_pitch_distance_semitones: float = _ui_field(
        2.0,
        title="Maximum pitch distance",
        description="Largest accepted octave-corrected pitch distance.",
        ge=0,
    )


class ReportSettings(SettingsModel):
    include_pitch_frames: bool = _ui_field(
        True,
        title="Include pitch frames",
        description="Render raw pitch/confidence frames in the validation report.",
    )
    include_pauses: bool = _ui_field(
        True, title="Include pauses", description="Render detected pauses in the report."
    )
    # Reserved for the later media-rich report phase. Keep these centralized,
    # but do not expose controls until the report renderer implements them.
    include_stem_links: bool = True
    embed_audio: bool = False


class AppSettings(SettingsModel):
    """Complete application settings, loaded once and injected into services."""

    config_version: Literal[1] = 1
    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ffmpeg: FfmpegSettings = Field(default_factory=FfmpegSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    whisperx: WhisperXSettings = Field(default_factory=WhisperXSettings)
    separation: SeparationSettings = Field(default_factory=SeparationSettings)
    pitch: PitchSettings = Field(default_factory=PitchSettings)
    pauses: PauseSettings = Field(default_factory=PauseSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    tempo: TempoSettings = Field(default_factory=TempoSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)

    def apply_ui_overrides(self, overrides: Mapping[str, Any] | None) -> AppSettings:
        """Return a new settings object containing safe, per-run browser overrides."""
        flattened = _flatten_mapping(overrides or {})
        allowed = self.ui_override_paths()
        rejected = sorted(set(flattened) - allowed)
        if rejected:
            joined = ", ".join(rejected)
            raise ValueError(f"settings are not UI-overridable or do not exist: {joined}")

        values = self.model_dump(mode="python", round_trip=True)
        for path, value in flattened.items():
            _set_dotted_value(values, path, value)
        return AppSettings.model_validate(values)

    @classmethod
    def ui_override_paths(cls) -> frozenset[str]:
        """Return the authoritative whitelist of safe dotted UI setting paths."""
        return frozenset(_ui_option_schema(cls))

    def ui_override_schema(self) -> dict[str, dict[str, JsonValue]]:
        """Describe safe controls so Dash can generate its Advanced Settings panel."""
        options = _ui_option_schema(type(self))
        defaults = self.model_dump(mode="json")
        for path, metadata in options.items():
            metadata["default"] = _get_dotted_value(defaults, path)
        return options

    def effective_snapshot(
        self, overrides: Mapping[str, Any] | None = None
    ) -> EffectiveSettingsSnapshot:
        """Create a reproducible, immutable snapshot for a pipeline run."""
        flattened = _flatten_mapping(overrides or {})
        effective = self.apply_ui_overrides(flattened)
        json_overrides = json.loads(json.dumps(flattened, default=str))
        return EffectiveSettingsSnapshot(settings=effective, ui_overrides=json_overrides)


class EffectiveSettingsSnapshot(SettingsModel):
    """Versioned configuration document persisted beside run artifacts."""

    schema_version: Literal[SNAPSHOT_SCHEMA_VERSION] = SNAPSHOT_SCHEMA_VERSION
    settings: AppSettings
    ui_overrides: dict[str, JsonValue] = Field(default_factory=dict)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, value: str | bytes) -> EffectiveSettingsSnapshot:
        return cls.model_validate_json(value)

    def write(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, source: str | Path) -> EffectiveSettingsSnapshot:
        return cls.from_json(Path(source).read_text(encoding="utf-8"))


LEGACY_ENVIRONMENT_PATHS: dict[str, str] = {
    "TMP_DIR": "paths.temp_dir",
    "DRAFTS_DIR": "paths.drafts_dir",
    "WHISPER_MODEL": "transcription.model",
    "ALIGN_ENGINE": "transcription.engine",
    "WHISPERX_MODEL": "whisperx.model",
    "WHISPERX_DEVICE": "whisperx.device",
    "WHISPERX_COMPUTE_TYPE": "whisperx.compute_type",
    "WHISPERX_BATCH_SIZE": "whisperx.batch_size",
    "WHISPERX_PYTHON": "whisperx.python_executable",
    "WHISPERX_LANGUAGE": "transcription.language",
}


def load_settings(
    config_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load and validate application settings without mutating process state.

    Namespaced variables use ``ULTRASONGS_<GROUP>__<FIELD>``. Existing legacy
    variables such as ``TMP_DIR`` and ``WHISPER_MODEL`` remain mapped during the
    migration. Namespaced variables take precedence over their legacy aliases.
    """
    environment = os.environ if environ is None else environ
    values: dict[str, Any] = AppSettings().model_dump(mode="python", round_trip=True)

    selected_file = config_file or environment.get(CONFIG_ENV_VAR)
    if selected_file:
        values = _deep_merge(values, _load_config_file(Path(selected_file)))

    environment_values: dict[str, Any] = {}
    for variable, path in LEGACY_ENVIRONMENT_PATHS.items():
        if variable in environment:
            _set_dotted_value(
                environment_values,
                path,
                _parse_environment_value(environment[variable]),
            )

    for variable, raw_value in environment.items():
        if not variable.startswith(ENV_PREFIX) or variable == CONFIG_ENV_VAR:
            continue
        path = variable[len(ENV_PREFIX) :].lower().replace("__", ".")
        if not path:
            continue
        _set_dotted_value(environment_values, path, _parse_environment_value(raw_value))

    return AppSettings.model_validate(_deep_merge(values, environment_values))


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    else:
        raise ValueError(f"unsupported configuration format {suffix!r}; use .json or .toml")
    if not isinstance(document, dict):
        raise ValueError("configuration file must contain an object/table")
    if "ultrasongs" in document:
        document = document["ultrasongs"]
        if not isinstance(document, dict):
            raise ValueError("the [ultrasongs] configuration value must be a table")
    return document


def _parse_environment_value(value: str) -> Any:
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _flatten_mapping(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        key = str(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, path))
        else:
            flattened[path] = value
    return flattened


def _set_dotted_value(values: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = values
    for part in parts[:-1]:
        current = target.get(part)
        if not isinstance(current, dict):
            current = {}
            target[part] = current
        target = current
    target[parts[-1]] = value


def _get_dotted_value(values: Mapping[str, Any], path: str) -> Any:
    current: Any = values
    for part in path.split("."):
        current = current[part]
    return current


def _ui_option_schema(model: type[BaseModel], prefix: str = "") -> dict[str, dict[str, JsonValue]]:
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})
    options: dict[str, dict[str, JsonValue]] = {}

    def visit(node: Mapping[str, Any], current_prefix: str) -> None:
        if "$ref" in node:
            ref_name = str(node["$ref"]).rsplit("/", 1)[-1]
            visit(definitions[ref_name], current_prefix)
            return
        for name, field_schema in node.get("properties", {}).items():
            path = f"{current_prefix}.{name}" if current_prefix else name
            if field_schema.get("ui_override") is True:
                metadata = {
                    key: deepcopy(value)
                    for key, value in field_schema.items()
                    if key not in {"default", "ui_override"}
                }
                options[path] = metadata
                continue
            visit(field_schema, path)

    visit(schema, prefix)
    return options


__all__ = [
    "AlignmentSettings",
    "AppSettings",
    "EffectiveSettingsSnapshot",
    "ExportSettings",
    "FfmpegSettings",
    "PathSettings",
    "PauseSettings",
    "PitchSettings",
    "ReportSettings",
    "SecuritySettings",
    "SeparationSettings",
    "ServerSettings",
    "TempoSettings",
    "TranscriptionSettings",
    "ValidationError",
    "ValidationSettings",
    "WhisperXSettings",
    "load_settings",
]
