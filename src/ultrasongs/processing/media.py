"""FFmpeg-backed media normalization and inspection."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ultrasongs.config import FfmpegSettings

AUDIO_EXTENSIONS = frozenset({".mp3", ".ogg", ".flac", ".wav", ".m4a"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".avi"})
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class MediaProcessingError(RuntimeError):
    """Raised when FFmpeg or FFprobe cannot process a media artifact."""


def is_supported_media(filename: str | Path) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def is_video_media(filename: str | Path) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


class MediaService:
    """Invoke configured FFmpeg tools without shell interpolation."""

    def __init__(self, settings: FfmpegSettings) -> None:
        self.settings = settings

    def normalize_audio(self, input_path: Path, output_path: Path) -> Path:
        """Create the mono, constant-bitrate MP3 used by downstream stages."""

        source = Path(input_path).resolve()
        destination = Path(output_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input media not found: {source}")
        if not is_supported_media(source):
            raise ValueError(f"Unsupported media type: {source.suffix.lower() or '(none)'}")
        if source == destination:
            raise ValueError("Normalized output must differ from the input path")

        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.settings.executable,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(self.settings.channels),
            "-ar",
            str(self.settings.sample_rate_hz),
            "-c:a",
            self.settings.audio_codec,
            "-b:a",
            f"{self.settings.audio_bitrate_kbps}k",
            str(destination),
        ]
        self._run(command, operation="audio normalization")

        if not destination.is_file():
            raise MediaProcessingError(
                f"FFmpeg reported success but did not create {destination}"
            )
        return destination

    def probe_duration(self, media_path: Path) -> float:
        """Return media duration in seconds using FFprobe."""

        source = Path(media_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Media not found: {source}")
        command = [
            self.settings.ffprobe_executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ]
        result = self._run(command, operation="duration probe")
        try:
            duration = float(result.stdout.strip())
        except ValueError as exc:
            raise MediaProcessingError(
                f"FFprobe returned an invalid duration for {source}: {result.stdout!r}"
            ) from exc
        if duration < 0:
            raise MediaProcessingError(f"FFprobe returned a negative duration for {source}")
        return duration

    def _run(
        self,
        command: list[str],
        *,
        operation: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise MediaProcessingError(
                f"Executable not found while running {operation}: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaProcessingError(
                f"Timed out after {self.settings.timeout_seconds}s during {operation}"
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise MediaProcessingError(
                f"FFmpeg failed during {operation} with exit code "
                f"{result.returncode}: {detail}"
            )
        return result
