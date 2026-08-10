"""FFmpeg audio extraction: video/audio -> mono MP3."""

import shutil
import subprocess
from pathlib import Path

from cli.config import Config
from cli.logging_setup import get_logger

logger = get_logger("cli.ffmpeg_extract")

# Supported input extensions
_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
_AUDIO_EXTS = {".mp3", ".ogg", ".flac", ".wav", ".m4a"}
_SUPPORTED_EXTS = _VIDEO_EXTS | _AUDIO_EXTS


def extract_audio(input_path: Path, output_path: Path, config: Config) -> Path:
    """Extract and normalize audio to mono 128kbps MP3 via FFmpeg.

    Args:
        input_path: Source audio/video file.
        output_path: Destination MP3 path.
        config: Pipeline configuration.

    Returns:
        The output_path (for chaining).

    Raises:
        ValueError: If input extension is unsupported.
        RuntimeError: If FFmpeg is missing or fails.
        FileNotFoundError: If input file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_path.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file extension: {ext!r}. "
            f"Supported: {sorted(_SUPPORTED_EXTS)}"
        )

    # Check FFmpeg availability
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not found in PATH")

    logger.info(f"Extracting audio: {input_path} -> {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",                  # overwrite output
        "-i", str(input_path),
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-b:a", config.ffmpeg_audio_bitrate,
        "-codec:a", "libmp3lame",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )

    logger.info(f"Audio extracted: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path
