"""FFmpeg PCM extraction: MP3 -> raw float32 PCM bytes."""

import shutil
import subprocess
from pathlib import Path

from cli.logging_setup import get_logger

logger = get_logger("cli.ffmpeg_pcm")


def extract_pcm(mp3_path: Path, sample_rate: int = 44100) -> bytes:
    """Extract raw float32 mono PCM from an MP3 via FFmpeg.

    Args:
        mp3_path: Source MP3 file.
        sample_rate: Target sample rate.

    Returns:
        Raw float32 PCM bytes.

    Raises:
        RuntimeError: If FFmpeg is missing or fails.
    """
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(mp3_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_f32le",
        "-f", "f32le",
        "-",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg PCM extraction failed (exit {result.returncode}):\n{result.stderr.decode('utf-8', errors='replace')}"
        )

    logger.debug(f"Extracted {len(result.stdout)} bytes of PCM ({sample_rate} Hz)")
    return result.stdout
