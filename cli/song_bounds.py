"""Song boundary detection: first and last audible time slot of the full track."""

from pathlib import Path

from cli.logging_setup import get_logger

logger = get_logger("cli.song_bounds")

FRAME_MS = 100
HOP_MS = 10
THRESHOLD_PERCENTILE = 5.0


def detect_song_bounds(audio_path: Path) -> tuple[float, float]:
    """Return (start_sec, end_sec) of the track's audible region.

    start_sec is the first frame whose RMS energy goes above the 5th
    percentile of all frame RMS values; end_sec is the end of the last such
    frame.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return 0.0, 0.0

    frame_len = max(1, int(sr * FRAME_MS / 1000))
    hop_len = max(1, int(sr * HOP_MS / 1000))
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    idx = np.arange(n_frames)[:, None] * hop_len + np.arange(frame_len)
    idx = np.clip(idx, 0, len(audio) - 1)
    rms = np.sqrt((audio[idx] ** 2).mean(axis=1))

    threshold = float(np.percentile(rms, THRESHOLD_PERCENTILE))
    active = np.nonzero(rms > threshold)[0]
    if len(active) == 0:
        return 0.0, len(audio) / sr

    start_sec = float(active[0] * hop_len / sr)
    end_sec = float((active[-1] * hop_len + frame_len) / sr)
    logger.info(f"Song bounds: start {start_sec:.3f} s, end {end_sec:.3f} s")
    return start_sec, end_sec
