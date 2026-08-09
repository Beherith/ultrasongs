"""RMS-based vocal pause detection extracted from the legacy service."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def detect_pauses(
    vocals: FloatArray,
    sample_rate: int,
    *,
    frame_ms: int = 25,
    hop_ms: int = 10,
    minimum_silence_ms: int = 400,
    threshold_ratio: float = 0.05,
) -> list[dict[str, float]]:
    """Detect sustained low-RMS regions and return second-based ranges."""

    audio = np.asarray(vocals)
    if audio.ndim != 1:
        raise ValueError("Pause detection expects mono audio")
    if len(audio) == 0:
        return []
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if frame_ms <= 0 or hop_ms <= 0 or minimum_silence_ms <= 0:
        raise ValueError("Frame, hop, and minimum silence durations must be positive")
    if not 0 <= threshold_ratio <= 1:
        raise ValueError("Pause threshold ratio must be between 0 and 1")

    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    frame_count = max(1, (len(audio) - frame_length) // hop_length + 1)

    indices = np.arange(frame_count)[:, None] * hop_length + np.arange(frame_length)
    indices = np.clip(indices, 0, len(audio) - 1)
    rms = np.sqrt((audio[indices] ** 2).mean(axis=1))

    reference = float(np.percentile(rms, 95))
    if reference < 1e-6:
        return []

    silent = rms < threshold_ratio * reference
    minimum_frames = max(1, round(minimum_silence_ms / hop_ms))
    times = np.arange(frame_count, dtype=np.float32) * (hop_ms / 1000)

    pauses: list[dict[str, float]] = []
    in_silence = False
    silence_start = 0

    for index, is_silent in enumerate(silent):
        if is_silent and not in_silence:
            in_silence = True
            silence_start = index
        elif not is_silent and in_silence:
            in_silence = False
            if index - silence_start >= minimum_frames:
                pauses.append(
                    {"start": float(times[silence_start]), "end": float(times[index])}
                )

    if in_silence and frame_count - silence_start >= minimum_frames:
        pauses.append(
            {
                "start": float(times[silence_start]),
                "end": float(times[frame_count - 1]),
            }
        )

    return pauses
