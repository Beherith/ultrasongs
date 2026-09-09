"""Pure pitch post-processing helpers.

The GPU-backed torchcrepe inference service will live in a separate module.
Keeping these functions independent makes the word/pitch aggregation contract
cheap to test and usable with frozen transcription fixtures.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def hz_to_midi(hz: float, *, fallback_midi: int = 60) -> int:
    """Convert a frequency to the nearest MIDI note."""

    if not math.isfinite(hz) or hz <= 0:
        return fallback_midi
    return round(12 * math.log2(hz / 440.0) + 69)


def _validate_pitch_arrays(
    times: FloatArray,
    frequencies: FloatArray,
    confidences: FloatArray,
) -> None:
    if not (len(times) == len(frequencies) == len(confidences)):
        raise ValueError("Pitch time, frequency, and confidence arrays must have equal lengths")


def get_midi_for_range(
    times: FloatArray,
    frequencies: FloatArray,
    confidences: FloatArray,
    start_sec: float,
    end_sec: float,
    *,
    confidence_thresholds: Sequence[float] = (0.5, 0.3, 0.1),
    fallback_midi: int = 60,
) -> int:
    """Return median pitch for a time range using descending confidence gates."""

    _validate_pitch_arrays(times, frequencies, confidences)
    if end_sec < start_sec:
        raise ValueError("Pitch range end must be greater than or equal to its start")

    for threshold in confidence_thresholds:
        mask = (
            (times >= start_sec)
            & (times <= end_sec)
            & (confidences > threshold)
            & (frequencies > 0)
        )
        selected = frequencies[mask]
        if len(selected):
            return hz_to_midi(float(np.median(selected)), fallback_midi=fallback_midi)
    return fallback_midi


def get_pitch_frames_for_range(
    times: FloatArray,
    frequencies: FloatArray,
    confidences: FloatArray,
    start_sec: float,
    end_sec: float,
    *,
    minimum_confidence: float = 0.1,
    fallback_midi: int = 60,
) -> list[dict[str, float | int]]:
    """Return serializable pitch frames within a word or syllable range."""

    _validate_pitch_arrays(times, frequencies, confidences)
    if end_sec < start_sec:
        raise ValueError("Pitch range end must be greater than or equal to its start")

    mask = (
        (times >= start_sec)
        & (times <= end_sec)
        & (confidences > minimum_confidence)
        & (frequencies > 0)
    )
    return [
        {
            "time": float(time),
            "midi": hz_to_midi(float(frequency), fallback_midi=fallback_midi),
            "confidence": float(confidence),
        }
        for time, frequency, confidence in zip(
            times[mask], frequencies[mask], confidences[mask], strict=True
        )
    ]


def attach_pitch_to_words(
    raw_words: Iterable[Mapping[str, Any]],
    times: FloatArray,
    frequencies: FloatArray,
    confidences: FloatArray,
    *,
    confidence_thresholds: Sequence[float] = (0.5, 0.3, 0.1),
    frame_minimum_confidence: float = 0.1,
    fallback_midi: int = 60,
) -> list[dict[str, Any]]:
    """Attach aggregate MIDI and detailed pitch frames to timestamped words.

    The output intentionally retains the legacy ``pitchFrames`` key so frozen
    transcription JSON remains directly comparable during migration.
    """

    words: list[dict[str, Any]] = []
    for raw_word in raw_words:
        try:
            start = float(raw_word["start"])
            end = float(raw_word["end"])
            word = str(raw_word["word"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each word must contain word, start, and end values") from exc

        words.append(
            {
                "word": word,
                "start": start,
                "end": end,
                "midi": get_midi_for_range(
                    times,
                    frequencies,
                    confidences,
                    start,
                    end,
                    confidence_thresholds=confidence_thresholds,
                    fallback_midi=fallback_midi,
                ),
                "pitchFrames": get_pitch_frames_for_range(
                    times,
                    frequencies,
                    confidences,
                    start,
                    end,
                    minimum_confidence=frame_minimum_confidence,
                    fallback_midi=fallback_midi,
                ),
            }
        )
    return words
