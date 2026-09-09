from __future__ import annotations

import numpy as np
import pytest

from ultrasongs.processing.pitch import (
    attach_pitch_to_words,
    get_midi_for_range,
    get_pitch_frames_for_range,
    hz_to_midi,
)


def test_hz_to_midi() -> None:
    assert hz_to_midi(440.0) == 69
    assert hz_to_midi(261.625565) == 60
    assert hz_to_midi(0) == 60


def test_midi_uses_lower_confidence_when_needed() -> None:
    times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    frequencies = np.array([440.0, 440.0, 440.0], dtype=np.float32)
    confidences = np.array([0.2, 0.25, 0.2], dtype=np.float32)

    assert get_midi_for_range(times, frequencies, confidences, 0.0, 0.4) == 69


def test_pitch_frames_preserve_legacy_shape() -> None:
    times = np.array([0.1, 0.2], dtype=np.float32)
    frequencies = np.array([440.0, 880.0], dtype=np.float32)
    confidences = np.array([0.5, 0.1], dtype=np.float32)

    frames = get_pitch_frames_for_range(times, frequencies, confidences, 0.0, 0.3)

    assert len(frames) == 1
    assert frames[0]["midi"] == 69


def test_attach_pitch_to_words() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    frequencies = np.full(4, 440.0, dtype=np.float32)
    confidences = np.full(4, 0.8, dtype=np.float32)

    result = attach_pitch_to_words(
        [{"word": " hello ", "start": 0.0, "end": 0.2}],
        times,
        frequencies,
        confidences,
    )

    assert result[0]["word"] == "hello"
    assert result[0]["midi"] == 69
    assert len(result[0]["pitchFrames"]) == 3


def test_pitch_arrays_must_have_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        get_midi_for_range(
            np.array([0.0]),
            np.array([440.0, 441.0]),
            np.array([0.9]),
            0.0,
            1.0,
        )
