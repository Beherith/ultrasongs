from __future__ import annotations

import numpy as np
import pytest

from ultrasongs.processing.pauses import detect_pauses


def test_detects_sustained_silence_between_loud_regions() -> None:
    sample_rate = 1000
    audio = np.concatenate(
        [
            np.full(500, 0.5, dtype=np.float32),
            np.zeros(500, dtype=np.float32),
            np.full(500, 0.5, dtype=np.float32),
        ]
    )

    pauses = detect_pauses(
        audio,
        sample_rate,
        frame_ms=20,
        hop_ms=10,
        minimum_silence_ms=300,
    )

    assert len(pauses) == 1
    assert pauses[0]["start"] == pytest.approx(0.5, abs=0.02)
    assert pauses[0]["end"] == pytest.approx(1.0, abs=0.02)


def test_silent_track_returns_no_contextual_pauses() -> None:
    assert detect_pauses(np.zeros(1000, dtype=np.float32), 1000) == []


def test_rejects_multichannel_audio() -> None:
    with pytest.raises(ValueError, match="mono"):
        detect_pauses(np.zeros((2, 100), dtype=np.float32), 1000)
