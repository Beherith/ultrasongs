"""Tests for BPM, first-beat detection, and chunk stability check."""

import numpy as np
import pytest
import soundfile as sf

from cli.bpm_detect import CHUNK_DURATION_S, _chunk_bpms, detect_bpm
from cli.config import Config
from cli.pipeline_types import BpmResult

_SR = 22050


def _click_track(
    duration_s: float,
    bpm: float,
    first_beat_s: float = 0.0,
    sr: int = _SR,
) -> np.ndarray:
    """Synthesize a metronome click track at the given BPM."""
    y = np.zeros(int(duration_s * sr), dtype=np.float32)
    beat = 60.0 / bpm
    t = first_beat_s
    while t < duration_s - 0.05:
        idx = int(t * sr)
        if idx >= len(y):
            break
        length = min(int(0.02 * sr), len(y) - idx)
        env = np.exp(-3.0 * np.arange(length) / length)
        y[idx:idx + length] += (
            0.9 * np.sin(2 * np.pi * 800 * np.arange(length) / sr) * env
        )
        t += beat
    return y


def _write_wav(tmp_path, y: np.ndarray, name: str):
    path = tmp_path / name
    sf.write(str(path), y, _SR)
    return path


class TestDetectBpm:
    def test_constant_tempo_is_stable(self, tmp_path):
        y = _click_track(55, 120, first_beat_s=0.7)
        result = detect_bpm(_write_wav(tmp_path, y, "stable.wav"), Config())

        assert abs(result.bpm - 120) <= 5
        assert result.stable is True
        # 30 s first chunk + 25 s tail, both above the minimum chunk size
        assert len(result.chunk_bpms) == 2
        assert all(abs(c - result.bpm) <= 5 for c in result.chunk_bpms)

    def test_first_beat_ms(self, tmp_path):
        y = _click_track(55, 120, first_beat_s=0.7)
        result = detect_bpm(_write_wav(tmp_path, y, "first_beat.wav"), Config())

        assert 550 <= result.first_beat_ms <= 850

    def test_first_beat_at_start(self, tmp_path):
        y = _click_track(55, 120, first_beat_s=0.0)
        result = detect_bpm(_write_wav(tmp_path, y, "start.wav"), Config())

        assert result.first_beat_ms <= 50

    def test_changing_tempo_is_not_stable(self, tmp_path):
        y = np.zeros(int(80 * _SR), dtype=np.float32)
        y1 = _click_track(40, 120, first_beat_s=0.3)
        y2 = _click_track(40, 150, first_beat_s=0.0)
        y[: len(y1)] += y1
        y[len(y1):len(y1) + len(y2)] += y2
        result = detect_bpm(_write_wav(tmp_path, y, "mixed.wav"), Config())

        assert result.stable is False
        assert len(result.chunk_bpms) >= 2
        assert max(result.chunk_bpms) - min(result.chunk_bpms) > 5

    def test_fallback_on_error(self, tmp_path, monkeypatch):
        import librosa

        def _raise(*args, **kwargs):
            raise RuntimeError("no audio")

        monkeypatch.setattr(librosa, "load", _raise)
        path = tmp_path / "broken.mp3"
        path.write_bytes(b"not an audio file")

        result = detect_bpm(path, Config())

        assert result.bpm == 120.0
        assert result.first_beat_ms == 0.0
        assert result.stable is True
        assert result.chunk_bpms == []


class TestChunkBpms:
    def test_split_into_30s_chunks(self):
        y = _click_track(65, 120)
        estimates = _chunk_bpms(y, _SR)

        assert CHUNK_DURATION_S == 30.0
        # 30 s first chunk + 35 s second chunk (tail above the 10 s minimum)
        assert len(estimates) == 2
        assert all(abs(c - 120) <= 5 for c in estimates)

    def test_short_tail_is_skipped(self):
        y = _click_track(35, 120)
        estimates = _chunk_bpms(y, _SR)

        # The 5 s tail after the first 30 s chunk is below the minimum
        assert len(estimates) == 1
        assert abs(estimates[0] - 120) <= 5

    def test_empty_audio(self):
        assert _chunk_bpms(np.zeros(0, dtype=np.float32), _SR) == []


class TestBpmResult:
    def test_round_trip(self):
        result = BpmResult(
            bpm=120.5,
            first_beat_ms=735.0,
            stable=True,
            chunk_bpms=[118.2, 122.4],
        )
        restored = BpmResult.from_dict(result.to_dict())
        assert restored == result
