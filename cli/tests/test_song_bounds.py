"""Tests for song bound detection (#START/#END tags)."""

import numpy as np
import pytest
import soundfile as sf

from cli.song_bounds import detect_song_bounds


def _write_wav(path, audio, sr=44100):
    sf.write(str(path), audio.astype(np.float32), sr)
    return path


class TestDetectSongBounds:
    def test_silence_intro_and_outro(self, tmp_path):
        sr = 44100
        t = np.arange(sr * 10) / sr
        audio = np.zeros_like(t)
        mask = (t >= 2.0) & (t < 8.0)
        audio[mask] = 0.5 * np.sin(2 * np.pi * 440 * t[mask])
        path = _write_wav(tmp_path / "song.wav", audio, sr)

        start, end = detect_song_bounds(path)
        assert start == pytest.approx(2.0, abs=0.15)
        assert end == pytest.approx(8.0, abs=0.15)

    def test_no_silence(self, tmp_path):
        sr = 44100
        t = np.arange(sr * 3) / sr
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        path = _write_wav(tmp_path / "full.wav", audio, sr)

        start, end = detect_song_bounds(path)
        assert start == pytest.approx(0.0, abs=0.15)
        assert end == pytest.approx(3.0, abs=0.15)

    def test_fully_silent(self, tmp_path):
        path = _write_wav(tmp_path / "silence.wav", np.zeros(44100))
        start, end = detect_song_bounds(path)
        assert start == 0.0
        assert end == pytest.approx(1.0)

    def test_stereo_is_downmixed(self, tmp_path):
        sr = 44100
        n = sr * 4
        t = np.arange(n) / sr
        left = np.zeros(n)
        right = np.zeros(n)
        mask = (t >= 1.0) & (t < 3.0)
        tone = np.sin(2 * np.pi * 330 * t)
        left[mask] = 0.5 * tone[mask]
        right[mask] = 0.5 * tone[mask]
        path = _write_wav(tmp_path / "stereo.wav", np.stack([left, right], axis=1), sr)

        start, end = detect_song_bounds(path)
        assert start == pytest.approx(1.0, abs=0.15)
        assert end == pytest.approx(3.0, abs=0.15)
