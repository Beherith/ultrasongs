from __future__ import annotations

from pathlib import Path

from ultrasongs.config import TempoSettings
from ultrasongs.processing.tempo import TempoService


def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "song.mp3"
    path.write_bytes(b"fixture")
    return path


def test_detects_and_normalizes_tempo(tmp_path: Path) -> None:
    service = TempoService(
        TempoSettings(minimum_bpm=40, maximum_bpm=240),
        analyzer=lambda path: (30.0, (0.0, 2.0)),
    )

    result = service.detect(audio_file(tmp_path))

    assert result.bpm == 60
    assert result.beat_times == (0.0, 2.0)
    assert not result.used_fallback


def test_uses_central_fallback_on_failure(tmp_path: Path) -> None:
    def fail(_: Path) -> tuple[float, tuple[float, ...]]:
        raise RuntimeError("no beat")

    result = TempoService(TempoSettings(fallback_bpm=123), analyzer=fail).detect(
        audio_file(tmp_path)
    )

    assert result.bpm == 123
    assert result.used_fallback
    assert "no beat" in (result.warning or "")
