from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ultrasongs.config import FfmpegSettings
from ultrasongs.processing.media import (
    MediaProcessingError,
    MediaService,
    is_supported_media,
    is_video_media,
)


def settings() -> FfmpegSettings:
    return FfmpegSettings(executable="ffmpeg-test", ffprobe_executable="ffprobe-test")


def test_media_extension_checks() -> None:
    assert is_supported_media("song.MP3")
    assert is_supported_media("clip.mkv")
    assert not is_supported_media("song.txt")
    assert is_video_media("clip.MOV")
    assert not is_video_media("song.flac")


def test_normalize_audio_builds_configured_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture")
    destination = tmp_path / "nested" / "output.mp3"
    captured: list[str] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured.extend(command)
        destination.write_bytes(b"mp3")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert MediaService(settings()).normalize_audio(source, destination) == destination.resolve()
    assert captured[0] == "ffmpeg-test"
    assert "-vn" in captured
    assert captured[-1] == str(destination.resolve())


def test_probe_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fixture")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "123.45\n", ""),
    )

    assert MediaService(settings()).probe_duration(source) == pytest.approx(123.45)


def test_subprocess_failure_is_descriptive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "bad input"),
    )

    with pytest.raises(MediaProcessingError, match="bad input"):
        MediaService(settings()).normalize_audio(source, tmp_path / "output.mp3")
