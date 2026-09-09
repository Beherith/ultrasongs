"""Tests for ffmpeg_extract helpers."""

from pathlib import Path

from cli.ffmpeg_extract import (
    _SUPPORTED_EXTS,
    _VIDEO_EXTS,
    is_video_path,
)


class TestIsVideoPath:
    def test_video_extensions(self):
        for ext in sorted(_VIDEO_EXTS):
            assert is_video_path(Path(f"song{ext}"))

    def test_video_extension_case_insensitive(self):
        assert is_video_path(Path("Song.MP4"))

    def test_audio_extensions(self):
        assert not is_video_path(Path("song.mp3"))
        assert not is_video_path(Path("song.wav"))
        assert not is_video_path(Path("song.flac"))

    def test_no_extension(self):
        assert not is_video_path(Path("song"))

    def test_unsupported(self):
        assert not is_video_path(Path("song.bin"))


class TestSupportedExtensions:
    def test_video_subset(self):
        assert _VIDEO_EXTS <= _SUPPORTED_EXTS
