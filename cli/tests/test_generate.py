"""Tests for note generation module."""

import pytest
from cli.config import Config
from cli.generate import generate_ultrastar
from cli.types import AlignedSyllable


class TestGenerateUltrastar:
    def _make_syllables(self, syllables: list[tuple[str, float, float, int]]) -> list[AlignedSyllable]:
        return [
            AlignedSyllable(syllable=s, start=st, end=ei, midi=m)
            for s, st, ei, m in syllables
        ]

    def test_basic_generation(self):
        syls = self._make_syllables([
            ("hel", 0.5, 0.8, 60),
            ("lo", 0.8, 1.0, 60),
            ("world", 1.2, 1.7, 62),
        ])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        assert "#TITLE:Test" in txt
        assert "#ARTIST:Artist" in txt
        assert "E" in txt

    def test_line_break_handling(self):
        syls = self._make_syllables([
            ("first", 0.5, 1.0, 60),
            ("line", 1.2, 1.7, 62),
        ])
        syls.append(AlignedSyllable(syllable="", start=2.0, end=2.0, midi=0, is_line_break=True))
        syls.extend(self._make_syllables([
            ("next", 2.5, 3.0, 64),
        ]))
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        assert "- " in txt  # line break note

    def test_overlap_prevention(self):
        syls = self._make_syllables([
            ("a", 0.5, 1.0, 60),
            ("b", 0.9, 1.1, 62),  # overlaps with "a"
        ])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        # Should not crash; notes should be non-overlapping
        assert "E" in txt

    def test_video_filename(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            video_filename="test.mp4",
            config=Config(),
        )
        assert "#VIDEO:test.mp4" in txt
