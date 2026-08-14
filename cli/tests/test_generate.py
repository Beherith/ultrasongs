"""Tests for note generation module."""

import pytest
from cli.config import Config
from cli.generate import generate_ultrastar
from cli.pipeline_types import AlignedSyllable
from cli.ultrastar import parse_ultrastar_txt


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
        _, notes = parse_ultrastar_txt(txt)
        line_break_index = next(i for i, note in enumerate(notes) if note.note_type == "-")
        previous = notes[line_break_index - 1]
        line_break = notes[line_break_index]
        following = notes[line_break_index + 1]
        assert previous.start_beat + previous.duration <= line_break.start_beat
        assert line_break.start_beat <= following.start_beat

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
        _, notes = parse_ultrastar_txt(txt)
        singing = [note for note in notes if note.note_type != "-"]
        assert all(
            current.start_beat + current.duration <= following.start_beat
            for current, following in zip(singing, singing[1:])
        )

    def test_rounding_does_not_create_overlap(self):
        syls = self._make_syllables([
            ("a", 0.5, 0.8, 60),
            ("b", 0.8, 1.1, 62),
            ("c", 1.1, 1.4, 64),
        ])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=123.05,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )

        _, notes = parse_ultrastar_txt(txt)
        assert all(
            current.start_beat + current.duration <= following.start_beat
            for current, following in zip(notes, notes[1:])
            if current.note_type != "-" and following.note_type != "-"
        )

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
