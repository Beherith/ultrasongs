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

    def test_creator_tag_stamped(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.creator is not None
        assert meta.creator.startswith("https://github.com/Beherith/ultrasongs  - ")
        header = txt.split("\n\n")[0].split("\n")
        assert header.index("#CREATOR:" + meta.creator) == header.index("#ARTIST:Artist") + 1

    def test_start_end_tags(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            start_sec=1.5,
            end_ms=678000,
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.start == 1.5
        assert meta.end_ms == 678000

    def test_no_start_end_tags(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.start is None
        assert meta.end_ms is None

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

    def test_glissando_continuation_uses_tilde(self):
        syls = self._make_syllables([
            ("ba", 0.5, 0.8, 60),
            ("", 0.8, 1.0, 62),
            ("lo", 1.2, 1.7, 64),
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
        assert singing[0].syllable == "ba"
        assert singing[0].pitch == 60
        assert singing[1].syllable == "~"
        assert singing[1].pitch == 62
        assert singing[2].syllable == "lo"

    def test_first_beat_ms_used_as_gap(self):
        syls = self._make_syllables([("hi", 1.5, 2.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            first_beat_ms=735.0,
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.gap == 735

    def test_gap_falls_back_to_first_note_without_first_beat(self):
        syls = self._make_syllables([("hi", 1.5, 2.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.gap == 1000

    def test_medley_tags_from_lyrics(self):
        lyrics = "\n".join([
            "verse line one",
            "verse line two",
            "chorus line a",
            "chorus line b",
            "verse line three",
            "verse line four",
            "chorus line a",
            "chorus line b",
        ])
        syls = self._make_syllables([
            (line.split()[0], index * 1.0, index * 1.0 + 0.5, 60)
            for index, line in enumerate(lyrics.split("\n"))
        ])
        for index in range(7):
            syls.insert((index + 1) * 2 - 1, AlignedSyllable(syllable="", start=(index + 1) * 1.0, end=(index + 1) * 1.0, midi=0, is_line_break=True))
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            first_beat_ms=0.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            lyrics_text=lyrics,
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        # output_bpm = 240 (beat resolution multiplier 2), beat = 250 ms, gap = 0
        # chorus lines are 0-based lines 2-3 -> first note at beat 32, last ends at beat 56
        assert meta.medley_start_beat == 32
        assert meta.medley_end_beat == 56
        assert meta.preview_start == 8.0

    def test_no_medley_tags_without_lyrics(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.medley_start_beat is None
        assert meta.medley_end_beat is None
        assert meta.preview_start is None

    def test_no_medley_tags_when_lyrics_do_not_repeat(self):
        syls = self._make_syllables([
            ("one", 0.5, 1.0, 60),
            ("two", 1.2, 1.7, 62),
            ("three", 2.2, 2.7, 64),
            ("four", 3.2, 3.7, 65),
        ])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            first_beat_ms=0.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            lyrics_text="one\ntwo\nthree\nfour",
            config=Config(),
        )
        meta, _ = parse_ultrastar_txt(txt)
        assert meta.medley_start_beat is None
        assert meta.medley_end_beat is None
        assert meta.preview_start is None

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

    def test_cover_filename(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            cover_filename="test_cover.jpeg",
            config=Config(),
        )
        assert "#COVER:test_cover.jpeg" in txt

    def test_without_cover(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        assert "#COVER:" not in txt

    def test_stem_filenames(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            video_filename="test.mp4",
            vocals_filename="test_vocals.mp3",
            instrumental_filename="test_accompaniment.mp3",
            config=Config(),
        )
        assert "#VOCALS:test_vocals.mp3" in txt
        assert "#INSTRUMENTAL:test_accompaniment.mp3" in txt
        header = txt.split("\n\n")[0].split("\n")
        assert header.index("#VOCALS:test_vocals.mp3") < header.index("#INSTRUMENTAL:test_accompaniment.mp3")
        assert header.index("#INSTRUMENTAL:test_accompaniment.mp3") < header.index("#BPM:240.00")

    def test_without_stems(self):
        syls = self._make_syllables([("hi", 0.5, 1.0, 60)])
        txt = generate_ultrastar(
            aligned_syllables=syls,
            bpm=120.0,
            gap_ms=500,
            title="Test",
            artist="Artist",
            mp3_filename="test.mp3",
            config=Config(),
        )
        assert "#VOCALS:" not in txt
        assert "#INSTRUMENTAL:" not in txt
