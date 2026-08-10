"""Tests for Ultrastar format parser and builder."""

import pytest
from cli.pipeline_types import UltrastarMeta, UltrastarNote
from cli.ultrastar import build_ultrastar_txt, ms_to_beats, parse_ultrastar_txt


class TestMsToBeats:
    def test_basic(self):
        # 1000ms at 120 BPM, gap=0 -> 1 beat = 125ms, so 1000ms = 8 beats
        result = ms_to_beats(1000, 120.0, 0)
        assert result == 8

    def test_with_gap(self):
        result = ms_to_beats(1500, 120.0, 500)
        assert result == 8

    def test_before_gap(self):
        result = ms_to_beats(200, 120.0, 500)
        assert result < 0


class TestBuildUltrastarTxt:
    def test_minimal(self):
        meta = UltrastarMeta(title="Test", artist="Artist", mp3="test.mp3", bpm=120.0, gap=500)
        notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=4, pitch=60, syllable="hello"),
        ]
        txt = build_ultrastar_txt(notes, meta)
        assert "#TITLE:Test" in txt
        assert "#ARTIST:Artist" in txt
        assert "#MP3:test.mp3" in txt
        assert "#BPM:120.00" in txt
        assert "#GAP:500" in txt
        assert ": 0 4 60 hello" in txt
        assert txt.strip().endswith("E")

    def test_with_video(self):
        meta = UltrastarMeta(title="Test", artist="Artist", mp3="test.mp3", bpm=120.0, gap=500, video="test.mp4")
        notes = []
        txt = build_ultrastar_txt(notes, meta)
        assert "#VIDEO:test.mp4" in txt

    def test_line_break(self):
        meta = UltrastarMeta(title="Test", artist="Artist", mp3="test.mp3", bpm=120.0, gap=500)
        notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=4, pitch=60, syllable="hello"),
            UltrastarNote(note_type="-", start_beat=10, duration=0, pitch=0, syllable=""),
        ]
        txt = build_ultrastar_txt(notes, meta)
        assert "- 10" in txt


class TestParseUltrastarTxt:
    def test_basic(self):
        content = """#TITLE:Test Song
#ARTIST:Test Artist
#MP3:test.mp3
#BPM:120.00
#GAP:500

: 0 4 60 hello
: 5 4 62 world
E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert meta.title == "Test Song"
        assert meta.artist == "Test Artist"
        assert meta.mp3 == "test.mp3"
        assert meta.bpm == 120.0
        assert meta.gap == 500
        assert len(notes) == 2
        assert notes[0].syllable == "hello"
        assert notes[1].syllable == "world"

    def test_with_video(self):
        content = """#TITLE:Test
#ARTIST:Artist
#MP3:test.mp3
#VIDEO:test.mp4
#BPM:120.00
#GAP:500

: 0 4 60 hi
E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert meta.video == "test.mp4"

    def test_line_break(self):
        content = """#TITLE:Test
#ARTIST:Artist
#MP3:test.mp3
#BPM:120.00
#GAP:500

: 0 4 60 hi
- 10
: 15 4 62 bye
E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert len(notes) == 3
        assert notes[1].note_type == "-"
        assert notes[1].start_beat == 10

    def test_comma_bpm(self):
        content = """#TITLE:Test
#ARTIST:Artist
#MP3:test.mp3
#BPM:120,50
#GAP:500

E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert meta.bpm == 120.5


class TestRoundTrip:
    def test_build_then_parse(self):
        original_meta = UltrastarMeta(title="Round", artist="Trip", mp3="r.mp3", bpm=115.5, gap=300)
        original_notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=8, pitch=64, syllable="hello"),
            UltrastarNote(note_type="-", start_beat=20, duration=0, pitch=0, syllable=""),
            UltrastarNote(note_type=":", start_beat=25, duration=4, pitch=67, syllable="world"),
        ]
        txt = build_ultrastar_txt(original_notes, original_meta)
        parsed_meta, parsed_notes = parse_ultrastar_txt(txt)

        assert parsed_meta.title == original_meta.title
        assert parsed_meta.artist == original_meta.artist
        assert parsed_meta.mp3 == original_meta.mp3
        assert parsed_meta.bpm == original_meta.bpm
        assert parsed_meta.gap == original_meta.gap
        assert len(parsed_notes) == len(original_notes)

        for i, (orig, parsed) in enumerate(zip(original_notes, parsed_notes)):
            assert parsed.note_type == orig.note_type
            assert parsed.start_beat == orig.start_beat
            assert parsed.duration == orig.duration
            assert parsed.pitch == orig.pitch
            assert parsed.syllable == orig.syllable
