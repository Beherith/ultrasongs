"""Tests for Ultrastar format parser and builder."""

import pytest
from cli.pipeline_types import UltrastarMeta, UltrastarNote
from cli.ultrastar import build_ultrastar_txt, extract_lyrics_from_ultrastar, ms_to_beats, parse_ultrastar_txt


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

    def test_decimal_comma_gap(self):
        content = """#TITLE:Test
#ARTIST:Artist
#MP3:test.mp3
#BPM:340
#GAP:5073,53

: 0 4 60 hi
E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert meta.gap == 5073

    def test_line_break_with_extra_numbers(self):
        content = """#TITLE:Test
#ARTIST:Artist
#MP3:test.mp3
#BPM:120
#GAP:0

: 0 4 60 hi
- 105 608
: 15 4 62  bye
E
"""
        meta, notes = parse_ultrastar_txt(content)
        assert len(notes) == 3
        assert notes[1].note_type == "-"
        assert notes[1].start_beat == 105

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


class TestExtractLyricsFromUltrastar:
    def test_basic(self):
        content = """#TITLE:Test Song
#ARTIST:Test Artist
#MP3:test.mp3
#BPM:120.00
#GAP:500

: 0 4 60 hello
: 5 4 62  world
- 10
: 15 4 63 how
: 20 4 64  are
: 25 4 65  you
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello world\nhow are you\n"

    def test_syllable_reassembly(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 2 60 Broth
: 2 2 61 ers
: 4 2 62  of
: 6 2 63  the
E
"""
        assert extract_lyrics_from_ultrastar(content) == "Brothers of the\n"

    def test_unvoiced_notes_skipped(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 4 60 hello
: 5 4 61 ~
: 10 4 62  world
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello world\n"

    def test_empty_syllable_continuation(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 4 60 hello
: 5 4 61
: 10 4 62  world
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello world\n"

    def test_high_notes_included(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 4 60 hello
* 5 4 62  world
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello world\n"

    def test_trailing_unvoiced_at_line_end(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 4 60 stone
: 5 4 61 ~
- 10
: 15 4 62 home
E
"""
        assert extract_lyrics_from_ultrastar(content) == "stone\nhome\n"

    def test_consecutive_line_breaks_collapse(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

: 0 4 60 hello
- 10
- 12
: 15 4 62  world
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello\nworld\n"

    def test_trailing_space_ends_word(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0
-1360
: 1362 1 64 I
: 1364 2 64 ma
: 1366 2 61 gi
: 1368 4 66 na
: 1372 3 61 tion
-1375
: 1376 1 66 Life 
: 1378 1 66 is 
: 1380 2 66 your 
: 1382 1 64 cre
: 1384 4 68 a
: 1388 4 66 tion
-1392
: 1392 1 64 Come 
: 1394 2 64 on 
: 1396 2 61 Bar
: 1398 2 59 bie
E
"""
        assert extract_lyrics_from_ultrastar(content) == "Imagination\nLife is your creation\nCome on Barbie\n"

    def test_spaceless_line_break(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0
: 0 4 60 hello
-10
: 15 4 62  world
E
"""
        assert extract_lyrics_from_ultrastar(content) == "hello\nworld\n"

    def test_no_notes(self):
        content = """#TITLE:Test
#ARTIST:A
#MP3:t.mp3
#BPM:120
#GAP:0

E
"""
        assert extract_lyrics_from_ultrastar(content) == ""


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

    def test_preserves_leading_word_space_and_empty_continuation(self):
        meta = UltrastarMeta(title="Test", artist="A", mp3="a.mp3", bpm=120, gap=0)
        original = [
            UltrastarNote(":", 0, 2, 60, "hello"),
            UltrastarNote(":", 2, 2, 62, " world"),
            UltrastarNote(":", 4, 2, 64, ""),
        ]

        _, parsed = parse_ultrastar_txt(build_ultrastar_txt(original, meta))

        assert [note.syllable for note in parsed] == ["hello", " world", ""]
