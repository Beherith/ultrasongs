from __future__ import annotations

import pytest

from ultrasongs.domain.ultrastar import (
    LineBreak,
    NoteType,
    UltrastarParseError,
    UltrastarSong,
    beat_to_ms,
    beats_to_ms,
    ms_to_beats,
    parse_ultrastar_text,
    reconstruct_lyrics,
    write_ultrastar_text,
)


def test_reconstruct_lyrics_uses_explicit_word_boundaries() -> None:
    song = parse_ultrastar_text(
        """#TITLE:Example
#BPM:120
#GAP:0
: 0 1 60 Broth
: 1 1 60 ers
: 2 1 60  of
: 3 1 60  the
: 4 1 60  mine
- 6
: 7 1 60 Sing
: 8 1 60  now
E
"""
    )

    assert reconstruct_lyrics(song) == "Brothers of the mine\nSing now"


def test_reconstruct_lyrics_falls_back_to_reviewable_tokens() -> None:
    song = parse_ultrastar_text(
        """#BPM:120
: 0 1 60 Broth
: 1 1 60 ers
: 2 1 60 of
E
"""
    )

    assert reconstruct_lyrics(song) == "Broth ers of"

SONG_TEXT = """#TITLE:Tést Song
#ARTIST:An Artist
#MP3:song.mp3
#VIDEO:song.mp4
#BPM:121,5
#GAP:32646
#CREATOR:Test Suite

: 0 2 17 Broth
* 2 2 29 ers
F 4 1 19  of
- 6 8
: 8 2 20 the
E
"""


def test_parser_preserves_metadata_events_and_lyric_spacing() -> None:
    song = parse_ultrastar_text(SONG_TEXT, strict=True)

    assert song.metadata.title == "Tést Song"
    assert song.metadata.bpm == 121.5
    assert song.metadata.gap_ms == 32646
    assert song.metadata.extras == {"CREATOR": "Test Suite"}
    assert [note.note_type for note in song.notes] == [
        NoteType.NORMAL,
        NoteType.GOLDEN,
        NoteType.FREESTYLE,
        NoteType.NORMAL,
    ]
    assert song.notes[2].lyric == " of"
    assert song.line_breaks == (LineBreak(6, 8),)
    assert tuple(song.verses()) == (song.notes[:3], song.notes[3:])


def test_writer_round_trip_is_semantically_lossless() -> None:
    song = parse_ultrastar_text(SONG_TEXT, strict=True)
    serialized = write_ultrastar_text(song)

    assert serialized.endswith("E\n")
    assert parse_ultrastar_text(serialized, strict=True) == song


def test_permissive_and_strict_parsing() -> None:
    content = "#BPM:120\nnot an event\n: 0 2 1 valid\nE\n"
    assert len(parse_ultrastar_text(content).notes) == 1
    with pytest.raises(UltrastarParseError, match="line 2"):
        parse_ultrastar_text(content, strict=True)


def test_invalid_bpm_has_domain_error() -> None:
    with pytest.raises(UltrastarParseError, match="BPM"):
        parse_ultrastar_text("#BPM:nope\nE\n")
    with pytest.raises(UltrastarParseError, match="greater than zero"):
        parse_ultrastar_text("#BPM:0\nE\n")


def test_time_conversions_use_ultrastar_sixteenth_note_units() -> None:
    assert beats_to_ms(4, 120) == pytest.approx(500)
    assert beat_to_ms(4, 120, 1000) == pytest.approx(1500)
    assert ms_to_beats(1500, 120, 1000) == 4
    assert ms_to_beats(1562.5, 120, 1000) == 5
    assert ms_to_beats(1562.5, 120, 1000, round_result=False) == pytest.approx(4.5)


def test_song_property_returns_only_notes() -> None:
    song = parse_ultrastar_text(": 0 2 1 one\n- 3\n: 4 2 2 two\nE\n")
    assert isinstance(song, UltrastarSong)
    assert [note.lyric for note in song.notes] == ["one", "two"]
