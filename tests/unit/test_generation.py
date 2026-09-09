from __future__ import annotations

from ultrasongs.domain.alignment import AlignedSyllable
from ultrasongs.domain.ultrastar import (
    LineBreak,
    generate_song_from_alignment,
)


def test_generates_gap_and_non_overlapping_notes() -> None:
    result = generate_song_from_alignment(
        [
            AlignedSyllable("Hel", 1.0, 1.5, 60),
            AlignedSyllable("lo", 1.4, 1.8, 62),
        ],
        title="Hello",
        artist="Singer",
        mp3_filename="hello.mp3",
        bpm=120,
    )

    first, second = result.song.notes
    assert result.gap_ms == 500
    assert first.start_beat == 4
    assert second.start_beat >= first.end_beat + 1


def test_line_break_caps_trailing_note() -> None:
    result = generate_song_from_alignment(
        [
            AlignedSyllable("long", 1.0, 5.0, 60),
            AlignedSyllable("", 2.0, 2.0, 0, is_line_break=True),
            AlignedSyllable("next", 2.0, 2.5, 62),
        ],
        title="Song",
        artist="Singer",
        mp3_filename="song.mp3",
        bpm=120,
        gap_ms=0,
    )

    assert result.song.notes[0].duration_beats == 6
    assert isinstance(result.song.events[1], LineBreak)
    assert result.song.notes[1].start_beat > result.song.events[1].start_beat


def test_accepts_legacy_camel_case_line_break() -> None:
    result = generate_song_from_alignment(
        [
            {"syllable": "one", "start": 1.0, "end": 1.2, "midi": 60},
            {
                "syllable": "",
                "start": 2.0,
                "end": 2.0,
                "midi": 0,
                "isLineBreak": True,
            },
        ],
        title="Song",
        artist="Singer",
        mp3_filename="song.mp3",
        bpm=120,
    )

    assert len(result.song.line_breaks) == 1
