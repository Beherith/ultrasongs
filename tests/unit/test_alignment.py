from __future__ import annotations

import pytest

from ultrasongs.domain.alignment import (
    PitchFrame,
    WordTimestamp,
    align_lyrics,
    align_lyrics_with_debug,
    midi_for_range,
    normalize_character,
    phonetic_score,
    smith_waterman,
    split_word,
)


def word(text: str, start: float, end: float, midi: int = 60) -> WordTimestamp:
    return WordTimestamp(word=text, start=start, end=end, midi=midi)


def test_normalization_phonetic_groups_and_smith_waterman() -> None:
    assert normalize_character("É") == "e"
    assert phonetic_score("a", "a") == 1.0
    assert phonetic_score("a", "e") == pytest.approx(0.5)
    assert phonetic_score("x", "q") == -0.3

    result = smith_waterman(list("cafe"), list("cafe"))
    assert result.max_score == 16.0
    assert (result.max_i, result.max_j) == (4, 4)
    assert [step.matrix for step in result.backtrack] == ["M", "M", "M", "M"]


def test_accented_lyrics_align_to_unaccented_transcription() -> None:
    result = align_lyrics_with_debug(
        "Café világ",
        [word("cafe", 1.0, 1.5, 64), word("vilag", 1.6, 2.2, 67)],
        "hu",
    )

    assert result.debug.summary.aligned_words == 2
    assert [item.word for item in result.debug.words] == ["Café", "világ"]
    assert result.debug.words[0].start == 1.0
    assert result.debug.words[1].end == 2.2


def test_repeated_lyrics_map_to_distinct_transcription_regions() -> None:
    result = align_lyrics_with_debug(
        "go now go",
        [word("go", 0.0, 0.3), word("now", 0.4, 0.8), word("go", 0.9, 1.2)],
        "en",
    )

    assert result.debug.summary.aligned_words == 3
    assert result.debug.words[0].transcription_word_indices == (0,)
    assert result.debug.words[2].transcription_word_indices == (2,)
    assert [item.start for item in result.debug.words] == [0.0, 0.4, 0.9]


def test_partial_match_interpolates_before_between_and_after() -> None:
    result = align_lyrics_with_debug(
        "intro hello missing world outro",
        [word("hello", 2.0, 2.5, 60), word("world", 4.0, 4.5, 64)],
        "zz",
    )
    sources = [item.source for item in result.debug.words]

    assert sources == [
        "interpolated_before",
        "sw_aligned",
        "interpolated_between",
        "sw_aligned",
        "interpolated_after",
    ]
    assert result.debug.words[0].end == 2.0
    assert result.debug.words[2].start == pytest.approx(3.25)
    assert result.debug.words[2].midi == 62
    assert result.debug.words[4].start == 5.0


def test_pitch_frame_median_uses_strict_confidence_fallbacks() -> None:
    timestamp = WordTimestamp(
        word="sing",
        start=0.0,
        end=1.0,
        midi=55,
        pitch_frames=(
            PitchFrame(time=0.1, midi=60, confidence=0.5),
            PitchFrame(time=0.2, midi=61, confidence=0.51),
            PitchFrame(time=0.3, midi=64, confidence=0.9),
            PitchFrame(time=1.2, midi=90, confidence=1.0),
        ),
    )

    assert midi_for_range(timestamp, 0.0, 1.0) == (63, 2)
    assert midi_for_range(timestamp, 1.1, 1.3) == (90, 1)
    assert midi_for_range(timestamp, 2.0, 3.0) == (55, 0)


def test_mapping_inputs_pitch_syllables_and_line_breaks() -> None:
    result = align_lyrics(
        "singing\nagain",
        [
            {
                "word": "singing",
                "start": 0,
                "end": 1,
                "midi": 50,
                "pitchFrames": [
                    {"time": 0.1, "midi": 60, "confidence": 0.9},
                    {"time": 0.8, "midi": 65, "confidence": 0.9},
                ],
            },
            {"word": "again", "start": 1.2, "end": 2.0, "midi": 67},
        ],
        "en",
    )

    line_breaks = [item for item in result if item.is_line_break]
    sung = [item for item in result if not item.is_line_break]
    assert len(line_breaks) == 1
    assert line_breaks[0].midi == 0
    assert line_breaks[0].start == next(
        item.start for item in sung if item.syllable in {"again", "a"}
    )
    assert {item.midi for item in sung[:2]} <= {50, 60, 65}


def test_no_match_and_empty_input_are_deterministic() -> None:
    unmatched = align_lyrics_with_debug("xyz", [word("aaa", 3.0, 4.0)], "zz")
    assert unmatched.debug.summary.interpolated_words == 1
    assert unmatched.syllables[0].start == 0.0
    assert unmatched.syllables[0].end == 0.01

    empty = align_lyrics_with_debug("\n", [], "en")
    assert empty.syllables == ()
    assert empty.debug.summary.total_lyric_words == 0


def test_syllable_fallback_preserves_word_and_short_words() -> None:
    assert split_word("I", "zz") == ["I"]
    parts = split_word("banana", "zz")
    assert "".join(parts) == "banana"
    assert parts == split_word("banana", "zz")
