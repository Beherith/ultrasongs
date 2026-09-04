import pytest

from cli.hybrid_transcribe import (
    align_lyrics_approximately,
    build_lyric_chunks,
    offset_aligned_words,
    slice_audio,
)
from cli.pipeline_types import Pause


def _words(items):
    return [
        {"word": word, "start": start, "end": end}
        for word, start, end in items
    ]


def test_approximate_alignment_uses_authoritative_lyrics():
    aligned = align_lyrics_approximately(
        "hello brave world",
        _words([("hello", 0.2, 0.6), ("world", 1.2, 1.7)]),
    )

    assert [word.word for word in aligned] == ["hello", "brave", "world"]
    assert aligned[0].start == pytest.approx(0.2)
    assert aligned[1].start == pytest.approx(0.6)
    assert aligned[1].end == pytest.approx(1.2)
    assert aligned[2].end == 1.7


def test_chunks_split_only_at_pauses_longer_than_one_second():
    chunks = build_lyric_chunks(
        "one two three four",
        _words([
            ("one", 0.2, 0.6),
            ("two", 0.7, 1.0),
            ("three", 2.4, 2.8),
            ("four", 3.0, 3.4),
        ]),
        [Pause(1.0, 2.0), Pause(1.1, 2.2)],
        audio_duration=4.0,
        min_pause_seconds=1.0,
    )

    # The 1.0-second pause is not long enough; the 1.1-second pause is.
    assert [chunk.text for chunk in chunks] == ["one two", "three four"]
    assert chunks[0].start == 0.0
    assert chunks[0].end == pytest.approx(1.65)
    assert chunks[1].start == pytest.approx(1.65)
    assert chunks[1].end == 4.0


def test_chunk_boundaries_use_pause_midpoint_without_dropping_audio():
    chunks = build_lyric_chunks(
        "before after",
        _words([("before", 0.0, 0.8), ("after", 2.2, 3.0)]),
        [Pause(0.9, 2.1)],
        audio_duration=3.0,
    )

    assert chunks[0].end == chunks[1].start == 1.5


def test_slice_audio_maps_global_seconds_to_samples():
    audio = list(range(40))
    assert slice_audio(audio, 1.0, 3.0, 4.0) == list(range(10, 30))


def test_offset_aligned_words_offsets_characters_without_mutating_input():
    words = [{
        "word": "hello",
        "start": 0.1,
        "end": 0.5,
        "characters": [
            {"char": "h", "start": 0.1, "end": 0.2, "score": 0.9},
            {"char": "?", "start": None, "end": None, "score": None},
        ],
    }]

    shifted = offset_aligned_words(words, 2.0)

    assert shifted[0]["start"] == 2.1
    assert shifted[0]["characters"][0]["end"] == 2.2
    assert shifted[0]["characters"][1]["start"] is None
    assert words[0]["start"] == 0.1
