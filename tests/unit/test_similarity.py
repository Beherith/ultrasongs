from __future__ import annotations

import json

import pytest

from ultrasongs.domain.scoring import compare_song_files, compare_songs, octave_corrected_distance
from ultrasongs.domain.ultrastar import parse_ultrastar_text


def song(body: str, *, bpm: float = 120, gap: float = 0):
    return parse_ultrastar_text(f"#BPM:{bpm}\n#GAP:{gap}\n{body}\nE\n", strict=True)


def test_identical_songs_score_zero_with_full_coverage() -> None:
    reference = song(": 0 2 17 one\n: 4 2 19 two")
    result = compare_songs(reference, reference)

    assert result.has_matches
    assert result.matched_notes == 2
    assert result.matched_ratio == 1
    assert result.reference_coverage == 1
    assert result.candidate_coverage == 1
    assert result.timing_rmse_ms == 0
    assert result.duration_rmse_ms == 0
    assert result.pitch_distance_semitones == 0


def test_global_gap_offset_is_measured_in_milliseconds() -> None:
    reference = song(": 0 2 17 one\n: 4 2 19 two")
    candidate = song(": 0 2 17 one\n: 4 2 19 two", gap=250)

    result = compare_songs(reference, candidate)
    assert result.timing_rmse_ms == pytest.approx(250)
    assert result.timing_median_error_ms == pytest.approx(250)
    assert result.timing_max_error_ms == pytest.approx(250)


def test_duration_and_octave_corrected_pitch_metrics() -> None:
    reference = song(": 0 2 17 one\n: 4 2 19 two")
    candidate = song(": 0 3 29 one\n: 4 4 21 two")

    result = compare_songs(reference, candidate)
    assert result.duration_rmse_ms == pytest.approx((125**2 + 250**2) ** 0.5 / 2**0.5)
    assert result.pitch_distance_semitones == pytest.approx(1)
    assert octave_corrected_distance(17, 29) == 0
    assert octave_corrected_distance(19, 21) == 2


def test_repeated_lyrics_are_matched_once_to_nearest_time() -> None:
    reference = song(": 0 1 1 la\n: 100 1 2 la\n: 200 1 3 missing")
    candidate = song(": 98 1 2 la\n: 2 1 1 la")

    result = compare_songs(reference, candidate)
    assert result.matched_notes == 2
    assert result.matched_ratio == 1
    assert result.reference_coverage == pytest.approx(2 / 3)
    assert result.candidate_coverage == 1
    assert result.timing_rmse_ms == pytest.approx(250)


def test_zero_match_does_not_look_like_a_perfect_score() -> None:
    result = compare_songs(song(": 0 2 17 one"), song(": 0 2 17 other"))

    assert not result.has_matches
    assert result.matched_ratio == 0
    assert result.timing_rmse_ms is None
    assert result.duration_rmse_ms is None
    assert result.pitch_distance_semitones is None
    assert json.loads(result.to_json())["has_matches"] is False


def test_compare_song_files_uses_canonical_parser(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    content = "#BPM:120\n#GAP:100\n: 40 2 17 hello\nE\n"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    result = compare_song_files(first, second)
    assert result.matched_notes == 1
    assert result.timing_rmse_ms == 0
