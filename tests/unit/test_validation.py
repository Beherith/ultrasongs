from __future__ import annotations

from ultrasongs.config import ValidationSettings
from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.domain.validation import (
    evaluate_similarity,
    inspect_reference_bytes,
)


def similarity(**updates: object) -> SimilarityResult:
    values: dict[str, object] = {
        "reference_notes": 10,
        "candidate_notes": 10,
        "matched_notes": 10,
        "matched_ratio": 1.0,
        "reference_coverage": 1.0,
        "candidate_coverage": 1.0,
        "timing_rmse_ms": 10.0,
        "duration_rmse_ms": 10.0,
        "pitch_distance_semitones": 0.2,
        "timing_median_error_ms": 5.0,
        "timing_max_error_ms": 20.0,
        "duration_median_error_ms": 5.0,
        "duration_max_error_ms": 20.0,
        "pitch_median_distance_semitones": 0.0,
        "pitch_max_distance_semitones": 1.0,
    }
    values.update(updates)
    return SimilarityResult(**values)  # type: ignore[arg-type]


def test_inspects_uploaded_reference_bytes() -> None:
    inspection = inspect_reference_bytes(
        b"#TITLE:Example\r\n#ARTIST:Artist\r\n#BPM:120\r\n#GAP:1000\r\n"
        b": 0 4 60 Hel\r\n: 4 4 62  lo\r\nE\r\n"
    )

    assert inspection.title == "Example"
    assert inspection.note_count == 2
    assert inspection.duration_ms == 2000
    assert inspection.reconstructed_lyrics == "Hel lo"


def test_validation_passes_within_thresholds() -> None:
    outcome = evaluate_similarity(similarity(), ValidationSettings())

    assert outcome.passed
    assert outcome.failures == ()


def test_validation_reports_all_threshold_failures() -> None:
    settings = ValidationSettings(
        minimum_matched_notes=8,
        minimum_match_ratio=0.9,
        maximum_timing_rmse_ms=100,
        maximum_duration_rmse_ms=100,
        maximum_pitch_distance_semitones=1,
    )
    outcome = evaluate_similarity(
        similarity(
            matched_notes=5,
            matched_ratio=0.5,
            reference_coverage=0.5,
            timing_rmse_ms=200.0,
            duration_rmse_ms=150.0,
            pitch_distance_semitones=2.0,
        ),
        settings,
    )

    assert not outcome.passed
    assert len(outcome.failures) == 5


def test_zero_matches_never_looks_perfect() -> None:
    outcome = evaluate_similarity(
        similarity(
            matched_notes=0,
            matched_ratio=0.0,
            reference_coverage=0.0,
            candidate_coverage=0.0,
            timing_rmse_ms=None,
            duration_rmse_ms=None,
            pitch_distance_semitones=None,
            timing_median_error_ms=None,
            timing_max_error_ms=None,
            duration_median_error_ms=None,
            duration_max_error_ms=None,
            pitch_median_distance_semitones=None,
            pitch_max_distance_semitones=None,
        ),
        ValidationSettings(),
    )

    assert not outcome.passed
    assert outcome.failures[0].startswith("No notes matched")
