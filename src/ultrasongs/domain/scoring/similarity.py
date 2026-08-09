"""Similarity scoring for canonical UltraStar songs."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ultrasongs.domain.ultrastar import (
    UltrastarNote,
    UltrastarSong,
    beat_to_ms,
    beats_to_ms,
    parse_ultrastar_file,
)


@dataclass(frozen=True, slots=True)
class MatchedNotes:
    reference: UltrastarNote
    candidate: UltrastarNote
    reference_start_ms: float
    candidate_start_ms: float
    reference_duration_ms: float
    candidate_duration_ms: float


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Machine-readable output from comparing two songs.

    Error fields are ``None`` when no notes match. This deliberately avoids
    presenting an empty comparison as a perfect zero-error score.
    """

    reference_notes: int
    candidate_notes: int
    matched_notes: int
    matched_ratio: float
    reference_coverage: float
    candidate_coverage: float
    timing_rmse_ms: float | None
    duration_rmse_ms: float | None
    pitch_distance_semitones: float | None
    timing_median_error_ms: float | None
    timing_max_error_ms: float | None
    duration_median_error_ms: float | None
    duration_max_error_ms: float | None
    pitch_median_distance_semitones: float | None
    pitch_max_distance_semitones: float | None

    @property
    def has_matches(self) -> bool:
        return self.matched_notes > 0

    def to_dict(self) -> dict[str, int | float | bool | None]:
        data = asdict(self)
        data["has_matches"] = self.has_matches
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def compare_songs(reference: UltrastarSong, candidate: UltrastarSong) -> SimilarityResult:
    pairs = align_notes(reference, candidate)
    total_reference = len(reference.notes)
    total_candidate = len(candidate.notes)
    matched = len(pairs)
    denominator = min(total_reference, total_candidate)

    if not pairs:
        return SimilarityResult(
            reference_notes=total_reference,
            candidate_notes=total_candidate,
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
        )

    timing_errors = [abs(pair.reference_start_ms - pair.candidate_start_ms) for pair in pairs]
    duration_errors = [
        abs(pair.reference_duration_ms - pair.candidate_duration_ms) for pair in pairs
    ]
    pitch_errors = [
        octave_corrected_distance(pair.reference.pitch, pair.candidate.pitch) for pair in pairs
    ]
    return SimilarityResult(
        reference_notes=total_reference,
        candidate_notes=total_candidate,
        matched_notes=matched,
        matched_ratio=matched / denominator if denominator else 0.0,
        reference_coverage=matched / total_reference if total_reference else 0.0,
        candidate_coverage=matched / total_candidate if total_candidate else 0.0,
        timing_rmse_ms=_rmse(timing_errors),
        duration_rmse_ms=_rmse(duration_errors),
        pitch_distance_semitones=sum(pitch_errors) / matched,
        timing_median_error_ms=_legacy_median(timing_errors),
        timing_max_error_ms=max(timing_errors),
        duration_median_error_ms=_legacy_median(duration_errors),
        duration_max_error_ms=max(duration_errors),
        pitch_median_distance_semitones=_legacy_median(pitch_errors),
        pitch_max_distance_semitones=max(pitch_errors),
    )


def compare_song_files(reference_path: str | Path, candidate_path: str | Path) -> SimilarityResult:
    return compare_songs(parse_ultrastar_file(reference_path), parse_ultrastar_file(candidate_path))


def align_notes(reference: UltrastarSong, candidate: UltrastarSong) -> tuple[MatchedNotes, ...]:
    """Match lyric-identical notes using the legacy nearest-time strategy."""

    available = set(range(len(candidate.notes)))
    result: list[MatchedNotes] = []
    reference_meta = reference.metadata
    candidate_meta = candidate.metadata

    for reference_note in reference.notes:
        reference_start = beat_to_ms(
            reference_note.start_beat, reference_meta.bpm, reference_meta.gap_ms
        )
        matching = [
            index
            for index in available
            if candidate.notes[index].lyric.strip() == reference_note.lyric.strip()
        ]
        if not matching:
            continue
        best_index = min(
            matching,
            key=lambda index: abs(
                beat_to_ms(
                    candidate.notes[index].start_beat,
                    candidate_meta.bpm,
                    candidate_meta.gap_ms,
                )
                - reference_start
            ),
        )
        available.remove(best_index)
        candidate_note = candidate.notes[best_index]
        result.append(
            MatchedNotes(
                reference=reference_note,
                candidate=candidate_note,
                reference_start_ms=reference_start,
                candidate_start_ms=beat_to_ms(
                    candidate_note.start_beat, candidate_meta.bpm, candidate_meta.gap_ms
                ),
                reference_duration_ms=beats_to_ms(
                    reference_note.duration_beats, reference_meta.bpm
                ),
                candidate_duration_ms=beats_to_ms(
                    candidate_note.duration_beats, candidate_meta.bpm
                ),
            )
        )
    return tuple(result)


def octave_corrected_distance(pitch_a: int, pitch_b: int) -> int:
    """Return chromatic pitch-class distance, ignoring octave displacement."""

    difference = abs((pitch_a % 12) - (pitch_b % 12))
    return min(difference, 12 - difference)


def _rmse(errors: Sequence[float | int]) -> float:
    return math.sqrt(sum(float(error) ** 2 for error in errors) / len(errors))


def _legacy_median(values: Sequence[float | int]) -> float:
    """Retain the original scorer's upper-middle definition for even samples."""

    ordered = sorted(float(value) for value in values)
    return ordered[len(ordered) // 2]
