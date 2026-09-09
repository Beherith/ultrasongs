"""Reference-song inspection and similarity acceptance rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultrasongs.config import ValidationSettings
from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.domain.ultrastar import (
    UltrastarSong,
    beat_to_ms,
    beats_to_ms,
    parse_ultrastar_text,
    reconstruct_lyrics,
)


@dataclass(frozen=True, slots=True)
class ReferenceSongInspection:
    song: UltrastarSong
    title: str
    artist: str
    bpm: float
    gap_ms: float
    note_count: int
    duration_ms: float
    reconstructed_lyrics: str


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    passed: bool
    failures: tuple[str, ...]
    similarity: SimilarityResult

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "similarity": self.similarity.to_dict(),
        }


def inspect_reference_bytes(data: bytes) -> ReferenceSongInspection:
    """Decode and inspect uploaded TXT bytes without mutating the source."""

    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Reference Ultrastar TXT must be UTF-8 encoded") from exc
    return inspect_reference_text(content)


def inspect_reference_file(path: str | Path) -> ReferenceSongInspection:
    return inspect_reference_bytes(Path(path).read_bytes())


def inspect_reference_text(content: str) -> ReferenceSongInspection:
    song = parse_ultrastar_text(content, strict=True)
    notes = song.notes
    duration_ms = max(
        (
            beat_to_ms(note.start_beat, song.metadata.bpm, song.metadata.gap_ms)
            + beats_to_ms(note.duration_beats, song.metadata.bpm)
            for note in notes
        ),
        default=0.0,
    )
    return ReferenceSongInspection(
        song=song,
        title=song.metadata.title,
        artist=song.metadata.artist,
        bpm=song.metadata.bpm,
        gap_ms=song.metadata.gap_ms,
        note_count=len(notes),
        duration_ms=duration_ms,
        reconstructed_lyrics=reconstruct_lyrics(song),
    )


def evaluate_similarity(
    similarity: SimilarityResult,
    settings: ValidationSettings,
) -> ValidationOutcome:
    """Apply centrally configured thresholds to a structured comparison."""

    failures: list[str] = []
    if not similarity.has_matches:
        failures.append("No notes matched between the reference and candidate")
    if similarity.matched_notes < settings.minimum_matched_notes:
        failures.append(
            f"Matched notes {similarity.matched_notes} is below "
            f"{settings.minimum_matched_notes}"
        )
    # The configured threshold is defined as coverage of the uploaded
    # reference, not coverage of the smaller of the two songs.
    if similarity.reference_coverage < settings.minimum_match_ratio:
        failures.append(
            f"Reference coverage {similarity.reference_coverage:.3f} is below "
            f"{settings.minimum_match_ratio:.3f}"
        )

    _maximum_metric(
        failures,
        "Timing RMSE",
        similarity.timing_rmse_ms,
        settings.maximum_timing_rmse_ms,
        "ms",
    )
    _maximum_metric(
        failures,
        "Duration RMSE",
        similarity.duration_rmse_ms,
        settings.maximum_duration_rmse_ms,
        "ms",
    )
    _maximum_metric(
        failures,
        "Pitch distance",
        similarity.pitch_distance_semitones,
        settings.maximum_pitch_distance_semitones,
        "semitones",
    )
    return ValidationOutcome(not failures, tuple(failures), similarity)


def _maximum_metric(
    failures: list[str],
    label: str,
    value: float | None,
    maximum: float,
    unit: str,
) -> None:
    if value is None:
        if not any(failure.startswith("No notes matched") for failure in failures):
            failures.append(f"{label} is unavailable")
    elif value > maximum:
        failures.append(f"{label} {value:.3f} {unit} exceeds {maximum:.3f} {unit}")
