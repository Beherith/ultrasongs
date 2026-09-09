#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for UltraStar similarity scoring."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Permit running this compatibility script directly from a source checkout.
_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from ultrasongs.domain.scoring import (  # noqa: E402
    SimilarityResult,
    compare_song_files,
    octave_corrected_distance,
)
from ultrasongs.domain.ultrastar import (  # noqa: E402
    beat_to_ms,
    beats_to_ms,
    parse_ultrastar_file,
)


def parse_ultrastar(filepath: str) -> dict[str, Any]:
    """Return the dictionary shape exposed by the original utility script."""

    song = parse_ultrastar_file(filepath)
    metadata = song.metadata.as_headers()
    notes = [
        {
            "start": beat_to_ms(note.start_beat, song.metadata.bpm, song.metadata.gap_ms),
            "duration": beats_to_ms(note.duration_beats, song.metadata.bpm),
            "pitch": note.pitch,
            "lyric": note.lyric,
            "chorus": note.chorus,
        }
        for note in song.notes
    ]
    return {"metadata": metadata, "notes": notes, "times_are_ms": False}


def align_notes(notes_a: Sequence[dict[str, Any]], notes_b: Sequence[dict[str, Any]]):
    """Legacy dictionary-based matcher retained for importing scripts."""

    used_b: set[int] = set()
    pairs = []
    for note_a in notes_a:
        candidates = [
            index
            for index, note_b in enumerate(notes_b)
            if index not in used_b and note_b["lyric"].strip() == note_a["lyric"].strip()
        ]
        if candidates:
            best_index = min(
                candidates, key=lambda index: abs(notes_b[index]["start"] - note_a["start"])
            )
            used_b.add(best_index)
            pairs.append((note_a, notes_b[best_index]))
    return pairs


def score_timing(pairs) -> float | None:
    return _rmse_or_none([a["start"] - b["start"] for a, b in pairs])


def score_duration(pairs) -> float | None:
    return _rmse_or_none([a["duration"] - b["duration"] for a, b in pairs])


def score_pitch(pairs) -> float | None:
    if not pairs:
        return None
    return sum(octave_corrected_distance(a["pitch"], b["pitch"]) for a, b in pairs) / len(pairs)


def _rmse_or_none(errors: Sequence[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(error**2 for error in errors) / len(errors))


def _format_metric(value: float | None, unit: str) -> str:
    return "n/a (no matched notes)" if value is None else f"{value:>10.2f} {unit}"


def format_text_report(
    result: SimilarityResult,
    *,
    title_a: str,
    title_b: str,
    bpm_a: str,
    bpm_b: str,
) -> str:
    lines = [
        "=" * 52,
        "  UltraStar Song Similarity Score",
        "=" * 52,
        f"  File A: {title_a}  (BPM {bpm_a}, {result.reference_notes} notes, times in beats)",
        f"  File B: {title_b}  (BPM {bpm_b}, {result.candidate_notes} notes, times in beats)",
        "  Matched: "
        f"{result.matched_notes} / "
        f"{min(result.reference_notes, result.candidate_notes)} notes",
        "-" * 52,
        f"  Timing RMSE:        {_format_metric(result.timing_rmse_ms, 'ms')}",
        f"  Duration RMSE:      {_format_metric(result.duration_rmse_ms, 'ms')}",
        "  Pitch distance:     "
        + _format_metric(result.pitch_distance_semitones, "semitones (octave-corrected)"),
        "=" * 52,
    ]
    if result.has_matches:
        lines.extend(
            [
                "  Timing  - median: "
                f"{result.timing_median_error_ms:.1f} ms, "
                f"max: {result.timing_max_error_ms:.1f} ms",
                "  Duration- median: "
                f"{result.duration_median_error_ms:.1f} ms, "
                f"max: {result.duration_max_error_ms:.1f} ms",
                "  Pitch   - median: "
                f"{result.pitch_median_distance_semitones:.1f} st, "
                f"max: {result.pitch_max_distance_semitones:.1f} st",
                "=" * 52,
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_output = False
    if "--json" in arguments:
        arguments.remove("--json")
        json_output = True
    if len(arguments) < 2:
        print("Usage: python score_songs.py <song1.txt> <song2.txt> [--json]")
        return 1

    file_a, file_b = map(Path, arguments[:2])
    for path in (file_a, file_b):
        if not path.exists():
            print(f"Error: {path} not found.")
            return 1

    song_a = parse_ultrastar_file(file_a)
    song_b = parse_ultrastar_file(file_b)
    result = compare_song_files(file_a, file_b)
    if json_output:
        payload = {
            "reference": str(file_a),
            "candidate": str(file_b),
            "result": result.to_dict(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            format_text_report(
                result,
                title_a=song_a.metadata.title or file_a.stem,
                title_b=song_b.metadata.title or file_b.stem,
                bpm_a=song_a.metadata.get("BPM", "-"),
                bpm_b=song_b.metadata.get("BPM", "-"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
