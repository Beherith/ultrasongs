from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultrasongs.config import AppSettings
from ultrasongs.domain.alignment import align_lyrics_with_debug
from ultrasongs.domain.reporting import write_pipeline_report
from ultrasongs.domain.scoring import compare_songs
from ultrasongs.domain.ultrastar import (
    generate_song_from_alignment,
    parse_ultrastar_file,
    parse_ultrastar_text,
    write_ultrastar_file,
)

pytestmark = pytest.mark.parity


def test_frozen_transcription_to_candidate_and_report(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[2]
    reference_path = workspace / "notes.txt"
    transcription_path = workspace / "Diggy Diggy Hole_transcribe.json"
    lyrics_path = workspace / "diggy_lyrics.txt"
    missing = [
        path.name
        for path in (reference_path, transcription_path, lyrics_path)
        if not path.is_file()
    ]
    if missing:
        pytest.skip(f"local migration fixtures unavailable: {', '.join(missing)}")

    transcription = json.loads(transcription_path.read_text(encoding="utf-8"))
    alignment = align_lyrics_with_debug(
        lyrics_path.read_text(encoding="utf-8"),
        transcription["words"],
        transcription.get("language", "en"),
        transcription.get("pauses", ()),
    )
    generation = generate_song_from_alignment(
        alignment.syllables,
        title="Diggy Diggy Hole",
        artist="Siouxsie and the Banshees12",
        mp3_filename="Diggy Diggy Hole.mp3",
        bpm=120,
    )
    candidate_path = tmp_path / "candidate.txt"
    write_ultrastar_file(generation.song, candidate_path)

    reference = parse_ultrastar_file(reference_path)
    candidate = parse_ultrastar_file(candidate_path)
    similarity = compare_songs(reference, candidate)
    report_path = write_pipeline_report(
        tmp_path / "validation-report.html",
        candidate=candidate,
        reference=reference,
        transcription=transcription,
        similarity=similarity,
        effective_config=AppSettings().model_dump(mode="json"),
    )

    assert generation.gap_ms == pytest.approx(32_120)
    assert alignment.debug.summary.aligned_words >= 300
    assert len(candidate.notes) >= 400
    # Current migration baseline against the known-good community chart.
    assert similarity.matched_notes >= 280
    assert similarity.pitch_distance_semitones is not None
    assert similarity.pitch_distance_semitones < 1.2
    assert parse_ultrastar_text(candidate_path.read_text(encoding="utf-8")).notes
    assert report_path.stat().st_size > 100_000
