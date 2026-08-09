from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tests.unit.test_pipeline import REFERENCE, make_runner
from ultrasongs.__main__ import main
from ultrasongs.cli import RepairWorkflowResult, run_repair_workflow
from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.domain.ultrastar import parse_ultrastar_file
from ultrasongs.domain.validation import ValidationOutcome


def test_repair_workflow_runs_pipeline_and_exports_review_bundle(tmp_path: Path) -> None:
    runner, projects, artifacts, _ = make_runner(tmp_path)
    audio = tmp_path / "existing.mp3"
    audio.write_bytes(b"existing audio")
    reference = tmp_path / "existing.txt"
    reference.write_bytes(REFERENCE)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("Hello world\n", encoding="utf-8")

    result = run_repair_workflow(
        runner.settings,
        audio_path=audio,
        song_path=reference,
        lyrics_path=lyrics,
        output_root=tmp_path / "exports",
        ui_overrides={"transcription.model": "small"},
        projects=projects,
        artifacts=artifacts,
        runner=runner,
    )

    assert result.reference_song_path.read_bytes() == REFERENCE
    assert result.lyrics_path.read_text(encoding="utf-8") == "Hello world\n"
    assert parse_ultrastar_file(result.updated_song_path).notes
    assert result.archive_path.is_file()
    with zipfile.ZipFile(result.archive_path) as archive:
        assert any(name.endswith(".txt") for name in archive.namelist())
    report = result.report_path.read_text(encoding="utf-8")
    assert "Reference:2" in report
    scores = json.loads(result.scores_path.read_text(encoding="utf-8"))
    assert scores["run_id"] == result.run_id
    assert scores["similarity"]["matched_notes"] == 2
    assert scores["validation"]["passed"] is True
    assert scores["validation_thresholds"]["minimum_match_ratio"] == 0.5


def test_repair_cli_parses_files_and_ui_overrides(monkeypatch, tmp_path: Path, capsys) -> None:
    audio = tmp_path / "song.mp3"
    song = tmp_path / "song.txt"
    captured: dict[str, object] = {}
    similarity = _similarity()
    outcome = ValidationOutcome(True, (), similarity)
    exported = tmp_path / "exported"
    expected = RepairWorkflowResult(
        project_id="prj_test",
        run_id="run_test",
        export_directory=exported,
        reference_song_path=exported / "original.txt",
        lyrics_path=exported / "lyrics.txt",
        updated_song_path=exported / "updated.txt",
        archive_path=exported / "updated.zip",
        report_path=exported / "report.html",
        scores_path=exported / "scores.json",
        similarity=similarity,
        validation_outcome=outcome,
    )

    def fake_workflow(settings, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("ultrasongs.cli.run_repair_workflow", fake_workflow)
    exit_code = main(
        [
            "repair",
            "--audio",
            str(audio),
            "--song",
            str(song),
            "--set",
            "transcription.model=small",
            "--set",
            "pitch.confidence_thresholds=[0.6,0.3]",
        ]
    )

    assert exit_code == 0
    assert captured["audio_path"] == audio
    assert captured["song_path"] == song
    assert captured["ui_overrides"] == {
        "transcription.model": "small",
        "pitch.confidence_thresholds": [0.6, 0.3],
    }
    assert "Configured validation: PASSED" in capsys.readouterr().out


def _similarity() -> SimilarityResult:
    return SimilarityResult(
        reference_notes=2,
        candidate_notes=2,
        matched_notes=2,
        matched_ratio=1.0,
        reference_coverage=1.0,
        candidate_coverage=1.0,
        timing_rmse_ms=10.0,
        duration_rmse_ms=20.0,
        pitch_distance_semitones=0.5,
        timing_median_error_ms=10.0,
        timing_max_error_ms=10.0,
        duration_median_error_ms=20.0,
        duration_max_error_ms=20.0,
        pitch_median_distance_semitones=0.0,
        pitch_max_distance_semitones=1.0,
    )
