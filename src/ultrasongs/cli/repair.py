"""Repair an existing MP3 + UltraStar TXT pair through the complete pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultrasongs.config import AppSettings
from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.domain.ultrastar import safe_filename
from ultrasongs.domain.validation import ValidationOutcome, inspect_reference_file
from ultrasongs.processing.pipeline import PipelineRunner, PipelineRunResult, ValidationInput
from ultrasongs.services import build_services
from ultrasongs.storage import ArtifactRepository, ProjectRepository


@dataclass(frozen=True, slots=True)
class RepairWorkflowResult:
    """User-facing outputs from one repair and comparison run."""

    project_id: str
    run_id: str
    export_directory: Path
    reference_song_path: Path
    lyrics_path: Path
    updated_song_path: Path
    archive_path: Path
    report_path: Path
    scores_path: Path
    similarity: SimilarityResult
    validation_outcome: ValidationOutcome | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "run_id": self.run_id,
            "export_directory": str(self.export_directory),
            "artifacts": {
                "reference_song": str(self.reference_song_path),
                "lyrics_used": str(self.lyrics_path),
                "updated_song": str(self.updated_song_path),
                "archive": str(self.archive_path),
                "report": str(self.report_path),
                "scores": str(self.scores_path),
            },
            "similarity": self.similarity.to_dict(),
            "validation": (
                self.validation_outcome.to_dict()
                if self.validation_outcome is not None
                else None
            ),
        }


def run_repair_workflow(
    settings: AppSettings,
    *,
    audio_path: str | Path,
    song_path: str | Path,
    output_root: str | Path | None = None,
    lyrics_path: str | Path | None = None,
    title: str | None = None,
    artist: str | None = None,
    ui_overrides: dict[str, Any] | None = None,
    projects: ProjectRepository | None = None,
    artifacts: ArtifactRepository | None = None,
    runner: PipelineRunner | None = None,
) -> RepairWorkflowResult:
    """Run media processing, regenerate the chart, score it, and export a review bundle."""

    audio = Path(audio_path).resolve()
    reference_path = Path(song_path).resolve()
    _require_file(audio, "audio")
    _require_file(reference_path, "UltraStar song")
    if reference_path.suffix.lower() != ".txt":
        raise ValueError("The existing UltraStar song must be a .txt file")

    snapshot = settings.effective_snapshot(ui_overrides)
    effective = snapshot.settings
    if audio.suffix.lower() not in effective.security.allowed_audio_extensions:
        allowed = ", ".join(effective.security.allowed_audio_extensions)
        raise ValueError(f"Unsupported audio extension {audio.suffix!r}; expected one of {allowed}")

    reference = inspect_reference_file(reference_path)
    selected_title = (title or reference.title or reference_path.stem).strip()
    selected_artist = (artist or reference.artist or "Unknown Artist").strip()
    selected_lyrics = (
        _read_lyrics_file(lyrics_path)
        if lyrics_path is not None
        else reference.reconstructed_lyrics.strip()
    )
    if not selected_title:
        raise ValueError("A title is required; pass --title when the reference has none")
    if not selected_lyrics:
        raise ValueError(
            "No lyrics could be reconstructed from the reference; pass --lyrics-file"
        )

    if projects is None and artifacts is None and runner is None:
        services = build_services(settings)
        projects = services.projects
        artifacts = services.artifacts
        runner = services.pipeline_runner
    else:
        project_root = Path(effective.paths.projects_dir)
        projects = projects or ProjectRepository(project_root)
        artifacts = artifacts or ArtifactRepository(project_root, projects=projects)
        runner = runner or PipelineRunner(settings, projects, artifacts)
    project = projects.create(
        title=selected_title,
        artist=selected_artist,
        metadata={
            "workflow": "repair",
            "source_audio": str(audio),
            "source_ultrastar": str(reference_path),
        },
    )
    pipeline_result = runner.run(
        project_id=project.project_id,
        source_path=audio,
        title=selected_title,
        artist=selected_artist,
        lyrics=selected_lyrics,
        ui_overrides=ui_overrides,
        validation=ValidationInput.from_path(reference_path),
    )
    return _export_repair_bundle(
        effective,
        reference_path=reference_path,
        lyrics=selected_lyrics,
        title=selected_title,
        pipeline_result=pipeline_result,
        artifacts=artifacts,
        output_root=output_root,
    )


def _export_repair_bundle(
    settings: AppSettings,
    *,
    reference_path: Path,
    lyrics: str,
    title: str,
    pipeline_result: PipelineRunResult,
    artifacts: ArtifactRepository,
    output_root: str | Path | None,
) -> RepairWorkflowResult:
    similarity = pipeline_result.similarity
    if similarity is None:
        raise RuntimeError("The repair pipeline completed without similarity scores")
    if pipeline_result.report_artifact_id is None:
        raise RuntimeError("The repair pipeline completed without an HTML report")

    root = Path(output_root or settings.paths.exports_dir).resolve()
    bundle = root / f"repair-{safe_filename(title)}-{pipeline_result.run_id}"
    bundle.mkdir(parents=True, exist_ok=False)
    stem = safe_filename(title)
    reference_export = bundle / f"{stem}-original.txt"
    lyrics_export = bundle / f"{stem}-lyrics-used.txt"
    updated_export = bundle / f"{stem}-updated.txt"
    archive_export = bundle / f"{stem}-updated.zip"
    report_export = bundle / f"{stem}-comparison.html"
    scores_export = bundle / f"{stem}-scores.json"

    shutil.copyfile(reference_path, reference_export)
    lyrics_export.write_text(lyrics.rstrip() + "\n", encoding="utf-8")
    _copy_artifact(
        artifacts, pipeline_result, pipeline_result.candidate_artifact_id, updated_export
    )
    _copy_artifact(
        artifacts, pipeline_result, pipeline_result.archive_artifact_id, archive_export
    )
    _copy_artifact(
        artifacts, pipeline_result, pipeline_result.report_artifact_id, report_export
    )

    result = RepairWorkflowResult(
        project_id=pipeline_result.project_id,
        run_id=pipeline_result.run_id,
        export_directory=bundle,
        reference_song_path=reference_export,
        lyrics_path=lyrics_export,
        updated_song_path=updated_export,
        archive_path=archive_export,
        report_path=report_export,
        scores_path=scores_export,
        similarity=similarity,
        validation_outcome=pipeline_result.validation_outcome,
    )
    scores_document = result.to_dict()
    scores_document["validation_thresholds"] = settings.validation.model_dump(mode="json")
    scores_export.write_text(
        json.dumps(scores_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _copy_artifact(
    artifacts: ArtifactRepository,
    result: PipelineRunResult,
    artifact_id: str,
    destination: Path,
) -> None:
    source = artifacts.get_artifact_path(
        result.project_id, result.run_id, artifact_id, verify=True
    )
    shutil.copyfile(source, destination)


def _read_lyrics_file(path: str | Path) -> str:
    source = Path(path).resolve()
    _require_file(source, "lyrics")
    try:
        return source.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("The lyrics file must be UTF-8 encoded") from exc


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"The {label} file does not exist: {path}")


__all__ = ["RepairWorkflowResult", "run_repair_workflow"]
