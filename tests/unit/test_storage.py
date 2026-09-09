from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ultrasongs.domain.models import (
    PROJECT_SCHEMA_VERSION,
    PipelineStageResult,
    PipelineStageStatus,
    utc_now_iso,
)
from ultrasongs.storage import (
    ArtifactIntegrityError,
    ArtifactRepository,
    ImmutableArtifactError,
    InvalidIdentifierError,
    ProjectRepository,
    RecordNotFoundError,
    SchemaVersionError,
)
from ultrasongs.storage.support import atomic_write_json


def repositories(tmp_path: Path) -> tuple[ProjectRepository, ArtifactRepository]:
    projects = ProjectRepository(tmp_path)
    return projects, ArtifactRepository(tmp_path, projects=projects)


def test_project_round_trip_update_and_listing(tmp_path: Path) -> None:
    projects, _ = repositories(tmp_path)

    created = projects.create(
        title="Diggy Diggy Hole",
        artist="Wind Rose",
        metadata={"mode": "validation"},
    )
    loaded = projects.get(created.project_id)

    assert loaded == created
    assert loaded.project_id.startswith("prj_")
    assert loaded.schema_version == PROJECT_SCHEMA_VERSION
    assert projects.list() == [created]

    updated = projects.update(created.project_id, title="Updated")
    assert projects.get(created.project_id) == updated
    assert updated.title == "Updated"
    assert updated.artist == "Wind Rose"
    assert updated.updated_at >= created.updated_at


def test_project_v0_schema_is_migrated_on_read(tmp_path: Path) -> None:
    projects, _ = repositories(tmp_path)
    project = projects.create()
    project_path = projects.project_directory(project.project_id) / "project.json"
    legacy = project.to_dict()
    legacy.pop("schema_version")
    legacy["id"] = legacy.pop("project_id")
    legacy.pop("metadata")
    atomic_write_json(project_path, legacy)

    loaded = projects.get(project.project_id)

    assert loaded.project_id == project.project_id
    assert loaded.schema_version == 1
    assert loaded.metadata == {}


def test_future_project_schema_is_rejected(tmp_path: Path) -> None:
    projects, _ = repositories(tmp_path)
    project = projects.create()
    future = replace(project, schema_version=PROJECT_SCHEMA_VERSION + 1)

    with pytest.raises(SchemaVersionError):
        projects.save(future)


@pytest.mark.parametrize(
    "unsafe_id",
    ["../outside", "prj_not-hex", "run_00000000000000000000000000000000"],
)
def test_project_ids_are_opaque_and_cannot_traverse(tmp_path: Path, unsafe_id: str) -> None:
    projects, _ = repositories(tmp_path)

    with pytest.raises(InvalidIdentifierError):
        projects.get(unsafe_id)


def test_reference_artifact_is_exact_hashed_and_immutable(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    project = projects.create()
    manifest = artifacts.create_manifest(project.project_id)
    reference_bytes = b"#TITLE:Diggy Diggy Hole\r\n: 0 4 12 Dig~\r\nE\r\n"

    record = artifacts.register_reference(
        project.project_id,
        manifest.run_id,
        reference_bytes,
        original_name="../../reference.txt",
    )

    assert record.kind == "reference_ultrastar"
    assert record.original_name == "reference.txt"
    assert record.immutable is True
    assert (
        artifacts.read_bytes(project.project_id, manifest.run_id, record.artifact_id)
        == reference_bytes
    )
    owned_path = artifacts.get_artifact_path(
        project.project_id, manifest.run_id, record.artifact_id, verify=True
    )
    assert owned_path.is_relative_to(projects.project_directory(project.project_id))
    assert owned_path.name == "reference.txt"
    assert owned_path.parent.name == "reference-ultrastar"
    assert record.relative_path == (
        f"artifacts/{manifest.run_id}/reference-ultrastar/reference.txt"
    )

    with pytest.raises(ImmutableArtifactError):
        artifacts.register_reference(
            project.project_id,
            manifest.run_id,
            b"replacement",
            original_name="replacement.txt",
        )


def test_effective_config_snapshot_is_registered_once(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    project = projects.create()
    config = {
        "schema_version": 1,
        "transcription": {"model": "small", "language": None},
        "alignment": {"engine": "smith_waterman"},
    }

    manifest = artifacts.create_manifest(project.project_id, effective_config=config)
    assert manifest.effective_config_artifact_id is not None
    snapshot = artifacts.read_bytes(
        project.project_id,
        manifest.run_id,
        manifest.effective_config_artifact_id,
    )
    assert json.loads(snapshot) == config

    with pytest.raises(ImmutableArtifactError):
        artifacts.store_effective_config(project.project_id, manifest.run_id, config)


def test_stage_results_and_artifact_ownership_are_persisted(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    first_project = projects.create()
    second_project = projects.create()
    first_run = artifacts.create_manifest(first_project.project_id)
    second_run = artifacts.create_manifest(second_project.project_id)
    candidate = artifacts.register_bytes(
        first_project.project_id,
        first_run.run_id,
        b"#TITLE:Candidate\nE\n",
        kind="candidate_ultrastar",
        original_name="candidate.txt",
    )
    stage = PipelineStageResult(
        stage="export",
        status=PipelineStageStatus.SUCCEEDED,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        artifact_ids=(candidate.artifact_id,),
        metrics={"notes": 42},
    )

    updated = artifacts.record_stage(first_project.project_id, first_run.run_id, stage)

    assert updated.stages == (stage,)
    assert updated.artifacts == (candidate,)
    with pytest.raises(RecordNotFoundError):
        artifacts.get_artifact_path(
            second_project.project_id, second_run.run_id, candidate.artifact_id
        )


def test_integrity_verification_detects_modified_payload(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    project = projects.create()
    manifest = artifacts.create_manifest(project.project_id)
    record = artifacts.register_bytes(
        project.project_id,
        manifest.run_id,
        b"original",
        kind="test",
        original_name="payload.bin",
    )
    path = artifacts.get_artifact_path(project.project_id, manifest.run_id, record.artifact_id)
    path.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError):
        artifacts.read_bytes(project.project_id, manifest.run_id, record.artifact_id)


def test_artifact_paths_are_human_readable_and_collision_safe(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    project = projects.create()
    manifest = artifacts.create_manifest(project.project_id)

    first = artifacts.register_bytes(
        project.project_id,
        manifest.run_id,
        b"first",
        kind="pipeline_report",
        original_name="My Song.html",
        media_type="text/html",
    )
    second = artifacts.register_bytes(
        project.project_id,
        manifest.run_id,
        b"second",
        kind="pipeline_report",
        original_name="My Song.html",
        media_type="text/html",
    )

    first_path = artifacts.get_artifact_path(
        project.project_id, manifest.run_id, first.artifact_id
    )
    second_path = artifacts.get_artifact_path(
        project.project_id, manifest.run_id, second.artifact_id
    )
    assert first_path.relative_to(projects.project_directory(project.project_id)).as_posix() == (
        f"artifacts/{manifest.run_id}/pipeline-report/My Song.html"
    )
    assert second_path.name == "My Song-2.html"
    assert first_path.read_bytes() == b"first"
    assert second_path.read_bytes() == b"second"


def test_missing_project_or_run_is_not_implicitly_created(tmp_path: Path) -> None:
    projects, artifacts = repositories(tmp_path)
    project = projects.create()

    with pytest.raises(RecordNotFoundError):
        artifacts.get_manifest(project.project_id, "run_00000000000000000000000000000000")
