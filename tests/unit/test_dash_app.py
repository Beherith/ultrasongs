from __future__ import annotations

import base64

import pytest
from dash.development.base_component import Component

from ultrasongs.app import create_app, request_content_limit
from ultrasongs.config import AppSettings, PathSettings, SecuritySettings
from ultrasongs.processing.pipeline import PipelineRunner as ProcessingPipelineRunner
from ultrasongs.storage import ArtifactRepository, ProjectRepository
from ultrasongs.web.callbacks.mode import mode_presentation
from ultrasongs.web.callbacks.pipeline import _result_text, build_pipeline_request
from ultrasongs.web.callbacks.reference import (
    inspect_reference_upload,
    reference_summary,
)
from ultrasongs.web.callbacks.settings import (
    collect_ui_overrides,
    default_control_values,
)
from ultrasongs.web.layout import build_advanced_settings, build_layout
from ultrasongs.web.local_submission import LocalSubmissionAdapter, PipelineJobStatus

REFERENCE = (
    "#TITLE:Test Song\n#ARTIST:Test Artist\n#MP3:test.mp3\n#BPM:120\n#GAP:500\n"
    ": 0 4 60 Hel\n: 4 4 62  lo\nE\n"
)


def _data_url(value: str) -> str:
    encoded = base64.b64encode(value.encode()).decode()
    return f"data:text/plain;base64,{encoded}"


def _component_ids(component: Component) -> set[str]:
    found: set[str] = set()
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        found.add(component_id)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, list | tuple):
        children = [children]
    for child in children:
        if isinstance(child, Component):
            found.update(_component_ids(child))
    return found


def test_layout_has_generation_validation_and_lightweight_stores() -> None:
    layout = build_layout(AppSettings())
    ids = _component_ids(layout)

    assert {
        "mode-selector",
        "audio-upload",
        "video-upload",
        "reference-upload",
        "title-input",
        "artist-input",
        "lyrics-input",
        "project-store",
        "run-store",
        "run-poll",
        "progress-placeholder",
        "result-placeholder",
    } <= ids


def test_app_factory_registers_feature_callbacks_and_default_services(tmp_path) -> None:
    settings = AppSettings(
        paths=PathSettings(
            temp_dir=tmp_path / "work",
            projects_dir=tmp_path / "data",
        ),
        security=SecuritySettings(max_upload_megabytes=2),
    )
    app = create_app(settings)
    services = app.server.extensions["ultrasongs"]

    assert app.title == "UltraSongs"
    assert len(app.callback_map) == 5
    assert "run-store.data" in " ".join(app.callback_map)
    assert app.server.config["MAX_CONTENT_LENGTH"] == request_content_limit(settings)
    assert isinstance(services["projects"], ProjectRepository)
    assert isinstance(services["artifacts"], ArtifactRepository)
    assert isinstance(services["pipeline_runner"], ProcessingPipelineRunner)
    assert isinstance(services["submission_adapter"], LocalSubmissionAdapter)
    services["submission_adapter"].shutdown()


def test_artifact_route_checks_ownership_and_serves_registered_name(tmp_path) -> None:
    settings = AppSettings(
        paths=PathSettings(temp_dir=tmp_path / "work", projects_dir=tmp_path / "data")
    )
    projects = ProjectRepository(settings.paths.projects_dir)
    artifacts = ArtifactRepository(settings.paths.projects_dir, projects=projects)
    project = projects.create(title="Download")
    manifest = artifacts.create_manifest(project.project_id)
    record = artifacts.register_bytes(
        project.project_id,
        manifest.run_id,
        b"song text",
        kind="candidate_ultrastar",
        original_name="song.txt",
        media_type="text/plain",
    )
    app = create_app(
        settings,
        projects=projects,
        artifacts=artifacts,
        submission_adapter=object(),  # callbacks are not invoked in this route test
    )
    client = app.server.test_client()

    response = client.get(
        f"/artifacts/{project.project_id}/{manifest.run_id}/{record.artifact_id}"
    )

    assert response.status_code == 200
    assert response.data == b"song text"
    assert "attachment" in response.headers["Content-Disposition"]
    assert "song.txt" in response.headers["Content-Disposition"]
    rejected = client.get(f"/artifacts/not-safe/{manifest.run_id}/{record.artifact_id}")
    assert rejected.status_code == 404


def test_success_result_renders_owned_artifact_links() -> None:
    status = PipelineJobStatus(
        job_id="job_" + "0" * 32,
        project_id="prj_" + "1" * 32,
        state="succeeded",
        message="Done",
        run_id="run_" + "2" * 32,
        artifact_ids={"candidate": "art_" + "3" * 32, "archive": "art_" + "4" * 32},
    )

    result = _result_text(status)
    links = [item.children for item in result.children[1].children]

    assert [link.children for link in links] == ["Download candidate", "Download archive"]
    assert all(link.href.startswith("/artifacts/prj_") for link in links)


def test_mode_presentation_reveals_reference_workflow() -> None:
    generated = mode_presentation("generate")
    validated = mode_presentation("validate")

    assert generated.reference_style == {"display": "none"}
    assert validated.reference_style == {"display": "block"}
    assert "validation" in validated.action_label.lower()


def test_uploaded_reference_prefills_reviewable_metadata_and_lyrics() -> None:
    inspection = inspect_reference_upload(_data_url(REFERENCE), "reference.txt")
    summary = reference_summary(inspection, "reference.txt")

    assert inspection.title == "Test Song"
    assert inspection.artist == "Test Artist"
    assert inspection.reconstructed_lyrics == "Hel lo"
    assert summary["note_count"] == 2
    assert "reconstructed_lyrics" not in summary
    assert "song" not in summary


def test_uploaded_reference_rejects_non_txt_and_invalid_data() -> None:
    with pytest.raises(ValueError, match=r"\.txt"):
        inspect_reference_upload(_data_url(REFERENCE), "reference.json")
    with pytest.raises(ValueError, match="base64"):
        inspect_reference_upload("data:text/plain;base64,!!!", "reference.txt")


def test_advanced_settings_are_generated_from_ui_safe_schema() -> None:
    settings = AppSettings()
    groups = build_advanced_settings(settings)

    assert groups
    assert len(groups) == len({path.split(".", 1)[0] for path in settings.ui_override_paths()})


def test_collect_and_reset_ui_settings_use_central_validation() -> None:
    settings = AppSettings()
    ids = [
        {"type": "setting-input", "path": "transcription.model"},
        {"type": "setting-input", "path": "pitch.confidence_thresholds"},
        {"type": "setting-input", "path": "report.include_pauses"},
    ]
    overrides = collect_ui_overrides(settings, ids, ["small", "[0.6, 0.3]", False])

    assert overrides == {
        "transcription.model": "small",
        "pitch.confidence_thresholds": [0.6, 0.3],
        "report.include_pauses": False,
    }
    assert default_control_values(settings, ids) == ["medium", "[0.5, 0.3, 0.1]", True]


def test_collect_ui_settings_reports_invalid_values() -> None:
    settings = AppSettings()
    ids = [{"type": "setting-input", "path": "pitch.confidence_thresholds"}]

    with pytest.raises(ValueError, match="Pitch confidence thresholds"):
        collect_ui_overrides(settings, ids, ["[0.3, 0.5]"])


def test_pipeline_request_keeps_effective_settings_and_requires_reference() -> None:
    settings = AppSettings()
    request = build_pipeline_request(
        settings,
        mode="validate",
        title=" Test Song ",
        artist=" Artist ",
        lyrics=" lyrics ",
        audio_contents="data:audio/mpeg;base64,AA==",
        audio_filename="song.mp3",
        video_contents=None,
        video_filename=None,
        reference_contents=_data_url(REFERENCE),
        reference_filename="reference.txt",
        overrides={"transcription.model": "small"},
    )

    assert request.title == "Test Song"
    assert request.settings.settings.transcription.model == "small"
    assert request.audio is not None
    assert request.audio.filename == "song.mp3"

    with pytest.raises(ValueError, match="reference"):
        build_pipeline_request(
            settings,
            mode="validate",
            title="Test",
            artist="Artist",
            lyrics="lyrics",
            audio_contents="data:audio/mpeg;base64,AA==",
            audio_filename="song.mp3",
            video_contents=None,
            video_filename=None,
            reference_contents=None,
            reference_filename=None,
            overrides=None,
        )
