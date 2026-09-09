"""Dash application factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import Dash

from ultrasongs.config import AppSettings
from ultrasongs.processing.pipeline import PipelineRunner as ProcessingPipelineRunner
from ultrasongs.services import build_services
from ultrasongs.storage import ArtifactRepository, ProjectRepository
from ultrasongs.web.callbacks import (
    register_mode_callbacks,
    register_pipeline_callbacks,
    register_reference_callbacks,
    register_settings_callbacks,
)
from ultrasongs.web.callbacks.pipeline import SubmissionAdapter
from ultrasongs.web.downloads import register_artifact_downloads
from ultrasongs.web.layout import build_layout
from ultrasongs.web.local_submission import LocalSubmissionAdapter


def create_app(
    settings: AppSettings,
    pipeline_runner: Any | None = None,
    *,
    projects: ProjectRepository | None = None,
    artifacts: ArtifactRepository | None = None,
    submission_adapter: SubmissionAdapter | None = None,
) -> Dash:
    """Create a Dash app with real local services unless explicitly injected."""

    assets_folder = Path(__file__).resolve().parent / "web" / "assets"
    app = Dash(
        __name__,
        assets_folder=str(assets_folder),
        suppress_callback_exceptions=False,
        title="UltraSongs",
        update_title="UltraSongs - Processing...",
    )
    # Dash uploads arrive as base64 JSON. Bound the whole request to three maximum-sized
    # uploads plus encoding/JSON overhead; decoded per-file limits are enforced by the adapter.
    app.server.config["MAX_CONTENT_LENGTH"] = request_content_limit(settings)

    if submission_adapter is None:
        if projects is None and artifacts is None and pipeline_runner is None:
            services = build_services(settings)
            projects = services.projects
            artifacts = services.artifacts
            pipeline_runner = services.pipeline_runner
        else:
            data_root = Path(settings.paths.projects_dir)
            projects = projects or ProjectRepository(data_root)
            artifacts = artifacts or ArtifactRepository(data_root, projects=projects)
            pipeline_runner = pipeline_runner or ProcessingPipelineRunner(
                settings,
                projects,
                artifacts,
            )
        submission_adapter = LocalSubmissionAdapter(
            settings,
            projects,
            artifacts,
            pipeline_runner,
        )

    app.server.extensions["ultrasongs"] = {
        "settings": settings,
        "projects": projects,
        "artifacts": artifacts,
        "pipeline_runner": pipeline_runner,
        "submission_adapter": submission_adapter,
    }
    download_artifacts = artifacts or getattr(submission_adapter, "artifacts", None)
    if download_artifacts is not None:
        register_artifact_downloads(app.server, download_artifacts)
    app.layout = lambda: build_layout(settings)
    register_mode_callbacks(app)
    register_reference_callbacks(app, settings)
    register_settings_callbacks(app, settings)
    register_pipeline_callbacks(app, settings, submission_adapter)
    return app


def request_content_limit(settings: AppSettings) -> int:
    """Bound a Dash callback carrying up to audio, video, and reference payloads."""

    maximum_file_bytes = settings.security.max_upload_megabytes * 1024 * 1024
    return maximum_file_bytes * 4 + 1024 * 1024


__all__ = ["create_app", "request_content_limit"]
