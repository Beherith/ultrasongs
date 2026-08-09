"""Shared construction of application repositories and pipeline services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultrasongs.config import AppSettings
from ultrasongs.processing.pipeline import PipelineRunner
from ultrasongs.storage import ArtifactRepository, ProjectRepository


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    projects: ProjectRepository
    artifacts: ArtifactRepository
    pipeline_runner: PipelineRunner


def build_services(settings: AppSettings) -> ApplicationServices:
    """Build the default local services used by Dash and command-line workflows."""

    project_root = Path(settings.paths.projects_dir)
    projects = ProjectRepository(project_root)
    artifacts = ArtifactRepository(project_root, projects=projects)
    return ApplicationServices(
        projects=projects,
        artifacts=artifacts,
        pipeline_runner=PipelineRunner(settings, projects, artifacts),
    )


__all__ = ["ApplicationServices", "build_services"]
