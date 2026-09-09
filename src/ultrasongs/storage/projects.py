"""Versioned filesystem repository for top-level projects."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ultrasongs.domain.models import PROJECT_SCHEMA_VERSION, Project, utc_now_iso

from .support import (
    RecordNotFoundError,
    SchemaVersionError,
    atomic_write_json,
    contained_path,
    new_opaque_id,
    read_json,
    validate_opaque_id,
)


class ProjectRepository:
    """Store project metadata beneath a caller-selected application data root."""

    def __init__(self, root: str | Path) -> None:
        # ``root`` is the centrally configured projects directory itself.
        self.root = Path(root).resolve()
        self.projects_root = self.root
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(
        self,
        *,
        title: str = "",
        artist: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        now = utc_now_iso()
        project = Project(
            project_id=new_opaque_id("prj"),
            created_at=now,
            updated_at=now,
            title=title,
            artist=artist,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self.save(project, require_existing=False)
        return project

    def save(self, project: Project, *, require_existing: bool = True) -> None:
        validate_opaque_id(project.project_id, "prj")
        if project.schema_version != PROJECT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Cannot write project schema {project.schema_version}; "
                f"expected {PROJECT_SCHEMA_VERSION}"
            )
        record_path = self._record_path(project.project_id)
        with self._lock:
            if require_existing and not record_path.is_file():
                raise RecordNotFoundError(f"Project does not exist: {project.project_id}")
            atomic_write_json(record_path, project.to_dict())

    def get(self, project_id: str) -> Project:
        validate_opaque_id(project_id, "prj")
        with self._lock:
            payload = self._migrate(read_json(self._record_path(project_id)))
        project = Project.from_dict(payload)
        if project.project_id != project_id:
            raise SchemaVersionError("Project record identifier does not match its location")
        return project

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for record_path in sorted(self.projects_root.glob("prj_*/project.json")):
            projects.append(self.get(record_path.parent.name))
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def update(
        self,
        project_id: str,
        *,
        title: str | None = None,
        artist: str | None = None,
        latest_run_id: str | None = None,
        reference_artifact_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        """Update supplied fields and return the newly persisted project.

        Optional ID fields are left unchanged when omitted.  Dedicated clear
        operations can be added with a schema migration if the UI needs them.
        """

        with self._lock:
            project = self.get(project_id)
            updated = replace(
                project,
                title=project.title if title is None else title,
                artist=project.artist if artist is None else artist,
                latest_run_id=(project.latest_run_id if latest_run_id is None else latest_run_id),
                reference_artifact_id=(
                    project.reference_artifact_id
                    if reference_artifact_id is None
                    else reference_artifact_id
                ),
                metadata=project.metadata if metadata is None else dict(metadata),
                updated_at=utc_now_iso(),
            )
            self.save(updated)
        return updated

    def exists(self, project_id: str) -> bool:
        validate_opaque_id(project_id, "prj")
        return self._record_path(project_id).is_file()

    def project_directory(self, project_id: str) -> Path:
        """Return the internal project directory after verifying ownership."""

        validate_opaque_id(project_id, "prj")
        path = contained_path(self.projects_root, project_id)
        if not contained_path(path, "project.json").is_file():
            raise RecordNotFoundError(f"Project does not exist: {project_id}")
        return path

    def _record_path(self, project_id: str) -> Path:
        return contained_path(self.projects_root, project_id, "project.json")

    @staticmethod
    def _migrate(payload: dict[str, Any]) -> dict[str, Any]:
        version = int(payload.get("schema_version", 0))
        if version > PROJECT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Project schema {version} is newer than supported version {PROJECT_SCHEMA_VERSION}"
            )
        if version == 0:
            migrated = dict(payload)
            if "project_id" not in migrated and "id" in migrated:
                migrated["project_id"] = migrated.pop("id")
            timestamp = migrated.get("updated_at") or migrated.get("created_at") or utc_now_iso()
            migrated.setdefault("created_at", timestamp)
            migrated.setdefault("updated_at", timestamp)
            migrated.setdefault("title", "")
            migrated.setdefault("artist", "")
            migrated.setdefault("latest_run_id", None)
            migrated.setdefault("reference_artifact_id", None)
            migrated.setdefault("metadata", {})
            migrated["schema_version"] = 1
            payload = migrated
        return payload
