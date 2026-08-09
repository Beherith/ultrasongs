"""Immutable artifact files and versioned per-run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from ultrasongs.domain.models import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactRecord,
    PipelineStageResult,
    utc_now_iso,
)

from .projects import ProjectRepository
from .support import (
    ArtifactIntegrityError,
    ImmutableArtifactError,
    RecordNotFoundError,
    SchemaVersionError,
    StorageError,
    atomic_write_json,
    contained_path,
    new_opaque_id,
    read_json,
    validate_opaque_id,
)

_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


class ArtifactRepository:
    """Own artifacts under ``projects/<project>/artifacts/<run>``.

    Every stored payload is immutable and content-hashed.  Paths are resolved
    only after project/run/artifact ownership has been checked in a manifest.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        projects: ProjectRepository | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.projects = projects or ProjectRepository(self.root)
        if self.projects.root != self.root:
            raise ValueError("Project and artifact repositories must share a root")
        self._lock = threading.RLock()

    def create_manifest(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        effective_config: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        project_dir = self.projects.project_directory(project_id)
        run_id = run_id or new_opaque_id("run")
        validate_opaque_id(run_id, "run")
        now = utc_now_iso()
        manifest = ArtifactManifest(
            manifest_id=new_opaque_id("man"),
            project_id=project_id,
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )
        manifest_path = contained_path(project_dir, "manifests", f"{run_id}.json")
        with self._lock:
            if manifest_path.exists():
                raise StorageError(f"Run already exists: {run_id}")
            atomic_write_json(manifest_path, manifest.to_dict())
            if effective_config is not None:
                self.store_effective_config(project_id, run_id, effective_config)
                manifest = self.get_manifest(project_id, run_id)
        return manifest

    def get_manifest(self, project_id: str, run_id: str) -> ArtifactManifest:
        manifest_path = self._manifest_path(project_id, run_id)
        with self._lock:
            payload = self._migrate_manifest(read_json(manifest_path))
        manifest = ArtifactManifest.from_dict(payload)
        if manifest.project_id != project_id or manifest.run_id != run_id:
            raise SchemaVersionError("Manifest ownership does not match its location")
        return manifest

    def list_manifests(self, project_id: str) -> list[ArtifactManifest]:
        project_dir = self.projects.project_directory(project_id)
        manifests_dir = contained_path(project_dir, "manifests")
        if not manifests_dir.exists():
            return []
        manifests = [
            self.get_manifest(project_id, path.stem)
            for path in sorted(manifests_dir.glob("run_*.json"))
        ]
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def register_bytes(
        self,
        project_id: str,
        run_id: str,
        data: bytes,
        *,
        kind: str,
        original_name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        from io import BytesIO

        with BytesIO(data) as source:
            return self._register_stream(
                project_id,
                run_id,
                source,
                kind=kind,
                original_name=original_name,
                media_type=media_type,
                metadata=metadata,
            )

    def register_file(
        self,
        project_id: str,
        run_id: str,
        source_path: str | Path,
        *,
        kind: str,
        original_name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Copy a trusted server-side path into project-owned storage."""

        source_path = Path(source_path)
        try:
            with source_path.open("rb") as source:
                return self._register_stream(
                    project_id,
                    run_id,
                    source,
                    kind=kind,
                    original_name=original_name or source_path.name,
                    media_type=media_type,
                    metadata=metadata,
                )
        except FileNotFoundError as exc:
            raise RecordNotFoundError(f"Artifact source does not exist: {source_path}") from exc

    def register_reference(
        self,
        project_id: str,
        run_id: str,
        source: bytes | str | Path,
        *,
        original_name: str | None = None,
        media_type: str = "text/plain",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Register exact uploaded Ultrastar bytes once for this run."""

        with self._lock:
            manifest = self.get_manifest(project_id, run_id)
            if any(artifact.kind == "reference_ultrastar" for artifact in manifest.artifacts):
                raise ImmutableArtifactError("This run already has an immutable reference song")
            if isinstance(source, bytes):
                return self.register_bytes(
                    project_id,
                    run_id,
                    source,
                    kind="reference_ultrastar",
                    original_name=original_name or "reference.txt",
                    media_type=media_type,
                    metadata=metadata,
                )
            return self.register_file(
                project_id,
                run_id,
                source,
                kind="reference_ultrastar",
                original_name=original_name,
                media_type=media_type,
                metadata=metadata,
            )

    def store_effective_config(
        self,
        project_id: str,
        run_id: str,
        effective_config: Mapping[str, Any],
    ) -> ArtifactRecord:
        """Persist the run's complete effective configuration exactly once."""

        with self._lock:
            manifest = self.get_manifest(project_id, run_id)
            if manifest.effective_config_artifact_id is not None:
                raise ImmutableArtifactError("Effective configuration is already frozen")
            try:
                payload = (
                    json.dumps(
                        effective_config,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
            except (TypeError, ValueError) as exc:
                raise StorageError(
                    f"Effective configuration is not JSON serializable: {exc}"
                ) from exc
            record = self.register_bytes(
                project_id,
                run_id,
                payload,
                kind="effective_config",
                original_name="effective-config.json",
                media_type="application/json",
            )
            manifest = self.get_manifest(project_id, run_id)
            self._save_manifest(
                replace(
                    manifest,
                    effective_config_artifact_id=record.artifact_id,
                    updated_at=utc_now_iso(),
                )
            )
        return record

    def record_stage(
        self,
        project_id: str,
        run_id: str,
        result: PipelineStageResult,
    ) -> ArtifactManifest:
        """Insert or replace the current result for one named stage."""

        with self._lock:
            manifest = self.get_manifest(project_id, run_id)
            stages = [stage for stage in manifest.stages if stage.stage != result.stage]
            stages.append(result)
            manifest = replace(
                manifest,
                stages=tuple(stages),
                updated_at=utc_now_iso(),
            )
            self._save_manifest(manifest)
            return manifest

    def get_artifact_path(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> Path:
        """Resolve a registered artifact; arbitrary relative paths are rejected."""

        validate_opaque_id(artifact_id, "art")
        manifest = self.get_manifest(project_id, run_id)
        record = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if record is None:
            raise RecordNotFoundError(f"Artifact does not belong to run: {artifact_id}")
        project_dir = self.projects.project_directory(project_id)
        path = contained_path(project_dir, *record.relative_path.split("/"))
        if not path.is_file():
            raise RecordNotFoundError(f"Artifact payload is missing: {artifact_id}")
        if verify:
            self._verify(path, record)
        return path

    def read_bytes(
        self, project_id: str, run_id: str, artifact_id: str, *, verify: bool = True
    ) -> bytes:
        return self.get_artifact_path(project_id, run_id, artifact_id, verify=verify).read_bytes()

    def _register_stream(
        self,
        project_id: str,
        run_id: str,
        source: BinaryIO,
        *,
        kind: str,
        original_name: str | None,
        media_type: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> ArtifactRecord:
        if not kind.strip():
            raise ValueError("Artifact kind cannot be empty")
        with self._lock:
            manifest = self.get_manifest(project_id, run_id)
            artifact_id = new_opaque_id("art")
            project_dir = self.projects.project_directory(project_id)
            artifact_dir = contained_path(project_dir, "artifacts", run_id, artifact_id)
            artifact_dir.mkdir(parents=True, exist_ok=False)
            safe_name = Path(original_name).name if original_name else None
            suffix = Path(safe_name).suffix if safe_name else ""
            if not _SAFE_SUFFIX_RE.fullmatch(suffix):
                suffix = ""
            destination = contained_path(artifact_dir, f"content{suffix.lower()}")
            digest = hashlib.sha256()
            size_bytes = 0
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "wb",
                    dir=artifact_dir,
                    prefix=".content.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_name = temporary.name
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)

            relative_path = destination.relative_to(project_dir).as_posix()
            record = ArtifactRecord(
                artifact_id=artifact_id,
                kind=kind,
                relative_path=relative_path,
                sha256=digest.hexdigest(),
                size_bytes=size_bytes,
                created_at=utc_now_iso(),
                original_name=safe_name,
                media_type=media_type,
                immutable=True,
                metadata=dict(metadata or {}),
            )
            self._save_manifest(
                replace(
                    manifest,
                    artifacts=(*manifest.artifacts, record),
                    updated_at=utc_now_iso(),
                )
            )
            return record

    def _manifest_path(self, project_id: str, run_id: str) -> Path:
        validate_opaque_id(run_id, "run")
        project_dir = self.projects.project_directory(project_id)
        return contained_path(project_dir, "manifests", f"{run_id}.json")

    def _save_manifest(self, manifest: ArtifactManifest) -> None:
        if manifest.schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Cannot write manifest schema {manifest.schema_version}; expected "
                f"{ARTIFACT_MANIFEST_SCHEMA_VERSION}"
            )
        atomic_write_json(
            self._manifest_path(manifest.project_id, manifest.run_id),
            manifest.to_dict(),
        )

    @staticmethod
    def _verify(path: Path, record: ArtifactRecord) -> None:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes != record.size_bytes or digest.hexdigest() != record.sha256:
            raise ArtifactIntegrityError(
                f"Artifact payload failed integrity verification: {record.artifact_id}"
            )

    @staticmethod
    def _migrate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
        version = int(payload.get("schema_version", 0))
        if version > ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Manifest schema {version} is newer than supported version "
                f"{ARTIFACT_MANIFEST_SCHEMA_VERSION}"
            )
        if version == 0:
            migrated = dict(payload)
            if "manifest_id" not in migrated and "id" in migrated:
                migrated["manifest_id"] = migrated.pop("id")
            timestamp = migrated.get("updated_at") or migrated.get("created_at") or utc_now_iso()
            migrated.setdefault("created_at", timestamp)
            migrated.setdefault("updated_at", timestamp)
            migrated.setdefault("artifacts", [])
            migrated.setdefault("stages", [])
            migrated.setdefault("effective_config_artifact_id", None)
            migrated["schema_version"] = 1
            payload = migrated
        return payload
