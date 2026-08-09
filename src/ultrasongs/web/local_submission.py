"""Secure local background submission for the Dash application."""

from __future__ import annotations

import base64
import binascii
import shutil
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

from ultrasongs.config import AppSettings
from ultrasongs.processing.pipeline import PipelineRunResult, ValidationInput
from ultrasongs.storage import ArtifactRepository, ProjectRepository
from ultrasongs.storage.support import contained_path, new_opaque_id, validate_opaque_id

from .callbacks.pipeline import BrowserUpload, PipelineRequest, PipelineSubmission

JobState = Literal["queued", "running", "succeeded", "failed"]
_REFERENCE_EXTENSIONS = frozenset({".txt"})


class ConcretePipelineRunner(Protocol):
    def run(
        self,
        *,
        project_id: str,
        source_path: str | Path,
        video_path: str | Path | None = None,
        title: str,
        artist: str,
        lyrics: str,
        ui_overrides: Mapping[str, object] | None = None,
        validation: ValidationInput | None = None,
    ) -> PipelineRunResult: ...


@dataclass(frozen=True, slots=True)
class PipelineJobStatus:
    job_id: str
    project_id: str
    state: JobState
    message: str
    run_id: str | None = None
    artifact_ids: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"succeeded", "failed"}

    def to_store(self) -> dict[str, object]:
        """Return the small polling payload that is safe for ``dcc.Store``."""

        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "status": self.state,
            "message": self.message,
            "artifact_ids": dict(self.artifact_ids),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _PreparedJob:
    request: PipelineRequest
    job_id: str
    project_id: str
    directory: Path
    source_path: Path
    video_path: Path | None
    reference_path: Path | None
    reference_name: str | None


class LocalSubmissionAdapter:
    """Validate uploads and run the concrete pipeline on a bounded local pool."""

    def __init__(
        self,
        settings: AppSettings,
        projects: ProjectRepository,
        artifacts: ArtifactRepository,
        runner: ConcretePipelineRunner,
        *,
        upload_root: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.projects = projects
        self.artifacts = artifacts
        self.runner = runner
        configured_root = Path(upload_root or settings.paths.temp_dir).resolve()
        self.upload_root = contained_path(configured_root, "uploads")
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.security.max_concurrent_jobs,
            thread_name_prefix="ultrasongs",
        )
        self._jobs: dict[str, PipelineJobStatus] = {}
        self._lock = threading.RLock()

    def submit(self, request: PipelineRequest) -> PipelineSubmission:
        """Persist a validated submission, create its project, and enqueue it."""

        prepared = self._prepare(request)
        initial = PipelineJobStatus(
            job_id=prepared.job_id,
            project_id=prepared.project_id,
            state="queued",
            message="Queued for local processing.",
        )
        with self._lock:
            self._jobs[prepared.job_id] = initial
        try:
            self._executor.submit(self._execute, prepared)
        except Exception:
            with self._lock:
                self._jobs.pop(prepared.job_id, None)
            shutil.rmtree(prepared.directory, ignore_errors=True)
            raise
        return PipelineSubmission(
            job_id=prepared.job_id,
            project_id=prepared.project_id,
            message=initial.message,
        )

    def status(self, job_id: str) -> PipelineJobStatus:
        """Return immutable status, augmented with persisted stage progress."""

        validate_opaque_id(job_id, "job")
        with self._lock:
            try:
                current = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError("Unknown pipeline job") from exc
        if current.state != "running":
            return current
        try:
            project = self.projects.get(current.project_id)
            if project.latest_run_id is None:
                return current
            manifest = self.artifacts.get_manifest(current.project_id, project.latest_run_id)
            message = current.message
            if manifest.stages:
                latest = manifest.stages[-1]
                message = f"{latest.stage.replace('_', ' ').title()}: {latest.status.value}"
            return replace(current, run_id=project.latest_run_id, message=message)
        except Exception:
            # A manifest may be between atomic updates. The next poll retries it.
            return current

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _prepare(self, request: PipelineRequest) -> _PreparedJob:
        source_upload = request.audio or request.video
        if source_upload is None:
            raise ValueError("Upload an audio track or video")

        validated: list[tuple[str, BrowserUpload, bytes, frozenset[str]]] = []
        if request.audio is not None:
            validated.append(
                (
                    "audio",
                    request.audio,
                    self._validate_upload(
                        request.audio,
                        frozenset(self.settings.security.allowed_audio_extensions),
                        "audio",
                    ),
                    frozenset(self.settings.security.allowed_audio_extensions),
                )
            )
        if request.video is not None:
            validated.append(
                (
                    "video",
                    request.video,
                    self._validate_upload(
                        request.video,
                        frozenset(self.settings.security.allowed_video_extensions),
                        "video",
                    ),
                    frozenset(self.settings.security.allowed_video_extensions),
                )
            )
        if request.reference is not None:
            validated.append(
                (
                    "reference",
                    request.reference,
                    self._validate_upload(request.reference, _REFERENCE_EXTENSIONS, "reference"),
                    _REFERENCE_EXTENSIONS,
                )
            )

        job_id = new_opaque_id("job")
        job_directory = contained_path(self.upload_root, job_id)
        job_directory.mkdir(parents=False, exist_ok=False)
        written: dict[str, Path] = {}
        try:
            for kind, upload, payload, _extensions in validated:
                suffix = _upload_suffix(upload.filename)
                destination = contained_path(job_directory, f"{new_opaque_id('upl')}{suffix}")
                destination.write_bytes(payload)
                written[kind] = destination
            project = self.projects.create(
                title=request.title,
                artist=request.artist,
                metadata={
                    "mode": request.mode,
                    "source_original_name": source_upload.filename,
                },
            )
        except Exception:
            shutil.rmtree(job_directory, ignore_errors=True)
            raise

        source_kind = "audio" if request.audio is not None else "video"
        reference_name = (
            _safe_display_name(request.reference.filename) if request.reference else None
        )
        return _PreparedJob(
            request=request,
            job_id=job_id,
            project_id=project.project_id,
            directory=job_directory,
            source_path=written[source_kind],
            video_path=(written.get("video") if source_kind == "audio" else None),
            reference_path=written.get("reference"),
            reference_name=reference_name,
        )

    def _validate_upload(
        self,
        upload: BrowserUpload,
        allowed_extensions: frozenset[str],
        label: str,
    ) -> bytes:
        suffix = _upload_suffix(upload.filename)
        if suffix not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise ValueError(f"Unsupported {label} extension {suffix or '(none)'}; use {allowed}")
        payload = decode_upload_data_url(upload.contents)
        maximum = self.settings.security.max_upload_megabytes * 1024 * 1024
        if len(payload) > maximum:
            raise ValueError(
                f"{label.title()} upload exceeds the {self.settings.security.max_upload_megabytes} "
                "MB limit"
            )
        return payload

    def _execute(self, prepared: _PreparedJob) -> None:
        self._replace_status(
            prepared.job_id,
            state="running",
            message="Starting pipeline.",
        )
        terminal: PipelineJobStatus
        try:
            validation = None
            if prepared.reference_path is not None:
                validation = ValidationInput(
                    content=prepared.reference_path.read_bytes(),
                    original_name=prepared.reference_name or "reference.txt",
                )
            result = self.runner.run(
                project_id=prepared.project_id,
                source_path=prepared.source_path,
                video_path=prepared.video_path,
                title=prepared.request.title,
                artist=prepared.request.artist,
                lyrics=prepared.request.lyrics,
                ui_overrides=prepared.request.settings.ui_overrides,
                validation=validation,
            )
            artifact_ids = {
                "candidate": result.candidate_artifact_id,
                "archive": result.archive_artifact_id,
            }
            if result.report_artifact_id is not None:
                artifact_ids["report"] = result.report_artifact_id
            terminal = PipelineJobStatus(
                job_id=prepared.job_id,
                project_id=prepared.project_id,
                state="succeeded",
                message="Pipeline completed successfully.",
                run_id=result.run_id,
                artifact_ids=artifact_ids,
            )
        except Exception as exc:
            run_id = getattr(exc, "run_id", None)
            if run_id is None:
                try:
                    run_id = self.projects.get(prepared.project_id).latest_run_id
                except Exception:
                    run_id = None
            terminal = PipelineJobStatus(
                job_id=prepared.job_id,
                project_id=prepared.project_id,
                state="failed",
                message="Pipeline failed.",
                run_id=run_id,
                error=str(exc),
            )
        finally:
            shutil.rmtree(prepared.directory, ignore_errors=True)
        with self._lock:
            self._jobs[prepared.job_id] = terminal

    def _replace_status(self, job_id: str, **updates: object) -> None:
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], **updates)


def decode_upload_data_url(contents: str) -> bytes:
    """Strictly decode the base64 payload from a Dash ``dcc.Upload`` data URL."""

    if not contents or "," not in contents:
        raise ValueError("Upload data is missing or malformed")
    metadata, encoded = contents.split(",", 1)
    if not metadata.startswith("data:") or ";base64" not in metadata.lower():
        raise ValueError("Upload must be a base64 data URL")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Upload data is not valid base64") from exc


def _upload_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(_safe_display_name(filename)).suffix.lower()


def _safe_display_name(filename: str | None) -> str:
    normalized = (filename or "upload").replace("\\", "/")
    return Path(normalized).name or "upload"


__all__ = [
    "ConcretePipelineRunner",
    "LocalSubmissionAdapter",
    "PipelineJobStatus",
    "decode_upload_data_url",
]
