from __future__ import annotations

import base64
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ultrasongs.config import AppSettings, PathSettings, SecuritySettings
from ultrasongs.storage import ArtifactRepository, ProjectRepository
from ultrasongs.web.callbacks.pipeline import BrowserUpload, PipelineRequest
from ultrasongs.web.local_submission import (
    LocalSubmissionAdapter,
    PipelineJobStatus,
    decode_upload_data_url,
)


def _upload(payload: bytes, filename: str, media_type: str = "audio/mpeg") -> BrowserUpload:
    encoded = base64.b64encode(payload).decode("ascii")
    return BrowserUpload(filename, f"data:{media_type};base64,{encoded}")


def _settings(tmp_path: Path, **security: Any) -> AppSettings:
    return AppSettings(
        paths=PathSettings(
            temp_dir=tmp_path / "work",
            projects_dir=tmp_path / "data",
        ),
        security=SecuritySettings(
            max_upload_megabytes=security.pop("max_upload_megabytes", 1),
            max_concurrent_jobs=security.pop("max_concurrent_jobs", 1),
            **security,
        ),
    )


def _request(
    settings: AppSettings,
    *,
    audio: BrowserUpload | None = None,
    video: BrowserUpload | None = None,
    reference: BrowserUpload | None = None,
) -> PipelineRequest:
    return PipelineRequest(
        mode="validate" if reference else "generate",
        title="Safe Song",
        artist="Tester",
        lyrics="Hello world",
        audio=audio,
        video=video,
        reference=reference,
        settings=settings.effective_snapshot({"transcription.model": "small"}),
    )


class FakeRunner:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Any:
        source = Path(kwargs["source_path"])
        self.calls.append({**kwargs, "source_bytes": source.read_bytes()})
        if self.failure:
            raise self.failure
        return SimpleNamespace(
            project_id=kwargs["project_id"],
            run_id="run_" + "1" * 32,
            candidate_artifact_id="art_" + "2" * 32,
            archive_artifact_id="art_" + "3" * 32,
            report_artifact_id="art_" + "4" * 32,
        )


def _adapter(
    tmp_path: Path,
    *,
    runner: FakeRunner | None = None,
    settings: AppSettings | None = None,
) -> tuple[LocalSubmissionAdapter, FakeRunner, AppSettings]:
    settings = settings or _settings(tmp_path)
    projects = ProjectRepository(settings.paths.projects_dir)
    artifacts = ArtifactRepository(settings.paths.projects_dir, projects=projects)
    runner = runner or FakeRunner()
    adapter = LocalSubmissionAdapter(settings, projects, artifacts, runner)
    return adapter, runner, settings


def _wait_for_terminal(adapter: LocalSubmissionAdapter, job_id: str) -> PipelineJobStatus:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = adapter.status(job_id)
        if status.terminal:
            return status
        time.sleep(0.01)
    raise AssertionError("background pipeline did not finish")


def test_strict_data_url_decoder_rejects_invalid_base64_and_non_base64() -> None:
    assert decode_upload_data_url("data:audio/mpeg;base64,SGVsbG8=") == b"Hello"

    with pytest.raises(ValueError, match="valid base64"):
        decode_upload_data_url("data:audio/mpeg;base64,%%%")
    with pytest.raises(ValueError, match="base64 data URL"):
        decode_upload_data_url("data:text/plain,hello")


def test_submission_rejects_extensions_and_decoded_size(tmp_path: Path) -> None:
    adapter, _runner, settings = _adapter(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unsupported audio extension"):
            adapter.submit(_request(settings, audio=_upload(b"audio", "song.exe")))
        with pytest.raises(ValueError, match="Unsupported reference extension"):
            adapter.submit(
                _request(
                    settings,
                    audio=_upload(b"audio", "song.mp3"),
                    reference=_upload(b"reference", "reference.json", "application/json"),
                )
            )
        oversized = b"x" * (1024 * 1024 + 1)
        with pytest.raises(ValueError, match="1 MB limit"):
            adapter.submit(_request(settings, audio=_upload(oversized, "song.mp3")))
    finally:
        adapter.shutdown()


def test_successful_job_uses_opaque_contained_file_and_reports_artifacts(
    tmp_path: Path,
) -> None:
    adapter, runner, settings = _adapter(tmp_path)
    try:
        submission = adapter.submit(
            _request(
                settings,
                audio=_upload(b"real audio", "../../unsafe name.mp3"),
                video=_upload(b"real video", "unsafe video.mp4", "video/mp4"),
                reference=_upload(
                    b"#TITLE:Reference\n#ARTIST:Tester\n#BPM:120\n#GAP:0\nE\n",
                    "reference.txt",
                    "text/plain",
                ),
            )
        )
        status = _wait_for_terminal(adapter, submission.job_id)

        assert status.state == "succeeded"
        assert status.run_id == "run_" + "1" * 32
        assert status.artifact_ids == {
            "candidate": "art_" + "2" * 32,
            "archive": "art_" + "3" * 32,
            "report": "art_" + "4" * 32,
        }
        assert runner.calls[0]["source_bytes"] == b"real audio"
        source = Path(runner.calls[0]["source_path"])
        video = Path(runner.calls[0]["video_path"])
        assert source.name.startswith("upl_")
        assert source.suffix == ".mp3"
        assert video.name.startswith("upl_")
        assert video.suffix == ".mp4"
        assert not source.exists()
        assert not video.exists()
        assert list(adapter.upload_root.iterdir()) == []
        assert adapter.projects.exists(submission.project_id)
        assert runner.calls[0]["validation"].original_name == "reference.txt"
        assert runner.calls[0]["ui_overrides"] == {"transcription.model": "small"}
    finally:
        adapter.shutdown()


def test_failed_background_job_exposes_error_and_cleans_uploads(tmp_path: Path) -> None:
    failure = RuntimeError("model exploded")
    adapter, _runner, settings = _adapter(tmp_path, runner=FakeRunner(failure=failure))
    try:
        submission = adapter.submit(_request(settings, audio=_upload(b"audio", "song.mp3")))
        status = _wait_for_terminal(adapter, submission.job_id)

        assert status.state == "failed"
        assert status.error == "model exploded"
        assert status.artifact_ids == {}
        assert list(adapter.upload_root.iterdir()) == []
    finally:
        adapter.shutdown()


def test_video_only_job_does_not_pass_source_as_explicit_video(tmp_path: Path) -> None:
    adapter, runner, settings = _adapter(tmp_path)
    try:
        submission = adapter.submit(
            _request(settings, video=_upload(b"video", "song.mp4", "video/mp4"))
        )
        status = _wait_for_terminal(adapter, submission.job_id)

        assert status.state == "succeeded"
        assert Path(runner.calls[0]["source_path"]).suffix == ".mp4"
        assert runner.calls[0]["video_path"] is None
    finally:
        adapter.shutdown()


def test_executor_honors_configured_concurrency(tmp_path: Path) -> None:
    settings = _settings(tmp_path, max_concurrent_jobs=2)
    adapter, _runner, _settings_value = _adapter(tmp_path, settings=settings)
    try:
        assert adapter._executor._max_workers == 2
    finally:
        adapter.shutdown()
