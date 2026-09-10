"""Tests for cli/web/jobs.py: JobManager, JobLogHandler, retention, snapshots."""

import dataclasses
import json
import threading
import time
from pathlib import Path

import pytest
from cli.pipeline import ProcessResult
from cli.config import Config
from cli.pipeline import ProcessRequest
from cli.web.jobs import (
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    Job,
    JobManager,
)


def _make_request(tmp: Path) -> ProcessRequest:
    config = Config(temp_dir=str(tmp / "req_tmp"), output_dir=str(tmp / "req_out"))
    return ProcessRequest(
        title="Test Song",
        artist="Tester",
        lyrics_text="Hello",
        input_path=tmp / "orig.mp3",
        video_path=None,
        cover_path=None,
        config=config,
        output_dir=tmp / "req_out",
    )


def _wait_done(manager: JobManager, job_id: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = manager.snapshot(job_id)
        if snap is not None and snap["done"]:
            return True
        time.sleep(0.02)
    return False


class TestSubmitValidation:
    def test_rejects_bad_extension(self, tmp_path):
        manager = JobManager(web_dir=tmp_path / "jobs")
        with pytest.raises(ValueError, match="Unsupported file type"):
            manager.submit(_make_request(tmp_path), b"x", "evil.exe")

    def test_rejects_oversized_upload(self, tmp_path):
        manager = JobManager(web_dir=tmp_path / "jobs", max_upload_mb=1)
        with pytest.raises(ValueError, match="too large"):
            manager.submit(_make_request(tmp_path), b"x" * (2 * 1024 * 1024), "big.mp3")

    def test_rejects_missing_extension(self, tmp_path):
        manager = JobManager(web_dir=tmp_path / "jobs")
        with pytest.raises(ValueError, match="Unsupported file type"):
            manager.submit(_make_request(tmp_path), b"x", "noext")

    def test_rejects_bad_cover_extension(self, tmp_path):
        manager = JobManager(web_dir=tmp_path / "jobs")
        with pytest.raises(ValueError, match="Unsupported cover file type"):
            manager.submit(_make_request(tmp_path), b"x", "song.mp3",
                           cover_bytes=b"png", cover_name="cover.png")

    def test_rejects_oversized_cover(self, tmp_path):
        manager = JobManager(web_dir=tmp_path / "jobs", max_upload_mb=1)
        with pytest.raises(ValueError, match="Cover too large"):
            manager.submit(_make_request(tmp_path), b"x", "song.mp3",
                           cover_bytes=b"x" * (2 * 1024 * 1024), cover_name="cover.jpg")


class TestSubmitCover:
    def test_submit_stores_cover_in_upload_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"ID3data", "song.mp3",
                                 cover_bytes=b"jpegdata", cover_name="Album Cover.JPG")
            assert job.status == STATUS_QUEUED
            cover = job.dir / "upload" / "cover.jpg"
            assert cover.is_file()
            assert cover.read_bytes() == b"jpegdata"
            assert job.request.cover_path == cover
        finally:
            manager.stop()

    def test_submit_without_cover(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"ID3data", "song.mp3")
            assert job.request.cover_path is None
            assert not (job.dir / "upload" / "cover.jpg").exists()
        finally:
            manager.stop()


class TestJobLifecycle:
    def test_submit_creates_dirs_and_upload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"ID3data", "song.mp3")
            assert job.status == STATUS_QUEUED
            assert job.upload_path.is_file()
            assert job.upload_path.read_bytes() == b"ID3data"
            assert (job.dir / "tmp").is_dir()
            assert (job.dir / "output").is_dir()
            assert (job.dir / "job.json").is_file()
            # Job config points inside the job dir
            assert Path(job.request.config.temp_dir) == job.temp_dir
            assert Path(job.request.config.output_dir) == job.output_dir
            assert job.request.input_path == job.upload_path
        finally:
            manager.stop()

    def test_success_flow(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "cli.pipeline.run_process",
            lambda req: ProcessResult(ok=True, txt_path=req.output_dir / "x.txt"),
        )
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            assert _wait_done(manager, job.id)
            assert job.status == STATUS_SUCCEEDED
            assert job.error is None
            snap = manager.snapshot(job.id)
            assert snap["done"] is True
            assert snap["status"] == STATUS_SUCCEEDED
        finally:
            manager.stop()

    def test_failure_flow(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=False, error="boom"))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            assert _wait_done(manager, job.id)
            assert job.status == STATUS_FAILED
            assert job.error == "boom"
        finally:
            manager.stop()

    def test_unhandled_exception_is_failure(self, tmp_path, monkeypatch):
        def boom(req):
            raise RuntimeError("crash")
        monkeypatch.setattr("cli.pipeline.run_process", boom)
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            assert _wait_done(manager, job.id)
            assert job.status == STATUS_FAILED
            assert "crash" in (job.error or "")
        finally:
            manager.stop()

    def test_fifo_single_flight(self, tmp_path, monkeypatch):
        release_first = threading.Event()
        order: list[str] = []
        order_lock = threading.Lock()

        def fake_run(req):
            with order_lock:
                order.append(req.title)
            if req.title == "first":
                release_first.wait(timeout=10)
            return ProcessResult(ok=True)

        monkeypatch.setattr("cli.pipeline.run_process", fake_run)
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            req1 = dataclasses.replace(_make_request(tmp_path), title="first")
            j1 = manager.submit(req1, b"x", "a.mp3")
            deadline = time.time() + 10
            while time.time() < deadline and j1.status != STATUS_RUNNING:
                time.sleep(0.02)
            assert j1.status == STATUS_RUNNING
            req2 = dataclasses.replace(_make_request(tmp_path), title="second")
            j2 = manager.submit(req2, b"x", "b.mp3")
            # Second job must still be queued while the first runs
            deadline = time.time() + 10
            while time.time() < deadline and j2.status != STATUS_QUEUED:
                time.sleep(0.02)
            assert j2.status == STATUS_QUEUED
            assert manager.queue_position(j2.id) == 0
            release_first.set()
            assert _wait_done(manager, j1.id)
            assert _wait_done(manager, j2.id)
            assert order == ["first", "second"]
        finally:
            manager.stop()


class TestLogCapture:
    def test_log_handler_captures_root_logging(self, tmp_path, monkeypatch):
        import logging

        def fake_run(req):
            logging.getLogger("test.job").info("hello from job")
            return ProcessResult(ok=True)

        monkeypatch.setattr("cli.pipeline.run_process", fake_run)
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            assert _wait_done(manager, job.id)
            assert any("hello from job" in line for line in job.logs)
            # Lines are formatted [timestamp] [name] message
            assert any(line.startswith("[") and "[test.job]" in line for line in job.logs)
        finally:
            manager.stop()

    def test_snapshot_log_tail_bounded(self, tmp_path, monkeypatch):
        import logging

        def fake_run(req):
            for i in range(300):
                logging.getLogger("test.job").info(f"line {i}")
            return ProcessResult(ok=True)

        monkeypatch.setattr("cli.pipeline.run_process", fake_run)
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            assert _wait_done(manager, job.id)
            snap = manager.snapshot(job.id)
            assert len(snap["logs_tail"]) == 200
            assert "line 100" in snap["logs_tail"][0]
            assert "line 299" in snap["logs_tail"][-1]
        finally:
            manager.stop()


class TestSnapshotShape:
    def test_snapshot_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            job = manager.submit(_make_request(tmp_path), b"x", "song.mp3")
            snap = manager.snapshot(job.id)
            for key in ("id", "title", "status", "position", "logs_tail", "error",
                        "done", "files", "run_dir", "zip_name", "html_name",
                        "editor_name", "created_at", "elapsed_s"):
                assert key in snap, key
            assert snap["done"] is False
            assert manager.snapshot("nope") is None
        finally:
            manager.stop()

    def test_split_job_run_dir_from_input_stem(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            req = dataclasses.replace(
                _make_request(tmp_path), title="", artist="", mode="split",
            )
            job = manager.submit(req, b"x", "My Band - Cool Song.mp3")
            assert job.title == "My Band - Cool Song.mp3"
            snap = manager.snapshot(job.id)
            assert snap["run_dir"] == "My Band - Cool Song"
            assert snap["title"] == "My Band - Cool Song.mp3"
            assert snap["zip_name"] is None
            assert snap["html_name"] is None
            assert snap["editor_name"] is None
        finally:
            manager.stop()

    def test_split_job_keeps_title_artist_when_given(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cli.pipeline.run_process", lambda req: ProcessResult(ok=True))
        manager = JobManager(web_dir=tmp_path / "jobs")
        manager.start()
        try:
            req = dataclasses.replace(_make_request(tmp_path), mode="split")
            job = manager.submit(req, b"x", "song.mp3")
            snap = manager.snapshot(job.id)
            assert snap["run_dir"] == "Tester - Test Song"
            assert snap["title"] == "Test Song"
        finally:
            manager.stop()


class TestRetention:
    def test_prune_old_job_dirs(self, tmp_path):
        web_dir = tmp_path / "jobs"
        web_dir.mkdir()
        old = web_dir / "oldjob"
        old.mkdir()
        (old / "job.json").write_text(json.dumps({"created_at": time.time() - 8 * 86400}), encoding="utf-8")
        fresh = web_dir / "freshjob"
        fresh.mkdir()
        (fresh / "job.json").write_text(json.dumps({"created_at": time.time()}), encoding="utf-8")

        manager = JobManager(web_dir=web_dir, retention_days=7)
        removed = manager.prune_old_jobs()
        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_prune_without_manifest_uses_mtime(self, tmp_path):
        web_dir = tmp_path / "jobs"
        web_dir.mkdir()
        old = web_dir / "oldjob"
        old.mkdir()
        import os
        old_ts = time.time() - 30 * 86400
        os.utime(old, (old_ts, old_ts))

        manager = JobManager(web_dir=web_dir, retention_days=7)
        assert manager.prune_old_jobs() == 1
        assert not old.exists()
