"""Web job queue: single worker thread, FIFO, bounded per-job log capture."""

import dataclasses
import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from cli.logging_setup import get_logger
from cli.pipeline import (
    ProcessRequest,
    ProcessResult,
    resolve_output_name,
    sanitize_filename,
)

logger = get_logger("cli.web.jobs")

ALLOWED_UPLOAD_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".oga",
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
}

COVER_UPLOAD_EXTENSIONS = {".jpg", ".jpeg"}

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = {STATUS_SUCCEEDED, STATUS_FAILED}

MANIFEST_NAME = "job.json"
LOGS_MAXLEN = 4000
LOGS_TAIL = 200


class JobLogHandler(logging.Handler):
    """logging.Handler that appends formatted lines into a job's log deque.

    Safe to attach to the root logger: the queue guarantees at most one job
    runs at a time.
    """

    def __init__(self, job: "Job"):
        super().__init__(level=logging.DEBUG)
        self._job = job
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._job.logs.append(self.format(record))
        except Exception:
            pass


@dataclass
class Job:
    id: str
    status: str
    title: str
    upload_name: str
    created_at: float
    dir: Path
    logs: deque = field(default_factory=lambda: deque(maxlen=LOGS_MAXLEN))
    error: str | None = None
    result: ProcessResult | None = None
    request: ProcessRequest | None = field(default=None, repr=False)

    @property
    def upload_path(self) -> Path:
        name = Path(self.upload_name)
        return self.dir / "upload" / f"{sanitize_filename(name.stem)}{name.suffix.lower()}"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def temp_dir(self) -> Path:
        return self.dir / "tmp"

    def write_manifest(self) -> None:
        manifest = {
            "id": self.id,
            "title": self.title,
            "upload_name": self.upload_name,
            "created_at": self.created_at,
            "status": self.status,
        }
        (self.dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class JobManager:
    """FIFO queue with a single daemon worker running cli.pipeline.run_process."""

    def __init__(
        self,
        web_dir: Path | str,
        retention_days: float = 7.0,
        max_upload_mb: float = 2048.0,
    ):
        self.web_dir = Path(web_dir)
        self.retention_days = float(retention_days)
        self.max_upload_bytes = int(max_upload_mb * 1024 * 1024)
        self._queue: "queue.Queue[tuple[Job, ProcessRequest]]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._worker is not None:
            return
        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.prune_old_jobs()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="ultrasongs-web-worker"
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._worker is None:
            return
        self._worker.join(timeout=timeout)
        self._worker = None

    # ── submission ─────────────────────────────────────────────────────────

    def submit(self, req: ProcessRequest, upload_bytes: bytes, upload_name: str,
               cover_bytes: bytes | None = None, cover_name: str | None = None) -> Job:
        """Create the job directory, store the upload, and enqueue the job.

        ``cover_bytes``/``cover_name`` optionally carry a JPEG album cover,
        stored in the job's upload dir and wired into the request as
        ``cover_path``.
        """
        suffix = Path(upload_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported file type: '{suffix or upload_name}'")
        if len(upload_bytes) > self.max_upload_bytes:
            raise ValueError(
                f"Upload too large: {len(upload_bytes) / (1024 * 1024):.0f} MB "
                f"(limit {self.max_upload_bytes / (1024 * 1024):.0f} MB)"
            )
        cover_path: Path | None = None
        if cover_bytes is not None:
            cover_suffix = Path(cover_name or "").suffix.lower()
            if cover_suffix not in COVER_UPLOAD_EXTENSIONS:
                raise ValueError(
                    f"Unsupported cover file type: '{cover_suffix or cover_name}' "
                    f"(use a .jpg or .jpeg)"
                )
            if len(cover_bytes) > self.max_upload_bytes:
                raise ValueError(
                    f"Cover too large: {len(cover_bytes) / (1024 * 1024):.0f} MB "
                    f"(limit {self.max_upload_bytes / (1024 * 1024):.0f} MB)"
                )

        self.prune_old_jobs()

        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            status=STATUS_QUEUED,
            title=req.title or upload_name,
            upload_name=upload_name,
            created_at=time.time(),
            dir=self.web_dir / job_id,
        )
        for sub in ("upload", "tmp", "output"):
            (job.dir / sub).mkdir(parents=True, exist_ok=True)
        job.upload_path.write_bytes(upload_bytes)

        if cover_bytes is not None:
            cover_path = job.dir / "upload" / f"cover{cover_suffix}"
            cover_path.write_bytes(cover_bytes)

        job.request = dataclasses.replace(
            req,
            config=dataclasses.replace(
                req.config,
                temp_dir=str(job.temp_dir),
                output_dir=str(job.output_dir),
            ),
            output_dir=job.output_dir,
            input_path=job.upload_path,
            cover_path=cover_path,
        )
        job.write_manifest()

        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, job.request))
        logger.info(f"Job {job.id} queued: {req.title!r} ({upload_name})")
        return job

    # ── worker ─────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while True:
            job, req = self._queue.get()
            try:
                self._run_job(job, req)
            except Exception as exc:
                logger.exception(f"Job {job.id} crashed")
                job.status = STATUS_FAILED
                job.error = str(exc)
            finally:
                self._queue.task_done()

    def _run_job(self, job: Job, req: ProcessRequest) -> None:
        from cli.pipeline import run_process

        job.status = STATUS_RUNNING
        job.write_manifest()
        handler = JobLogHandler(job)
        root = logging.getLogger()
        prev_level = root.level
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        try:
            job.result = run_process(req)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)

        if job.result is not None and job.result.ok:
            job.status = STATUS_SUCCEEDED
        else:
            job.status = STATUS_FAILED
            job.error = job.result.error if job.result is not None else "Unknown error"
        job.write_manifest()
        logger.info(f"Job {job.id} finished: {job.status}")

    # ── queries ────────────────────────────────────────────────────────────

    def job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job_id: str) -> int | None:
        """0-based position of a queued job (0 = next to run), else None."""
        items: list[tuple[Job, ProcessRequest]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for item in items:
            self._queue.put(item)
        for i, (j, _) in enumerate(items):
            if j.id == job_id:
                return i
        return None

    def snapshot(self, job_id: str) -> dict | None:
        job = self.job(job_id)
        if job is None:
            return None
        # The pipeline writes all outputs into a per-song subfolder.
        req = job.request
        safe = resolve_output_name(
            req.artist if req else "",
            req.title if req else "",
            req.input_path if req else None,
        )
        run_dir = job.output_dir / safe
        files: list[dict] = []
        if run_dir.is_dir():
            for f in sorted(run_dir.iterdir()):
                if f.is_file():
                    files.append({"name": f.name, "size": f.stat().st_size})
        return {
            "id": job.id,
            "title": job.title,
            "status": job.status,
            "position": self.queue_position(job.id),
            "logs_tail": list(job.logs)[-LOGS_TAIL:],
            "error": job.error,
            "done": job.status in TERMINAL_STATUSES,
            "files": files,
            "run_dir": safe,
            "zip_name": f"{safe}.zip" if (run_dir / f"{safe}.zip").is_file() else None,
            "html_name": f"{safe}.html" if (run_dir / f"{safe}.html").is_file() else None,
            "editor_name": (f"{safe}_editor.html"
                            if (run_dir / f"{safe}_editor.html").is_file() else None),
            "created_at": job.created_at,
            "elapsed_s": time.time() - job.created_at,
        }

    # ── retention ──────────────────────────────────────────────────────────

    def prune_old_jobs(self) -> int:
        """Delete job directories (and records) older than retention_days."""
        if not self.web_dir.is_dir():
            return 0
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        for child in list(self.web_dir.iterdir()):
            if not child.is_dir():
                continue
            created = None
            manifest = child / MANIFEST_NAME
            if manifest.is_file():
                try:
                    created = json.loads(manifest.read_text(encoding="utf-8")).get("created_at")
                except (json.JSONDecodeError, OSError):
                    created = None
            if created is None:
                try:
                    created = child.stat().st_mtime
                except OSError:
                    continue
            if created < cutoff:
                try:
                    shutil.rmtree(child)
                    removed += 1
                    with self._lock:
                        self._jobs.pop(child.name, None)
                except OSError as exc:
                    logger.warning(f"Could not prune job dir {child}: {exc}")
        return removed
