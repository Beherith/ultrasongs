All decisions incorporated. Here is the finalized plan.

# Final Plan: Dash Web Frontend for Ultrasongs

## Decisions (from your answers)

- **Settings**: full auto-generated form over all 44 config keys (typed, grouped, collapsible, reset-to-defaults).
- **ZIP intermediates**: included **by default in every ZIP** (CLI and web). New `--no-intermediates` flag on `process` strips them. Plot PNGs are **excluded** (JSON/HTML/TXT only).
- **Concurrency**: single worker, FIFO queue, one job at a time.
- **Password**: environment variable `ULTRASONGS_WEB_PASSWORD` (plaintext in env, constant-time compare). No `--set-password`; `--no-auth` allowed only when host is localhost/127.0.0.1.

## Architecture

```
Browser ──► Flask (Dash server, threaded=True, no reloader)
              │ before_request: session-auth guard (allow /login, /logout)
              │ /login, /logout      — Flask routes, minimal inline HTML form
              │ /download/<job_id>/<filename> — send_file, session + traversal-guarded
              └─ Dash layout at "/"
                   ├─ Song form: dcc.Upload, title, artist, lyrics textarea,
                   │   language dropdown, settings accordion (44 keys)
                   └─ Job card: status pill, live log tail (1s dcc.Interval poll),
                       Download ZIP / preview / per-file links

JobManager: 1 daemon worker thread, FIFO queue
   └─► cli/pipeline.run_process(ProcessRequest) with per-job Config
        (temp_dir/output_dir = <web_dir>/<job_id>/{tmp,output})
        JobLogHandler (logging.Handler on root logger) → job.log deque
```

## New files

| File | Purpose |
|---|---|
| `cli/pipeline.py` | `ProcessRequest`, `ProcessResult`, `run_process()`, `prepare_lyrics_input()`, `sanitize_filename()` |
| `cli/web/__init__.py` | empty |
| `cli/web/app.py` | `create_app()`, `run_server()`, layout, all callbacks |
| `cli/web/auth.py` | env-var password check, `/login` `/logout` routes, `before_request` guard |
| `cli/web/jobs.py` | `Job`, `JobManager`, `JobLogHandler`, retention pruning |
| `cli/web/settings_meta.py` | metadata table driving the settings form |
| `cli/web_config.jsonc` | `host`, `port`, `web_dir`, `job_retention_days`, `max_upload_mb`, `poll_interval_s` |
| `cli/tests/test_pipeline.py`, `test_web_auth.py`, `test_web_jobs.py`, `test_package_intermediates.py`, `test_config_dict.py`, `test_web_app.py` | see §10 |

Modified: `cli/__main__.py`, `cli/config.py`, `cli/package.py`, `cli/pyproject.toml`, `cli/requirements.txt`, `AGENTS.md`, `.gitignore` (`web_jobs/`).

## 1. `cli/pipeline.py` — extract orchestration

```python
@dataclass(frozen=True)
class ProcessRequest:
    title: str
    artist: str
    lyrics_text: str
    input_path: Path            # audio or video
    video_path: Path | None
    config: Config
    output_dir: Path
    stage: str = "all"
    resume_path: Path | None = None
    include_intermediates: bool = True

@dataclass(frozen=True)
class ProcessResult:
    ok: bool
    error: str | None = None
    txt_path: Path | None = None
    zip_path: Path | None = None
    html_path: Path | None = None
    output_dir: Path | None = None
    temp_dir: Path | None = None
```

- `run_process()` = the body of today's `_cmd_process` (cli/__main__.py:130-284) with `args.*` → `req.*`; failures logged **and** returned as `ok=False, error=...`. Records `run_started = time.time()` at entry (used for the staleness rule below).
- `prepare_lyrics_input(text) -> (lyrics_text, UltrastarMeta | None)`: moves `_looks_like_ultrastar` + `extract_lyrics_from_ultrastar`/`parse_ultrastar_txt` block; CLI keeps file-based title/artist/mp3 fallbacks, web uses it to auto-prefill when a pasted Ultrastar txt is detected.
- `sanitize_filename(title)`: strip OS-illegal chars, collapse whitespace, cap ~80 — used for output filenames by **both** CLI and web (small improvement, noted in AGENTS.md).
- `_cmd_process` becomes a thin adapter (read file → resolve fallbacks → `run_process` → exit code). CLI behavior otherwise unchanged.

## 2. `cli/config.py` — `config_from_dict()`

- Refactor `load_config()`: JSONC → dict → new `config_from_dict(data, source_path) -> Config`, which owns the existing `field_map` + all validation blocks (identical fallback-with-warning behavior).
- Web per-job config: merge `{file config values} ← {user form overrides}` as a dict, validate once via `config_from_dict`, then `dataclasses.replace(config, temp_dir=..., output_dir=...)` for job isolation. Since **all** intermediates are written via `config.temp_path`, per-job `temp_dir` gives natural isolation for free.

## 3. `cli/package.py` — intermediates in ZIP by default

- New params: `extra_files: Sequence[Path] = ()`, `intermediates_prefix: str = "intermediates"`, and a module-level helper `collect_intermediates(temp_path: Path, since_ts: float | None = None) -> list[Path]`:
  - includes `*.json`, `*.html`, `*.txt` present in `temp_path` (skips `*.mp3` — already in zip as title/vocals/accompaniment; skips `note_segments_plots/` entirely)
  - with `since_ts`: only files with `mtime >= since_ts - 2` (protects the shared CLI `tmp/` from stale files of other runs)
- `package_output(..., extra_files)` writes them as `intermediates/<name>` inside the existing ZIP; default `()` keeps the function backward compatible for `import`.
- `run_process` passes `collect_intermediates(req temp_path, since_ts=run_started)` when `req.include_intermediates` (default **True**).
- Resulting ZIP:
  ```
  <title>.txt  <title>.mp3  [<title>.mp4]  vocals.mp3  accompaniment.mp3
  intermediates/
    <stem>_transcribe.json
    <stem>_faster_whisper_passes.json        (or _whisperx_ for legacy backend)
    whisperx_pitch.json  whisperx_pitch.html
    align_debug.json  align_backtrace.txt  note_segments.txt   (when written)
  ```
- CLI flag: `process --no-intermediates` → `include_intermediates=False`.

## 4. `cli/web/jobs.py` — job manager

```python
@dataclass
class Job:
    id: str; status: str            # queued | running | succeeded | failed
    title: str; upload_name: str; created_at: float
    logs: deque[str]                # maxlen=4000
    error: str | None
    result: ProcessResult | None
    dir: Path                       # <web_dir>/<job_id>/  (tmp/ + output/ + upload/)

class JobManager:
    def submit(self, req: ProcessRequest, upload_bytes, upload_name) -> Job
    def snapshot(self, job_id) -> dict   # {status, position, logs_tail[200], error,
                                         #  files, zip_name, done}
```

- 1 daemon worker thread (created at app start), FIFO `queue.Queue`; exactly one `run_process` in flight.
- `JobLogHandler(logging.Handler)` attached to the root logger for the duration of the job (safe: single-flight); pushes formatted lines into `job.logs`.
- Upload: written to `<job_dir>/upload/original.<ext>`; whitelist `.mp3 .wav .flac .m4a .ogg .oga .mp4 .mkv .webm .mov .avi`; size cap `max_upload_mb`.
- Retention: prune job dirs + records older than `job_retention_days` at startup and on each submit.

## 5. `cli/web_config.jsonc`

```jsonc
{
  "host": "127.0.0.1",
  "port": 8030,
  "web_dir": "./web_jobs",
  "job_retention_days": 7,
  "max_upload_mb": 2048,
  "poll_interval_s": 1.0
}
```

Loader: re-export `load_jsonc(path)` from `cli/config.py` (reuses `_strip_jsonc`). Password is **not** in this file — it comes from `ULTRASONGS_WEB_PASSWORD`.

## 6. `cli/web/auth.py`

- `install_auth(server, password: str)`:
  - `GET/POST /login` — single inline-HTML form; POST verifies with `hmac.compare_digest`, sets `session["us_auth"]=True`, `session.permanent=True`, redirects to `/`.
  - `GET /logout` — clears session, redirects to `/login`.
  - `before_request` — everything else requires the session; secret key = `secrets.token_hex(32)` at startup.
- Startup: no env var and no `--no-auth` → log error, exit 1. `--no-auth` refused unless host ∈ {127.0.0.1, localhost}.
- Documented: no TLS — bind to localhost or use a reverse proxy for LAN access.

## 7. `cli/web/settings_meta.py` + `app.py` — the form

- `SETTINGS: list[SettingMeta]` covering **all 44 keys**: `(key, kind, choices|range, group)` with kinds `select / number / float / checkbox / text`; groups: **Models & backends, ASR & timing, Pitch & activity, Note segmentation, Pauses, Output & debug**. Enums (device, backends, compute types, models, interpolate methods) hard-coded to the validation sets in `config.py`; numbers get min/step from the validation rules (e.g. `whisperx_align_runs` step 2, odd).
- `create_app(web_cfg, pipeline_config, manager)`:
  - **Form card**: `dcc.Upload` (single, drag-drop, extension `accept`); `dcc.Input` Title/Artist (required); `dcc.Textarea` Lyrics (rows 14); `dcc.Dropdown` Language ("Auto-detect" → `""` + ISO-639-1 list → `whisper_language`); collapsible settings section auto-generated from `SETTINGS`, default values from the active `Config`, "Reset to file defaults" button; **Process** button (disabled while a job runs; label shows queue position when queued).
  - **Job card**: status pill, elapsed, scrollable log tail (last 200 lines), on success: **Download ZIP** (`/download/<job_id>/<title>.zip`), "Open preview" (`<title>.html` inline), per-file download list; on failure: error banner + log.
  - `dcc.Interval` (1s cadence from web config) polling callback is a no-op without an active job id in a `dcc.Store`.
  - Callbacks: `process_click` (validate → save upload → build overrides dict → `config_from_dict` → `dataclasses.replace` for job dirs → `manager.submit`), `poll`, `upload_picked` (name/size + early extension check), `lyrics_pasted` (prefill Title/Artist from Ultrastar headers).
  - `run_server`: `server.run(host, port, threaded=True, debug=False, use_reloader=False)` — no reloader (would duplicate the worker), threaded so polls don't block.
- No `dash-bootstrap-components`; plain `dash.html`/`dash.dcc` + minimal inline CSS.

## 8. CLI wiring & deps

- `cli/__main__.py`: new `web` subcommand (`--host --port --web-config --no-auth`) → lazy `cli.web.app.run_server` with a clear "install `ultrasongs-cli[web]`" error if Dash is missing.
- `cli/pyproject.toml`: `[project.optional-dependencies] web = ["dash>=2.17"]` (dash pulls flask + plotly; no torch/numpy conflicts).
- `cli/requirements.txt`: add `dash` (commented as web-only).
- `.gitignore`: `web_jobs/`. `AGENTS.md`: new "Web service" section (commands, env var, job layout, zip layout, one-at-a-time queue, no-TLS note) + updated zip/`--no-intermediates`/sanitized-title notes.

## 9. Risks / edge cases

- **Stale shared tmp (CLI)**: `since_ts` mtime filter (§3) prevents other runs' files entering the zip.
- **Server restart mid-job**: in-flight/queued jobs lost (documented); completed job dirs survive until retention prune.
- **Long jobs**: polling pattern tolerates 30+ min; log deque is bounded memory.
- **Path traversal**: download route resolves and checks `is_relative_to(job.dir)`.
- **`whisper_language=""`** (auto) survives dict round-trips.
- **Windows**: worker thread never calls `setup_logging`; FFmpeg DLL registration in `whisperx_transcribe.py` already handles the env.

## 10. Tests

| File | Coverage |
|---|---|
| `test_config_dict.py` | `config_from_dict` valid/invalid-value fallbacks (mirrors `test_config.py`); `load_config` unchanged |
| `test_pipeline.py` | `prepare_lyrics_input` (plain/Ultrastar/empty), `sanitize_filename`, `run_process` with monkeypatched stages: call sequence, resume path, video-from-input, error → `ok=False` |
| `test_package_intermediates.py` | `collect_intermediates` (includes json/html/txt, skips mp3 + plots, `since_ts` filter); `package_output` zip layout with/without extras; `import` unchanged |
| `test_web_jobs.py` | submit→running→succeeded/failed, FIFO, log-handler capture, retention pruning, `snapshot()` shape |
| `test_web_auth.py` | Flask `test_client`: redirect when anon, wrong/right password, logout, `--no-auth` host guard |
| `test_web_app.py` | layout builds 44 setting controls from `SETTINGS`; process-click validation rejects missing title/upload without submitting |

Manual smoke test: `python -m cli web --no-auth` + committed `test_song_full_audio.mp3` / `test_song_lyrics_only.txt`; verify zip contains `intermediates/*.json`.

## Build order

1. `config_from_dict` refactor + tests
2. `cli/pipeline.py` extraction; CLI regression (`pytest cli/tests/` + one `--stage extract` run)
3. `package.py` intermediates + `--no-intermediates` + tests
4. `web_config.jsonc` + `jobs.py` + tests
5. `auth.py` + tests
6. `settings_meta.py` + `app.py` + light tests
7. `web` subcommand, deps, docs, smoke test

The plan is complete — say the word and I'll start implementing (read-only until you approve).