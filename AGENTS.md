# AGENTS.md

## Repository mission

UltraSongs is migrating from a Next.js/TypeScript application plus a Python service to
a clean Python application with Dash as the frontend. The active implementation lives
under `src/ultrasongs` and covers the automatic song-generation, validation, repair,
artifact, and reporting workflows.

Read `python_migration_plan.md` before making migration-level changes.

## Scope boundaries

- Treat `src/ultrasongs` as the active implementation.
- Keep the existing TypeScript application under `app/` and legacy service under
  `python/` intact as migration references. Do not delete, move, broadly reformat, or
  "clean up" them unless the user explicitly authorizes the final cutover.
- The timeline/manual note editor is out of scope for this port. Do not port or recreate
  note dragging, piano-roll editing, microphone capture, note previews, or other editor
  behavior unless a later task explicitly starts an editor project.
- Reference TXT upload is for immutable validation input. Never silently make it an
  editable chart or overwrite its bytes.
- Preserve unrelated files and dirty-worktree changes. Local song fixtures and generated
  media may be user-owned even when untracked.

## Source-of-truth modules

- `src/ultrasongs/config.py`: all application configuration and validation.
- `src/ultrasongs/services.py`: shared default repository/pipeline construction.
- `src/ultrasongs/processing/pipeline.py`: the only full automatic pipeline orchestrator.
- `src/ultrasongs/domain/ultrastar/`: canonical UltraStar model, parser, writer, timing,
  generation, and archive behavior.
- `src/ultrasongs/domain/scoring/similarity.py`: canonical chart comparison.
- `src/ultrasongs/domain/reporting/pipeline_report.py`: canonical HTML visualization.
- `src/ultrasongs/domain/validation.py`: strict reference inspection and thresholds.
- `src/ultrasongs/storage/`: project/run manifests and immutable artifacts.
- `src/ultrasongs/cli/repair.py`: application adapter for repairing MP3 + TXT pairs.
- `src/ultrasongs/web/`: Dash layout, callbacks, local background submission, and
  ownership-checked downloads.

Root scripts `score_songs.py`, `ultrastar_to_html.py`, and `pitch_to_html.py` are
compatibility wrappers. Keep business logic in the package, not in these scripts.

## Architecture rules

### Configuration

- Define every configurable value in `config.py` as part of `AppSettings`.
- Load settings once at the application boundary and inject them into services.
- Do not read environment variables from domain, processing, storage, or web modules.
- Configuration precedence is defaults, file, environment, then validated per-run
  overrides.
- Expose a field to Dash/`--set` only with `_ui_field` and only when it is safe to change
  for one run. Paths, executables, devices, server settings, and security limits are
  startup-only.
- Persist an immutable effective-configuration snapshot for every pipeline run.

### Domain and processing

- Keep domain modules deterministic and free of Dash, filesystem storage, and ML model
  initialization.
- Keep optional/heavy dependencies lazily imported inside processing adapters. Importing
  `ultrasongs`, parsing a chart, or running unit tests must not load/download ML models.
- Release model/GPU resources at stage boundaries when an adapter exposes a close or
  cleanup operation.
- Inject FFmpeg, Demucs, transcription, pitch, tempo, alignment, scoring, and reporting
  boundaries into `PipelineRunner` so tests can replace them.
- Do not create a second repair/generation pipeline. Dash and CLI workflows must call the
  same `PipelineRunner`.
- Preserve the current stage order unless a change includes manifest, UI progress, and
  test updates:

  `intake -> normalize_audio -> load_audio -> separate -> pitch -> pauses -> transcribe -> tempo -> align -> generate -> package -> score -> report`

- `score` is conditional on a reference. Never treat zero matched notes as a perfect
  zero-error comparison; unavailable error metrics must remain `None`.
- Validation `minimum_match_ratio` means coverage of reference notes. Use
  `reference_coverage`, not coverage of the smaller chart.

### UltraStar and reports

- Parse charts once through `domain.ultrastar`; do not duplicate header, note, encoding,
  BPM/GAP, or beat/time rules.
- Preserve normal, golden, and freestyle note types plus line breaks and unknown metadata
  headers through supported round trips.
- Lyric reconstruction is best effort. Any expensive repair workflow must allow the user
  to review or replace reconstructed lyrics.
- Use the unified pipeline report for new visualization work. Do not grow separate HTML
  implementations in the compatibility scripts.
- Reports should remain self-contained and include intermediate data, candidate/reference
  charts, scoring, validation reasons, and effective configuration when available.

### Storage and safety

- Store the exact uploaded reference bytes as `reference_ultrastar` before scoring.
- Artifacts are immutable, content-hashed, and owned by an opaque project/run/artifact ID.
- Bump persisted schema versions and add forward migrations when project or manifest
  formats change; never silently reinterpret an existing record.
- Resolve downloads and exports through `ArtifactRepository`; verify ownership and hashes.
- Never expose or accept arbitrary repository-relative paths from a browser request.
- Keep temporary model/media work under configured `temp_dir` and clean it after success
  and failure.
- Human-facing repair exports must come from registered artifacts, not temporary files.
- Keep physical artifact paths human-readable as `<run>/<kind>/<meaningful filename>`.
  Opaque IDs belong in manifests and routes, not in `art_<id>/content.ext` disk paths.
- Avoid overwriting existing outputs. Repair bundles use a unique run ID.
- Do not commit MP3s, model weights, project stores, exports, reports, caches, or generated
  fixtures unless a small fixture is deliberately approved for tests.

## Development setup

Python 3.11+ and FFmpeg are required.

```bash
python -m venv .venv
python -m pip install -e ".[runtime,gpu,test,dev]"
python -m ultrasongs --print-config
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Install a PyTorch
build compatible with the local CPU/CUDA environment when the default resolver is not
appropriate.

Start Dash with:

```bash
python -m ultrasongs
```

Exercise the user-facing repair workflow with:

```bash
python -m ultrasongs repair --audio song.mp3 --song song.txt --output-dir exports
```

## Test expectations

Run tests proportional to the change. The normal minimum before handoff is:

```bash
python -m pytest tests/unit -q
python -m ruff check src tests score_songs.py ultrastar_to_html.py pitch_to_html.py
```

Also run these when the touched behavior warrants them:

```bash
python -m pytest tests/parity -m parity -q
python -m pytest tests/end_to_end/test_repair_cli.py -m e2e --e2e-audio song.mp3 --e2e-song song.txt
```

- Unit tests must be deterministic, offline, and model-download-free.
- Use injected fake processing services for a complete fast pipeline test.
- Parity tests may use documented local fixtures and must skip clearly when unavailable.
- The real E2E test is opt-in because it is slow and requires media, FFmpeg, ML packages,
  models, and possibly a GPU.
- For changes to repair, verify exact old TXT preservation, parseable new TXT, ZIP members,
  score JSON, validation outcome, HTML old/new sections, and successful manifest stages.
- For configuration changes, test defaults, file/environment precedence, UI-safe override
  acceptance, startup-only override rejection, and snapshot round trips.
- For storage changes, test containment, ownership, immutability, hashes, and cleanup.

Do not weaken production thresholds merely to make a parity fixture pass. If an observed
migration baseline changes, document why and keep acceptance policy independently
configurable.

## Code style

- Target Python 3.11 and keep Ruff clean with the rules in `pyproject.toml`.
- Use type annotations for public APIs and dependency-injection protocols.
- Prefer frozen dataclasses/Pydantic models and explicit return objects over unstructured
  dictionaries at module boundaries.
- Keep web callbacks small and grouped by feature. Business logic does not belong in Dash
  callbacks.
- Use `pathlib.Path`, explicit UTF-8/UTF-8-SIG handling, and safe filenames.
- Raise errors with the pipeline stage or input context needed to diagnose them.
- Update README usage and `python_migration_plan.md` when user-visible commands, outputs,
  configuration, or migration status changes.

## Completion checklist

Before declaring a task complete:

1. Confirm the change stays inside the non-editor Python migration scope.
2. Confirm configuration remains centralized and snapshotted.
3. Confirm no reference or user artifact can be overwritten or escaped by path input.
4. Confirm Dash and CLI still share the canonical pipeline/domain implementations.
5. Run relevant unit, parity, lint, startup, and opt-in E2E checks.
6. Report checks that were skipped, especially real-model tests, and why.
7. Leave the TypeScript reference runnable until the migration plan's cutover gate passes.
