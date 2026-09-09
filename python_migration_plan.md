# Python and Dash Migration Plan

## Goal

Replace the active Next.js/TypeScript application with a clean, modular Python application using Dash for the frontend. Keep the existing TypeScript implementation intact and runnable as a reference until the Python pipeline reaches end-to-end parity.

The initial Python port covers the automatic pipeline only:

1. Upload audio or video and enter song metadata/lyrics.
2. Normalize audio with FFmpeg.
3. Separate vocals and accompaniment.
4. Detect pitch and pauses.
5. Transcribe vocals.
6. Align the supplied lyrics to the transcription.
7. Generate an Ultrastar TXT file and downloadable ZIP.
8. Save and load processed projects.
9. Validate a generated song against an existing MP3 + Ultrastar TXT pair.
10. Produce a unified visual HTML report for intermediate and final results.

## Current implementation status (2026-08-09)

- [x] Phase 1 foundation: installable `src/ultrasongs` package, typed centralized
  configuration, effective snapshots, CLI entry point, and schema-driven UI overrides.
- [x] Phase 2 domain layer: canonical Ultrastar models, parser/writer, beat mapping,
  reference lyric reconstruction, and immutable reference artifacts.
- [x] Phase 3 deterministic pipeline: lyric alignment, syllabification, pitch/pause
  aggregation, note generation, and archive creation.
- [x] Phase 4 service boundaries: injected FFmpeg, Demucs, pitch, transcription,
  WhisperX, and tempo adapters with lazy optional dependencies.
- [x] Phase 5 validation: canonical similarity implementation, validation thresholds,
  and `score_songs.py` compatibility wrapper.
- [x] Phase 6 report: one self-contained HTML report for transcription, pitch, pauses,
  candidate/reference charts, configuration, similarity, and pass/fail reasons.
- [x] Phase 7 workflow skeleton: MP3/video + optional reference TXT submission,
  background execution, project/run manifests, and downloadable artifacts.
- [x] Phase 8 Dash shell: Generate/Validate modes, metadata and reconstructed-lyrics
  review, Advanced Settings, progress polling, and ownership-checked downloads.
- [x] Automated baseline: 105 unit tests, one frozen Diggy downstream parity test,
  Ruff, application-factory smoke test, and `score_songs.py` CLI smoke test.
- [x] Add a reusable `ultrasongs repair` command and opt-in real-model E2E test for
  an existing MP3 + UltraStar TXT pair. Export the original chart, reconstructed or
  corrected lyrics, updated chart, ZIP, score JSON, and unified visual comparison.
- [ ] Run the full pipeline with the real locally installed GPU/ML stack using an
  uploaded MP3 + known Ultrastar TXT, inspect the unified report, and tune parity.
- [ ] Complete cutover documentation and archive (do not delete) the TypeScript app
  only after the real-model end-to-end gate passes.

## Explicitly out of scope

- [ ] Do not port `TimelineEditor.tsx` in this migration.
- [ ] Do not implement a piano-roll or manual note editor.
- [ ] Do not implement note dragging, adding, deleting, splitting, or manual pitch editing.
- [ ] Do not implement microphone capture, live pitch tracing, or Web Audio note previews.
- [ ] Do not port the editor-specific "generate from edited notes" path.
- [ ] Defer general Ultrastar import-for-editing until an editor project is started.
- [ ] Do not optimize or redesign the alignment algorithm until parity tests are passing.

Uploading an existing Ultrastar TXT as a validation reference is in scope. It must not silently turn into an editable project or overwrite the uploaded reference.

## Target architecture

```text
Browser
  -> Dash layouts and callbacks
  -> application services / pipeline orchestrator
  -> domain modules
       alignment
       Ultrastar parsing and generation
       scoring
       reporting
  -> processing modules
       FFmpeg
       Demucs
       torchcrepe
       faster-whisper / WhisperX
       tempo and pause detection
  -> project/artifact repositories
  -> tmp/, drafts/, reports/, exports/
```

Recommended repository layout:

```text
pyproject.toml
src/ultrasongs/
  __init__.py
  __main__.py
  app.py
  config.py
  domain/
    models.py
    alignment/
      smith_waterman.py
      interpolation.py
      syllables.py
    ultrastar/
      models.py
      parser.py
      writer.py
      beat_mapping.py
    scoring/
      similarity.py
    reporting/
      pipeline_report.py
      templates/
  processing/
    pipeline.py
    media.py
    separation.py
    transcription.py
    pitch.py
    pauses.py
    tempo.py
    progress.py
  storage/
    projects.py
    drafts.py
    artifacts.py
  web/
    layout.py
    routes.py
    callbacks/
      upload.py
      processing.py
      validation.py
      drafts.py
      export.py
      settings.py
assets/
  app.css
tests/
  unit/
  integration/
  parity/
  end_to_end/
  fixtures/
legacy/typescript/
```

## Configuration policy

All configuration must be defined and validated in one place: `src/ultrasongs/config.py`.

Use a typed `AppSettings` model, with nested groups such as:

- `ServerSettings`
- `PathSettings`
- `FfmpegSettings`
- `TranscriptionSettings`
- `WhisperXSettings`
- `SeparationSettings`
- `PitchSettings`
- `PauseSettings`
- `AlignmentSettings`
- `TempoSettings`
- `ExportSettings`
- `ValidationSettings`
- `ReportSettings`

Configuration precedence must be explicit:

```text
built-in defaults
  < configuration file / environment variables
  < whitelisted per-project UI overrides
```

Startup-only settings must not be editable from the UI. These include filesystem roots, server host/port, worker backend, security limits, and executable paths. Safe pipeline settings may be exposed in an Advanced Settings panel, including model selection, alignment engine, language override, relevant confidence thresholds, pause parameters, fallback BPM, report options, and validation thresholds.

Every pipeline run must persist an immutable effective-configuration snapshot alongside its artifacts. This makes a result reproducible even after defaults change.

---

## Phase 0 - Baseline and migration safety

### Checklist

- [ ] Record the current working-tree state and avoid modifying unrelated user files.
- [ ] Confirm the current Next.js application and Python transcription service can still start.
- [ ] Preserve the current TypeScript application in place during development.
- [ ] Select at least one short, fast test fixture and the existing Diggy Diggy Hole fixture for full validation.
- [ ] Capture the current pipeline artifacts for each fixture:
  - [ ] normalized MP3 path and metadata
  - [ ] vocals and accompaniment files
  - [ ] transcription JSON with pitch frames and pauses
  - [ ] alignment notes/debug JSON
  - [ ] generated Ultrastar TXT
  - [ ] generated ZIP contents
  - [ ] current HTML visualizations
- [ ] Record current `score_songs.py` results for the generated TXT versus the reference TXT.
- [ ] Document known inconsistencies between the README and the implementation.
- [ ] Treat the current `align.ts` Smith-Waterman implementation, rather than the older README description, as the initial parity target.
- [ ] Create a feature-level acceptance checklist for the non-editor workflow.

### Exit gate

- [ ] Existing reference artifacts and baseline similarity scores are stored under `tests/fixtures/` or a documented external fixture location.
- [ ] The legacy application remains runnable and has not been reorganized yet.

---

## Phase 1 - Python package and centralized configuration

### Checklist

- [ ] Add `pyproject.toml` with runtime, development, GPU, and test dependency groups.
- [ ] Create the `src/ultrasongs` package and `python -m ultrasongs` entry point.
- [ ] Implement the application factory instead of placing all setup in a global `app.py` script.
- [ ] Implement the typed `AppSettings` hierarchy in `config.py`.
- [ ] Move all hard-coded paths, model names, thresholds, sample rates, device choices, and fallbacks into configuration.
- [ ] Load configuration once at startup and inject it into services.
- [ ] Prevent domain modules from reading environment variables directly.
- [ ] Categorize every option as one of:
  - [ ] startup-only
  - [ ] safe UI override
  - [ ] internal constant that is intentionally not configurable
- [ ] Add validation for invalid model names, engines, thresholds, sample rates, directories, and device combinations.
- [ ] Implement serialization of the effective settings snapshot.
- [ ] Add an Advanced Settings Dash panel generated from the whitelisted settings schema.
- [ ] Add a Restore Defaults action for UI overrides.
- [ ] Ensure UI overrides apply to the current project/run only unless explicitly saved as application defaults outside the UI.
- [ ] Add logging configuration with project/job identifiers in every processing message.

### Tests

- [ ] Test default settings.
- [ ] Test configuration-file/environment overrides.
- [ ] Test UI override precedence.
- [ ] Test rejection of startup-only UI overrides.
- [ ] Test effective-settings snapshot round trips.

### Exit gate

- [ ] No migrated module reads configuration independently from environment variables or scattered constants.
- [ ] A test run can print and persist one complete effective configuration document.

---

## Phase 2 - Canonical domain models and Ultrastar parser

### Checklist

- [ ] Define typed models for:
  - [ ] `Project`
  - [ ] `ArtifactManifest`
  - [ ] `WordTimestamp`
  - [ ] `PitchFrame`
  - [ ] `Pause`
  - [ ] `AlignedSyllable`
  - [ ] `UltrastarMetadata`
  - [ ] `UltrastarNote`
  - [ ] `UltrastarSong`
  - [ ] `SimilarityResult`
  - [ ] `PipelineStageResult`
- [ ] Create one canonical Ultrastar TXT parser in `domain/ultrastar/parser.py`.
- [ ] Support normal notes, golden notes, line breaks, metadata, comma/period BPM formats, UTF-8 lyrics, and files with CRLF or LF endings.
- [ ] Centralize beat-to-time and time-to-beat conversion.
- [ ] Remove duplicate parsing rules from future scoring and reporting modules.
- [ ] Preserve the original uploaded TXT bytes/path as an immutable reference artifact.
- [ ] Implement reference-lyric reconstruction from Ultrastar notes and line breaks.
- [ ] Expose reconstructed lyrics in the validation UI for review before running the pipeline.
- [ ] Keep the reference metadata and reference lyrics separate from generated candidate data.

### Tests

- [ ] Port representative parsing cases from existing song files.
- [ ] Test parse -> write -> parse semantic round trips.
- [ ] Test timing conversions against known BPM/GAP examples.
- [ ] Test lyric reconstruction across syllables, spaces, golden notes, and line breaks.
- [ ] Test malformed headers and note rows with useful error messages.

### Exit gate

- [ ] Generation, scoring, reporting, and validation can all depend on the same `UltrastarSong` model and parser.

---

## Phase 3 - Port deterministic pipeline logic

### Checklist

- [ ] Port `app/lib/align.ts` into focused alignment modules.
- [ ] Preserve current character normalization and phonetic scoring.
- [ ] Preserve current Smith-Waterman scoring, gap behavior, backtracking, and interpolation.
- [ ] Preserve pitch-frame selection thresholds and median MIDI behavior.
- [ ] Port syllabification with language mapping and test its output against the current implementation.
- [ ] Port BPM detection and its configured fallback.
- [ ] Port GAP calculation.
- [ ] Port Ultrastar beat mapping, overlap prevention, line breaks, headers, and TXT writing.
- [ ] Port ZIP packaging using Python's standard `zipfile` module.
- [ ] Produce structured debug artifacts without module-level mutable log handles.
- [ ] Keep algorithm improvements in separate follow-up work after parity.

### Tests

- [ ] Compare Python alignment output to frozen TypeScript alignment fixtures.
- [ ] Define explicit numeric tolerances for timestamps, durations, BPM, GAP, and MIDI values.
- [ ] Compare generated TXT files semantically through the canonical parser.
- [ ] Verify ZIP filenames and members.
- [ ] Test empty, partially matched, repeated, accented, and multilingual lyrics.

### Exit gate

- [ ] Frozen transcription JSON can be passed into the Python alignment/export path and produce a semantically equivalent Ultrastar song without running ML models.

---

## Phase 4 - Modularize the ML and media pipeline

### Checklist

- [ ] Replace Node FFmpeg wrappers with one Python `MediaService` that invokes a configured FFmpeg executable without shell interpolation.
- [ ] Validate supported extensions, maximum upload sizes, and generated paths centrally.
- [ ] Split `python/transcribe_service.py` into:
  - [ ] Demucs separation service
  - [ ] torchcrepe pitch service
  - [ ] pause detection service
  - [ ] faster-whisper transcription service
  - [ ] optional WhisperX service/adapter
- [ ] Remove global model initialization from import time.
- [ ] Add explicit model lifecycle management and sequential GPU cleanup.
- [ ] Implement a `PipelineRunner` with named stages and structured results.
- [ ] Emit progress events suitable for Dash background callbacks.
- [ ] Support cancellation and cleanup of incomplete jobs.
- [ ] Write every stage result into the project artifact manifest.
- [ ] Call processing modules directly; remove the Next.js -> FastAPI proxy hop from the new app.
- [ ] Preserve a single-worker local execution mode to avoid GPU contention.

### Tests

- [ ] Unit-test orchestration with mocked processing services.
- [ ] Run FFmpeg integration tests on a short fixture.
- [ ] Run CPU-safe tests for pause and pitch aggregation logic.
- [ ] Add one opt-in real-model smoke test.
- [ ] Verify failed stages retain useful logs and do not present partial output as complete.

### Exit gate

- [ ] A Python command can run upload normalization through final TXT/ZIP generation without Next.js or FastAPI.

---

## Phase 5 - Similarity scoring with `score_songs.py`

### Checklist

- [ ] Move the reusable logic from `score_songs.py` into `domain/scoring/similarity.py`.
- [ ] Keep `score_songs.py` as a backward-compatible thin CLI wrapper.
- [ ] Make the scoring API return a structured `SimilarityResult` instead of only printing text.
- [ ] Preserve and document the existing metrics:
  - [ ] matched-note count and ratio
  - [ ] timing RMSE in milliseconds
  - [ ] duration RMSE in milliseconds
  - [ ] octave-corrected pitch distance in semitones
  - [ ] median and maximum errors
- [ ] Replace the duplicated parser inside `score_songs.py` with the canonical Ultrastar parser.
- [ ] Review the current beat-versus-millisecond heuristic and replace it with explicit parser behavior where possible.
- [ ] Keep compatibility tests for existing CLI output and exit behavior.
- [ ] Add optional JSON output for automation and report generation.
- [ ] Define configurable validation thresholds only after baseline fixtures have been measured.
- [ ] Make validation pass/fail rules visible in both logs and the UI.
- [ ] Never report a good score when no notes were matched.

### Tests

- [ ] Test identical songs.
- [ ] Test global timing offsets.
- [ ] Test duration changes.
- [ ] Test octave-equivalent and chromatically different pitches.
- [ ] Test repeated lyrics and unmatched notes.
- [ ] Test zero-match behavior.
- [ ] Test both direct Python API use and `python score_songs.py song1.txt song2.txt`.

### Exit gate

- [ ] The new app can score its generated TXT against an uploaded reference TXT using the same maintained scoring implementation as the CLI.

---

## Phase 6 - Unified pipeline HTML report

The new report replaces the duplicated rendering logic in `ultrastar_to_html.py` and `pitch_to_html.py`. The original scripts remain untouched until cutover.

### Checklist

- [ ] Create `domain/reporting/pipeline_report.py`.
- [ ] Refactor shared note names, time axes, pitch axes, colors, SVG construction, metadata rendering, and HTML styling into reusable helpers/templates.
- [ ] Accept a project artifact manifest rather than unrelated ad hoc input formats.
- [ ] Include a summary page with:
  - [ ] title and artist
  - [ ] effective configuration
  - [ ] stage timings and statuses
  - [ ] detected language, BPM, GAP, duration, and note counts
  - [ ] similarity score and configured pass/fail thresholds
- [ ] Include an intermediate transcription/pitch section:
  - [ ] word timestamps
  - [ ] pitch frames colored by confidence
  - [ ] detected pauses
  - [ ] separated-track artifact links when available
- [ ] Include an alignment section:
  - [ ] reconstructed lyric lines
  - [ ] aligned/interpolated word and syllable timing
  - [ ] alignment source/confidence information
  - [ ] warnings and unmatched regions
- [ ] Include a final result section:
  - [ ] generated Ultrastar notes grouped by verse
  - [ ] timing, duration, pitch, lyric, and note type
  - [ ] reference notes when validation mode is active
- [ ] Include a comparison section with reference and candidate notes on a shared time/pitch axis.
- [ ] Visually highlight timing, duration, pitch, missing-note, and extra-note differences.
- [ ] Include a human-readable score table and machine-readable embedded score JSON.
- [ ] Produce a self-contained UTF-8 HTML file that works offline.
- [ ] Escape all uploaded metadata and lyrics before inserting them into HTML.
- [ ] Provide a CLI entry point for report generation from existing artifacts.
- [ ] Keep temporary compatibility wrappers for:
  - [ ] `python ultrastar_to_html.py song.txt output.html`
  - [ ] `python pitch_to_html.py transcription.json output.html`
- [ ] Add a direct Download Validation Report action in Dash.

### Tests

- [ ] Snapshot-test report sections and key labels.
- [ ] Parse generated HTML to verify valid structure and escaping.
- [ ] Test reports with and without reference songs, pitch frames, pauses, or optional stems.
- [ ] Visually inspect the report for the short fixture and Diggy Diggy Hole.

### Exit gate

- [ ] One generated HTML document clearly shows intermediate pitch/transcription data, alignment output, the final candidate song, reference comparison, and similarity metrics.

---

## Phase 7 - End-to-end reference-song validation workflow

### Required workflow

```text
Upload reference MP3 + Ultrastar TXT
  -> parse reference metadata and notes
  -> reconstruct and review lyrics
  -> choose optional safe UI settings
  -> run the complete pipeline using the MP3
  -> generate a separate candidate Ultrastar TXT
  -> score reference versus candidate with score_songs logic
  -> create the unified HTML report
  -> download candidate TXT/ZIP, scores JSON, and report HTML
```

### Checklist

- [ ] Add a clearly labeled Validation Mode to Dash.
- [ ] Require both an MP3 and Ultrastar TXT in Validation Mode.
- [ ] Validate the pair before starting expensive processing.
- [ ] Parse and display reference title, artist, BPM, GAP, duration, and note count.
- [ ] Reconstruct lyrics from the reference and allow the user to correct them before the run.
- [ ] Store the exact submitted lyrics in the effective run manifest.
- [ ] Keep the reference TXT immutable and give the generated file a distinct candidate filename.
- [ ] Run every normal processing stage; do not shortcut the pipeline using reference timing or pitch.
- [ ] Ensure reference metadata does not leak into measured candidate timing/pitch beyond explicitly supplied title, artist, and lyrics.
- [ ] Generate `candidate.txt` before invoking similarity scoring.
- [ ] Invoke the shared scoring API used by `score_songs.py`.
- [ ] Persist `similarity.json` and the effective validation thresholds.
- [ ] Generate the combined HTML report automatically.
- [ ] Show stage progress, errors, score summary, and artifact download links in Dash.
- [ ] Support rerunning the same reference with different UI settings while retaining separate run IDs.
- [ ] Add an artifact bundle containing:
  - [ ] original reference TXT
  - [ ] candidate TXT
  - [ ] normalized MP3 or a documented reference to it
  - [ ] transcription JSON
  - [ ] alignment/debug JSON
  - [ ] effective settings JSON
  - [ ] similarity JSON
  - [ ] combined report HTML
  - [ ] final Ultrastar ZIP

### Automated acceptance test

- [ ] Start the Python app in a test configuration.
- [ ] Upload the fixture MP3 and reference Ultrastar TXT through the same public endpoint/callback path as the UI.
- [ ] Wait for every background stage to complete.
- [ ] Assert that all expected artifacts exist and are registered in the manifest.
- [ ] Parse the generated candidate TXT with the canonical parser.
- [ ] Run the shared `score_songs` API against reference and candidate.
- [ ] Assert nonzero matched notes and measured baseline thresholds.
- [ ] Generate and structurally verify the combined HTML report.
- [ ] Confirm the downloadable ZIP contains the expected files.

### Exit gate

- [ ] A user can validate the entire pipeline from only an existing MP3 + Ultrastar TXT pair without using scripts or manually assembling intermediate files.

---

## Phase 8 - Dash application and persistence

### Checklist

- [ ] Build a simple Dash layout for normal generation and Validation Mode.
- [ ] Implement upload controls for audio/video, lyrics, and reference TXT.
- [ ] Avoid storing large pitch-frame arrays or media bytes in browser-side `dcc.Store` components.
- [ ] Store only project/run IDs and lightweight UI state in the browser.
- [ ] Run the long pipeline in a background worker with progress reporting.
- [ ] Add normal-mode result downloads for TXT and ZIP.
- [ ] Add validation-mode downloads for candidate, scores, report, and artifact bundle.
- [ ] Add project/draft save, list, load, rerun, and delete operations.
- [ ] Version persisted project and draft schemas.
- [ ] Use atomic JSON writes and opaque IDs rather than accepting arbitrary client filesystem paths.
- [ ] Implement safe Range-capable media routes if browser audio/video preview is retained.
- [ ] Expose the whitelisted Advanced Settings generated from central configuration metadata.
- [ ] Display the effective settings used by completed runs.
- [ ] Keep callback modules organized by feature rather than creating one large callback file.

### Tests

- [ ] Test callback/application services independently from Dash rendering.
- [ ] Test upload limits and invalid file handling.
- [ ] Test project save/load and schema migration.
- [ ] Test concurrent requests do not mix artifacts or configuration.
- [ ] Add Python browser tests for the normal and validation workflows.

### Exit gate

- [ ] The full non-editor workflow is usable through Dash without Node, Next.js, or the old FastAPI proxy service.

---

## Phase 9 - Parity, documentation, and cutover

### Checklist

- [ ] Run deterministic parity tests against all frozen TypeScript fixtures.
- [ ] Run the full reference-song validation workflow on Diggy Diggy Hole.
- [ ] Compare baseline and Python similarity metrics.
- [ ] Investigate material regressions before changing acceptance thresholds.
- [ ] Run at least one real GPU smoke test and one CPU-compatible reduced test.
- [ ] Verify fresh installation and startup on Windows.
- [ ] Replace the root README architecture and setup instructions with the Python/Dash workflow.
- [ ] Document configuration precedence and every UI-settable option.
- [ ] Document the validation workflow and interpretation of similarity metrics.
- [ ] Document report contents and artifact locations.
- [ ] Replace root startup scripts with `python -m ultrasongs` or a small wrapper around it.
- [ ] Move the old application source into `legacy/typescript/` using history-preserving moves where practical:
  - [ ] `app/`
  - [ ] `public/`
  - [ ] TypeScript/Next/Tailwind configuration
  - [ ] `package.json` and lockfiles
  - [ ] TypeScript pipeline scripts
  - [ ] the original Python HTTP microservice required by the legacy stack
- [ ] Add `legacy/typescript/README.md` with frozen setup and run instructions.
- [ ] Do not archive generated `node_modules/` or `.next/` directories.
- [ ] Keep the three original utility CLIs as compatibility wrappers until users have adopted the new commands.
- [ ] Confirm there are no active TypeScript imports or Node runtime steps outside `legacy/typescript/`.

### Final definition of done

- [ ] The active application starts with Python only.
- [ ] All active configuration is defined and validated in one central settings model.
- [ ] Safe processing and validation options can be overridden from the Dash UI.
- [ ] Every run records its effective configuration.
- [ ] Normal MP3/video + lyrics generation works end to end.
- [ ] MP3 + reference Ultrastar TXT validation works end to end.
- [ ] `score_songs.py` remains usable and shares its implementation with the app.
- [ ] Similarity metrics are persisted, shown in Dash, and included in the HTML report.
- [ ] One combined report replaces the separate Ultrastar and pitch HTML views while retaining compatibility wrappers.
- [ ] Generated Ultrastar TXT/ZIP artifacts are semantically equivalent to or measurably better than the baseline.
- [ ] Core processing, scoring, reporting, and persistence can be tested without starting Dash.
- [ ] The old TypeScript application remains available as a reference under `legacy/typescript/`.
- [ ] No editor functionality has been included in this migration.

## Recommended implementation order

Execute the phases in order, with two deliberate exceptions:

1. Build the canonical Ultrastar parser before modifying `score_songs.py` or either HTML generator.
2. Build the structured artifact manifest before the combined report, because the report should consume stable stage outputs instead of discovering files by naming convention.

Do not move the TypeScript source into `legacy/` until all Phase 9 parity and end-to-end validation gates pass.
