# UltraSongs

UltraSongs is a local Python and Dash application for generating UltraStar karaoke
songs from audio, and for repairing existing songs by reprocessing their MP3 and
comparing the updated chart with the original UltraStar TXT.

The active migration target is the automatic Python pipeline. The older
Next.js/TypeScript implementation remains in the repository as a reference. Its
timeline editor is intentionally not part of the Python port.

## What works

- Generate an UltraStar TXT and ZIP from audio or video plus lyrics.
- Validate a generated chart against an uploaded UltraStar TXT.
- Repair an existing MP3 + TXT pair from the command line.
- Separate vocals and accompaniment with Demucs.
- Detect vocal pitch with torchcrepe.
- Transcribe with faster-whisper, or use the configured WhisperX worker.
- Detect pauses and tempo, align supplied lyrics, and generate timed notes.
- Score old and new charts with timing, duration, coverage, and pitch metrics.
- Produce one self-contained HTML report showing intermediate data, the final chart,
  the reference chart, scores, effective configuration, and validation failures.
- Persist immutable inputs, artifacts, manifests, and effective configuration for every
  run.

There is no manual note editor in the Python application.

## Requirements

- Python 3.11 or newer.
- FFmpeg and FFprobe available on `PATH`.
- Enough disk space for model caches, separated stems, project artifacts, and exports.
- A CUDA-capable GPU is recommended. CPU processing is supported by the underlying
  libraries but can be very slow for full songs.

The ML libraries may download model weights the first time a model is used. Install a
PyTorch build appropriate for your CPU or CUDA environment before installing the
project extras when your platform requires a specific build.

The optional WhisperX engine runs through the separately configured
`whisperx.python_executable`. The current project extras do not create that environment
or install WhisperX into it; use faster-whisper unless a working WhisperX environment has
been prepared.

## Installation

Create and activate a virtual environment.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the application and the complete processing stack:

```bash
python -m pip install -e ".[runtime,gpu]"
```

The base install, `python -m pip install -e .`, is enough to inspect configuration and
develop the Dash/domain layers, but real processing also needs the `runtime` and `gpu`
extras. Despite the extra name, the ML packages can use CPU when configured to do so.

Confirm FFmpeg and the application entry point:

```bash
ffmpeg -version
python -m ultrasongs --print-config
```

## Run the Dash application

Start the local server:

```bash
python -m ultrasongs
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050). The host, port, and debug mode
are configurable.

### Generate a new song

1. Select **Generate a song**.
2. Upload an audio file. An optional video can be uploaded separately.
3. Enter title, artist, and complete lyrics.
4. Review Advanced Settings if the defaults are not appropriate.
5. Start the run and follow stage progress.
6. Download the generated TXT, ZIP, and pipeline report.

Supported audio extensions default to MP3, WAV, FLAC, M4A, and OGG. Supported video
extensions default to MP4, MKV, WEBM, MOV, and AVI.

### Validate against an existing song

1. Select **Validate against a reference**.
2. Upload the audio and the existing UltraStar TXT.
3. Review the title, artist, and reconstructed lyrics populated from the reference.
4. Correct the reconstructed lyrics before processing when the old TXT does not retain
   reliable word boundaries.
5. Run the pipeline.
6. Inspect the displayed scores and download the combined HTML report.

The uploaded reference bytes are kept unchanged as an immutable artifact. Validation
never turns the reference into an editable project and never overwrites it.

## Repair an existing MP3 + UltraStar TXT

The repair command is the most direct end-to-end workflow for an existing song:

```powershell
python -m ultrasongs repair --audio "C:\Songs\Example\song.mp3" --song "C:\Songs\Example\song.txt" --output-dir "exports"
```

The command:

1. Parses and preserves the original TXT.
2. Uses its title, artist, and reconstructed lyrics as pipeline inputs.
3. Normalizes the MP3 and separates vocals/accompaniment.
4. Detects pitch, pauses, transcription, and tempo.
5. Aligns lyrics and generates a new UltraStar chart.
6. Compares the new chart with the original using the canonical scorer behind
   `score_songs.py`.
7. Applies the configured validation thresholds.
8. Exports a review bundle and prints a score summary.

Because lyric reconstruction from legacy UltraStar syllable tokens is best effort, a
correct lyrics file is strongly recommended when one is available:

```powershell
python -m ultrasongs repair --audio "song.mp3" --song "song.txt" --lyrics-file "lyrics.txt" --output-dir "exports"
```

You can also override bad or missing metadata:

```powershell
python -m ultrasongs repair --audio "song.mp3" --song "song.txt" --title "Correct title" --artist "Correct artist"
```

### Per-run settings

`--set` accepts repeatable, UI-safe dotted configuration paths. Values are parsed as
JSON when possible:

```powershell
python -m ultrasongs repair --audio "song.mp3" --song "song.txt" --set transcription.model=small --set transcription.language=en --set "pitch.confidence_thresholds=[0.6,0.3]"
```

Startup-only options such as executable paths, storage roots, upload limits, and device
selection cannot be changed with `--set`; use a configuration file or environment
variables for those.

Other useful flags:

- `--config settings.toml` loads a JSON or TOML configuration file.
- `--json` prints a machine-readable final result.
- `--fail-on-threshold` returns exit code 3 when processing succeeds but the configured
  similarity thresholds fail.
- `--output-dir PATH` selects the parent directory for the unique repair bundle.

Processing or input errors return exit code 1. A completed run returns 0 unless
`--fail-on-threshold` is active and validation fails.

### Repair bundle

Each run creates `repair-<title>-<run-id>/` beneath the selected output directory. It
contains:

```text
<title>-original.txt       exact copy of the supplied reference
<title>-lyrics-used.txt    reconstructed or explicitly supplied lyrics
<title>-updated.txt        newly generated UltraStar chart
<title>-updated.zip        chart, normalized audio, and configured media artifacts
<title>-comparison.html    self-contained visual processing/comparison report
<title>-scores.json        similarity metrics, threshold outcome, and artifact paths
```

The application also retains the complete project/run manifest and intermediate
artifacts under `projects/` by default.

## Configuration

All Python configuration is declared and validated in
`src/ultrasongs/config.py`. Precedence is:

```text
built-in defaults < JSON/TOML file < environment variables < validated per-run overrides
```

Print the resolved startup configuration at any time:

```bash
python -m ultrasongs --print-config
python -m ultrasongs --config settings.toml --print-config
```

Example `settings.toml`:

```toml
[ultrasongs.server]
host = "127.0.0.1"
port = 8050
debug = false

[ultrasongs.paths]
temp_dir = "tmp"
projects_dir = "projects"
exports_dir = "exports"

[ultrasongs.transcription]
model = "medium"
device = "cuda"
compute_type = "float16"

[ultrasongs.separation]
model = "htdemucs"
device = "cuda"

[ultrasongs.pitch]
model = "full"
device = "cuda"

[ultrasongs.validation]
minimum_matched_notes = 1
minimum_match_ratio = 0.5
maximum_timing_rmse_ms = 500.0
maximum_duration_rmse_ms = 500.0
maximum_pitch_distance_semitones = 2.0
```

Pass it to Dash or repair:

```bash
python -m ultrasongs --config settings.toml
python -m ultrasongs repair --config settings.toml --audio song.mp3 --song song.txt
```

Environment variables use the `ULTRASONGS_` prefix and a double underscore between
configuration levels.

PowerShell:

```powershell
$env:ULTRASONGS_TRANSCRIPTION__MODEL = "small"
$env:ULTRASONGS_TRANSCRIPTION__DEVICE = "cuda"
$env:ULTRASONGS_PATHS__PROJECTS_DIR = "D:\UltraSongs\projects"
python -m ultrasongs
```

You can also point to a configuration file with `ULTRASONGS_CONFIG`.

Every run stores the complete effective configuration snapshot beside its artifacts,
including accepted UI/CLI overrides.

## Pipeline and artifacts

The default pipeline stages are:

```text
intake -> normalize_audio -> load_audio -> separate -> pitch -> pauses
       -> transcribe -> tempo -> align -> generate -> package -> score -> report
```

`score` is present when a reference TXT is supplied. The report is created whenever
reporting is enabled by the runner.

Default working directories:

- `tmp/`: temporary per-run processing files, removed after completion or failure.
- `projects/`: persistent projects, manifests, immutable inputs, and artifacts.
- `exports/`: human-facing repair bundles.
- `reports/` and `drafts/`: reserved application paths in the central configuration.

Project and artifact IDs are opaque. Artifact paths should be resolved through the
repository APIs, which verify ownership and optionally verify hashes.

## Standalone compatibility tools

The legacy command names remain as thin wrappers around the canonical Python domain
modules.

Compare two charts:

```bash
python score_songs.py reference.txt candidate.txt
python score_songs.py reference.txt candidate.txt --json
```

Render a chart as HTML:

```bash
python ultrastar_to_html.py song.txt song.html
```

Render transcription/pitch JSON as HTML:

```bash
python pitch_to_html.py transcription.json transcription.html
```

The repair pipeline report combines both types of visualization and should be preferred
for full end-to-end review.

## Testing

Install test and lint dependencies:

```bash
python -m pip install -e ".[test,dev]"
```

Run the fast suite:

```bash
python -m pytest tests/unit -q
python -m ruff check src tests score_songs.py ultrastar_to_html.py pitch_to_html.py
```

Run the optional local parity fixture:

```bash
python -m pytest tests/parity -m parity -q
```

Run the complete repair command against a real MP3/TXT pair:

```powershell
python -m pytest tests/end_to_end/test_repair_cli.py -m e2e --e2e-audio "C:\Songs\Example\song.mp3" --e2e-song "C:\Songs\Example\song.txt"
```

Add `--e2e-config settings.toml` when the real-model test needs non-default device or
model settings. The E2E test is skipped unless both explicit fixture paths are supplied.
It may take a long time and may download model weights. Default tests do not download
models or require a GPU.

The real E2E test invokes the same `python -m ultrasongs repair` entry point a user
runs, then verifies successful stages, exact reference preservation, parseable output,
ZIP contents, score JSON, and the old/new HTML report.

## Architecture

```text
src/ultrasongs/
  __main__.py            CLI and Dash entry point
  app.py                 Dash application factory
  config.py              only configuration source
  services.py            shared default service construction
  cli/repair.py          MP3 + TXT repair application workflow
  domain/
    alignment/           deterministic lyric alignment and syllabification
    ultrastar/           canonical models, parser, writer, generation, archive
    scoring/             canonical old/new similarity implementation
    reporting/           unified self-contained HTML report
    validation.py        reference inspection and threshold evaluation
  processing/
    pipeline.py          end-to-end orchestrator and stage persistence
    media.py             FFmpeg boundary
    separation.py        Demucs adapter
    pitch_detection.py   torchcrepe adapter
    transcription.py     faster-whisper adapter
    whisperx.py          WhisperX worker adapter
    pitch.py             deterministic pitch aggregation
    pauses.py            pause detection
    tempo.py             BPM detection and fallback
  storage/               project, manifest, and immutable artifact repositories
  web/                   Dash layout, callbacks, local jobs, and downloads
```

Tests are split into deterministic unit coverage, optional local parity tests, and an
opt-in real-model end-to-end command test.

The previous TypeScript implementation remains under `app/`, with its Python service
under `python/`. Keep it available as migration reference until the Python real-model
parity and cutover gates are complete.

See `python_migration_plan.md` for the phase checklist and current migration status.

## Troubleshooting

### FFmpeg is not found

Confirm both commands are on `PATH`, or set `ffmpeg.executable` and
`ffmpeg.ffprobe_executable` in a configuration file:

```bash
ffmpeg -version
ffprobe -version
```

### A processing module is missing

Install the complete extras in the active environment:

```bash
python -m pip install -e ".[runtime,gpu]"
```

If PyTorch or torchaudio cannot use CUDA, install mutually compatible builds for your
driver/platform and set the affected device settings to `cpu` or `auto` while diagnosing.

### The first run appears slow

Demucs, Whisper, and pitch models are expensive and may populate local caches during
their first use. A CPU run can be substantially slower than a CUDA run.

### Reconstructed lyrics contain incorrect spaces

UltraStar TXT files do not always preserve enough information to recover word boundaries.
Review lyrics in Dash or pass a known-good UTF-8 file with `--lyrics-file`.

### Similarity validation fails even though outputs exist

Generation and validation are separate outcomes. Inspect `<title>-scores.json` and the
comparison HTML for reference coverage, timing/duration RMSE, pitch distance, and the
specific configured threshold failures. Use `--fail-on-threshold` only when failed
validation should fail automation.
