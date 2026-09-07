# Ultrasongs - Agent Instructions

## Project Overview

Ultrasongs generates [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from audio/video + lyrics. Pure Python CLI app, runs fully local with no external AI APIs.

**Pipeline (default hybrid backend):** FFmpeg extract → N×(Demucs + faster-whisper) → Smith-Waterman consensus vote → pause detection → lyrics mapped onto transcript → pause-delimited lyric/audio chunks → N× WhisperX forced alignment (word + character timing) → median timing consolidation → torchcrepe pitch + band energy → BPM detection → Smith-Waterman lyric alignment + note segmentation → `.txt` + ZIP + HTML preview.

## Architecture

Single Python CLI package under `cli/`, invoked via `python -m cli` or `ultrasongs` (when the `ultrasongs-cli` package is installed, entry point `ultrasongs = cli.__main__:main`). An optional Dash web UI (`cli/web/`, `ultrasongs web`) wraps the same pipeline.

```
User (CLI) → argparse → cli/__main__.py → pipeline stages → output/
User (browser) → Flask/Dash → cli/web/jobs.py (FIFO queue) → cli/pipeline.run_process → web_jobs/<id>/output/
```

- **CLI layer**: `argparse` subcommands (`process`, `import`, `diff`, `preview`, `lyrics`, `edit`, `web`), global flags `-c/--config`, `-v/--verbose`, `-q/--quiet`
- **Pipeline**: sequential stages, each a module in `cli/`; heavy ML imports are lazy-loaded inside functions
- **Config**: `cli/config.jsonc` (JSON with comments), loaded into a frozen `Config` dataclass (`cli/config.py`); code-level fallback defaults live in the dataclass, invalid values fall back with a warning
- **Logging**: `cli/logging_setup.py` — stdout handler, `[timestamp] [name] message` format

## Stack

| Component | Technology |
|---|---|
| CLI | argparse (stdlib) |
| Web UI | Dash (optional extra `ultrasongs-cli[web]`; pulls Flask + Plotly) |
| Package | setuptools, `cli/pyproject.toml` (Python >=3.10,<3.14) |
| Audio separation | Demucs, torchaudio |
| ASR backends | faster-whisper (standalone, default) or WhisperX 3.8.6 (legacy/VAD path) |
| Forced alignment | WhisperX + language-specific wav2vec2 (character-level timings) |
| Pitch detection | torchcrepe (full model, Viterbi) |
| BPM detection | librosa |
| Syllabification | pyphen |
| Audio I/O | FFmpeg (subprocess), soundfile, lameenc |
| Numeric | numpy |
| Debug plots | matplotlib (implicit dependency of librosa) |

GPU note: CUDA 12.8 + PyTorch 2.8 install command is documented at the top of `cli/requirements.txt`. On Windows, `cli/whisperx_transcribe.py` registers FFmpeg's DLL directory so pyannote/TorchCodec can load shared libs.

## Commands

```bash
pip install -e cli/                    # Install package (creates `ultrasongs` cmd)
pip install -r cli/requirements.txt    # Install dependencies

python -m cli process --lyrics lyrics.txt --mp3 song.mp3 --title "Title" --artist "Artist" [--video song.mp4] [--output ./dir]
# With an Ultrastar .txt as lyrics input, --mp3/--title/--artist are optional
# and fall back to the file's #MP3/#TITLE/#ARTIST tags (BPM/GAP are ignored).
python -m cli process --lyrics output/tit31.txt
python -m cli import --txt existing.txt --mp3 existing.mp3
python -m cli diff --original a.txt --generated b.txt      # exit 0 if within tolerances, else 1
python -m cli preview --txt output.txt --pitch tmp/whisperx_pitch.json
python -m cli lyrics --txt output/tit31.txt [--output lyrics.txt]   # plain lyrics to stdout or file
python -m cli edit --txt output/tit31.txt [--pitch tmp/whisperx_pitch.json] [--vocals output/vocals.mp3] [--embed-audio] [--output out.html]
python -m cli process ... --no-intermediates   # skip intermediate temp files in the output ZIP

# Web UI (requires `pip install 'ultrasongs-cli[web]'` or dash in requirements.txt).
# Password from ULTRASONGS_WEB_PASSWORD env var, or ULTRASONGS_WEB_PASSWORD in ./.env.local (gitignored); --no-auth only on localhost.
ULTRASONGS_WEB_PASSWORD=secret python -m cli web [--host 127.0.0.1] [--port 8080] [--no-auth] [--web-config path]

python -m cli -v process ...           # Verbose (DEBUG) logging
python -m cli -q process ...           # Quiet (WARNING+) logging
python -m cli -c path/to/config.jsonc  # Custom config file
```

Partial execution and resume:
```bash
python -m cli process ... --stage transcribe                       # Stage choices: extract, transcribe, align, generate, all
python -m cli process ... --resume tmp/name_transcribe.json --stage align   # Resume from saved TranscribeResult JSON
```

Diff tolerances (`cli/diff.py`): BPM ±2, GAP exact match, singing-note count exact, per-note beat offset ±4, duration ±4, pitch ±3 semitones.

`process --lyrics` also accepts an Ultrastar `.txt` file: if the first non-empty line is a `#` header, plain lyrics are automatically extracted from it (`extract_lyrics_from_ultrastar`) and the pipeline runs on those. `--title`, `--artist`, and `--mp3` then fall back to the file's `#TITLE`, `#ARTIST`, and `#MP3` tags (the `#MP3` path is resolved relative to the .txt file's directory) when not given on the command line. `#BPM` and `#GAP` are ignored.

`--mp3` accepts video files (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`) as well as audio: the extract stage pulls the audio out with FFmpeg, and the original video is then treated like `--video` — copied into the output/ZIP and referenced by a `#VIDEO` tag (an explicit `--video` always wins). The package output always contains the extracted original MP3, the first-pass htdemucs vocals and accompaniment stems, and the original video when the input was one.

Each processing run writes into its own per-song folder `<output_dir>/<artist> - <title>/`, and **all** output files use the sanitized `<artist> - <title>` base name (`cli/pipeline.sanitize_output_name`, built on `sanitize_filename`): `.txt`, `.mp3`, video (same extension), `.zip`, `.html` (preview), `_editor.html`, `_vocals.mp3`, `_accompaniment.mp3`. The generated `.txt`'s `#MP3`/`#VIDEO` tags and the editor's audio hint reference these names. `--no-intermediates` (CLI) keeps the ZIP to the package output only; by default the ZIP also bundles the job's `tmp/` artifacts under `intermediates/` (`cli/package.collect_intermediates`).

## Note Editor (`edit`)

`python -m cli edit --txt song.txt` generates a **self-contained** HTML note editor (no server, no build step, no CDN). It visually matches the `preview` SVG output and lets the user hand-tune notes by drag (pitch/start), edge-drag (duration), split, merge, delete, gold-toggle, and lyric edit, with an undo stack, a live FFT spectrogram background, and audio + MIDI + metronome playback. Save = browser Blob download of the edited `.txt`.

- **Source of truth**: the parsed notes list. `start`/`dur` are beats (int), `pitch` is MIDI (int); ms is derived as `gap + start*beatMs` with `beatMs = 60000/bpm/4`. The file's `#BPM` is used verbatim (it already bakes in `beat_resolution_multiplier`).
- **Two beat concepts**: `beatMs` (16th note at the listed BPM) is the note **snap grid**; `pulseMs = 60000/bpm` (quarter note) is the **visual grid + metronome** spacing.
- **Header**: leading `#` lines are preserved verbatim (incl. `#COVER`/`#BACKGROUND`/`#CREATOR`); CRLF→LF on save. The body is serialized in the same shape as `build_ultrastar_txt`.
- **Reloading progress**: the toolbar's **Load .txt** button imports a saved Ultrastar `.txt` (UTF-8, falling back to Windows-1252 on decode failure), replacing notes, raw header, `#BPM`/`#GAP` (and derived `beatMs`/`pulseMs`), and title/artist in the top bar. The in-browser parser mirrors `cli/ultrastar.parse_ultrastar_txt` (note regex, rest/corrupted-note recovery, comma-decimal BPM). It resets selection, undo stack, and page, and keeps the audio + pitch overlay as-is; `srcName` follows the loaded file so the next download reuses its name. Failures show "txt load failed" in the meta chip.
- **Pitch overlay**: `--pitch` embeds the full `whisperx_pitch.json` word/frame list (s→ms) as `pitchWords`; the JS slices it into the live page window, so overlays stay correct under edits that change line structure. The toolbar's **Load pitch JSON** button can replace `pitchWords` at runtime by loading a `whisperx_pitch.json` file (same format as `--pitch`; accepts a bare `words` array or a `{words:[...]}` object), which re-renders the whisper word labels + CREPE dots; a top-bar chip shows the loaded file name + word count. `--vocals` names the suggested audio file; `--embed-audio` base64-embeds it.
- **Payload** is injected into `cli/editor_template.html` via `__EDITOR_DATA__`; a 64-entry magma LUT is injected via `__MAGMA__` (both escaped so `</` can't close the script).

## Web Service

`python -m cli web` runs a single-threaded-worker Dash app (`cli/web/`) that submits the same `cli.pipeline.run_process` the CLI uses. Requires the optional `dash` dependency (`pip install 'ultrasongs-cli[web]'`); the subcommand prints an install hint and exits 1 if Dash is missing.

- **Auth**: password from the `ULTRASONGS_WEB_PASSWORD` environment variable, falling back to a `ULTRASONGS_WEB_PASSWORD=...` entry in `./.env.local` (gitignored; env var wins) (constant-time compare, Flask session cookie). No TLS is provided — bind to `127.0.0.1` (default) or put a reverse proxy in front. `--no-auth` disables login but is rejected unless the host is `127.0.0.1`/`localhost`. Web server settings (host, port, `web_dir`, retention, upload cap, poll interval) live in `cli/web_config.jsonc` (`--web-config` to override).
- **Queue**: one job at a time, FIFO (`cli/web/jobs.py`). Uploads are staged, validated (extension + size cap), then moved into the job dir. Each job gets `web_jobs/<id>/{upload,tmp,output}` plus a `job.json` manifest; jobs older than `job_retention_days` are pruned at startup and before each submission.
- **UI**: one form (title/artist/lyrics + audio/video upload) plus an auto-generated settings accordion with one control per `config.jsonc` key (44), driven by `cli/web/settings_meta.py`. Jobs show live status, queue position, log tail, and download links (ZIP, note editor, HTML preview, stems) served from `/download/<job_id>/<file>`.
- **Per-job config**: form values override the base `Config` (revalidated via `config_from_dict`); the job's `temp_dir`/`output_dir` are forced to the job dir, and the ZIP includes intermediates.

## Code Conventions

- Python 3.10+, type hints via dataclasses in `cli/pipeline_types.py`
- All pipeline code in `cli/` package; repo root holds one-off debug scripts (see below), not pipeline code
- No emojis in code or comments unless explicitly requested
- Heavy model imports (torch, whisperx, demucs, ...) are done lazily inside functions so lightweight modules stay importable/testable
- Config via `cli/config.jsonc`, not environment variables
- Follow existing patterns in neighboring files before introducing new libraries

## Key Files

| File | Purpose |
|---|---|
| `cli/__main__.py` | CLI entry point, argparse parser, thin adapters over `cli.pipeline` (5 logged steps) |
| `cli/pipeline.py` | `ProcessRequest`/`ProcessResult` dataclasses, `run_process()` shared by CLI and web, `prepare_lyrics_input()`, `sanitize_filename()`, `sanitize_output_name()` |
| `cli/config.py` | JSONC loading/stripping (`load_jsonc`), `config_from_dict()`, frozen `Config` dataclass, validation |
| `cli/pipeline_types.py` | Shared dataclasses: `PitchFrame`, `CharacterTimestamp`, `WordTimestamp`, `Pause`, `AlignedSyllable`, `BpmResult`, `TranscribeResult`, `UltrastarNote`, `UltrastarMeta` (with `to_dict`/`from_dict` for resume) |
| `cli/ffmpeg_extract.py` | FFmpeg: video/audio → mono MP3 (128 kbps) in `tmp/` |
| `cli/ffmpeg_pcm.py` | FFmpeg: MP3 → raw float32 PCM bytes (used for note-plot spectrograms) |
| `cli/transcribe.py` | Orchestrates the transcribe stage: Demucs separation (N passes), ASR, consensus, pauses, hybrid chunking + WhisperX timing, stem saving, BPM, torchcrepe pitch + band-limited energy |
| `cli/whisperx_transcribe.py` | WhisperX/faster-whisper adapters: model loading, transcription, `align_segments()` forced alignment with character timings, hallucination-artifact filters, Windows DLL setup |
| `cli/hybrid_transcribe.py` | Hybrid path helpers: `align_lyrics_approximately()` (Smith-Waterman lyric→ASR mapping), `build_lyric_chunks()` (split at pause midpoints), `slice_audio()`, `offset_aligned_words()` |
| `cli/consensus.py` | Multi-run consolidation: `word_similarity()`, `consolidate_transcription_runs()` (majority vote, coherent timing), `consolidate_timing_runs()` (median coherent candidate) |
| `cli/bpm_detect.py` | `detect_bpm()`: median of overall + 30 s chunk tempo estimates, per-chunk stability, beat-grid phase check after pauses, first-beat time via `beat_track()` |
| `cli/debug_bpm.py` | Standalone debug script: splits an MP3 into 30 s chunks (FFmpeg) and compares BPM estimates |
| `cli/align.py` | `align_lyrics()`: Smith-Waterman phonetic alignment (affine gaps), interpolation for unmatched words, pause-boundary clamping, syllabification, `_note_segments()` per-syllable vocal-activity trimming + pitch-change splitting, optional matplotlib diagnostic plots |
| `cli/syllabify.py` | Syllable splitting via pyphen (language alias map, cached hyphenators) |
| `cli/ultrastar.py` | `ms_to_beats()`, `build_ultrastar_txt()`, `parse_ultrastar_txt()`, `extract_lyrics_from_ultrastar()` |
| `cli/generate.py` | `generate_ultrastar()`: beat mapping anchored to first beat (`#GAP`), overlap prevention, line breaks |
| `cli/package.py` | Output packaging (`.txt`, MP3, video, stems, ZIP), `collect_intermediates()` for ZIP bundling |
| `cli/web/app.py` | Dash app factory: layout, settings accordion, callbacks, `/download` route, `run_server()` |
| `cli/web/jobs.py` | `JobManager`: single-worker FIFO queue, per-job log capture, retention pruning, `snapshot()` |
| `cli/web/auth.py` | Flask session login (`/login`, `/logout`), constant-time password check, `resolve_auth()` host guard, `load_web_password()` (env var, then `.env.local`) |
| `cli/web/settings_meta.py` | `SettingMeta` list (one per config key) driving the auto-generated settings form |
| `cli/web_config.jsonc` | Web server config (host, port, `web_dir`, retention, upload cap, poll interval) |
| `cli/diff.py` | Compare two Ultrastar `.txt` files with tolerances, `DiffReport.print()` |
| `cli/html_preview.py` | `generate_preview()`: HTML with SVG pitch visualization, beat grid, confidence/amplitude colors, optional `whisperx_pitch.json` overlay |
| `cli/editor.py` | `generate_editor()`: build the note-editor payload (raw header, notes, full `pitchWords`), inject into the template with a magma LUT, optional base64 audio embed |
| `cli/editor_template.html` | Self-contained editor SPA (inlined CSS+JS): SVG note overlay, live FFT background, drag/split/merge/delete/gold/lyric + undo, Web Audio playback, Blob download |
| `cli/pitch_to_html.py` | Standalone script: render a pitch JSON as scrollable HTML verse visualizations |
| `cli/logging_setup.py` | `setup_logging()` / `get_logger()` |

Repo-root one-off debug scripts (read `tmp/`/`output/` artifacts, not part of the pipeline): `check_pitch.py`, `compare_timings.py`, `octave_spectrogram.py`, `pitch_to_html.py`, `score_songs.py`. Repo-root `extract_lyrics.py` is a standalone wrapper around `cli/ultrastar.extract_lyrics_from_ultrastar` (same function as the `lyrics` subcommand). Root `tests/` is an empty leftover (only stale `__pycache__`); real tests live in `cli/tests/`.

## Transcription Backends

Selected by `transcription_backend` in config (see `transcribe.py`):

- **`"faster-whisper"` (default, hybrid):** lyrics are required. Runs `transcribe_runs` × (Demucs separation on the original audio + standalone faster-whisper on that pass's vocals, with the lyrics as `initial_prompt`). Per-pass word lists are consolidated into one consensus (`consensus.py`). Pauses are detected on the first pass's vocals; lyrics are mapped onto the approximate transcript (`hybrid_transcribe.py`) and split into `LyricChunk`s at pauses longer than `whisperx_chunk_pause_ms` (cut at the silence midpoint, no audio removed). Each chunk is then force-aligned by WhisperX `whisperx_align_runs` times (positive odd number; artifact filtering disabled because the text is authoritative) and consolidated to the median coherent timing — word and character timestamps always come from the same pass.
- **`"whisperx"` (legacy):** each pass runs WhisperX's batched/VAD transcription (`whisperx_batch_size`) followed by forced alignment with artifact filtering; per-pass results are consolidated like above.

In both backends, stems, BPM, and pitch always come from the first pass's vocal track. Per-pass dumps are written to `tmp/{stem}_{backend}_passes.json`.

## Processing Pipeline

The `process` subcommand runs these stages (`--stage` cuts off after the given stage):

1. **Extract** (`cli/ffmpeg_extract.py`): video/audio → mono 128 kbps MP3 in `tmp/`
2. **Transcribe** (`cli/transcribe.py`, 6 internal steps): load audio → N× (Demucs + ASR + consolidation) → pause detection → hybrid chunking + WhisperX exact timing (default backend) → save stems → BPM detect → torchcrepe pitch + band energy → `TranscribeResult`. Persisted as `tmp/{stem}_transcribe.json` for `--resume`
3. **Align** (`cli/bpm_detect.py` + `cli/align.py`): BPM is reused from the `TranscribeResult` if present (it normally is), else re-detected on the accompaniment stem or full mix. Then Smith-Waterman alignment of lyrics to exact word/character timestamps, syllabification, and per-syllable note segmentation
4. **Generate** (`cli/generate.py`): aligned syllables + BPM → Ultrastar `.txt`, beat grid anchored to the detected first beat (`#GAP` = `first_beat_ms`, fallback `first note - gap_lead_in_ms`), exported BPM scaled by `beat_resolution_multiplier`
5. **Package** (`cli/package.py`): write `.txt`, copy MP3/video/stems, create ZIP in the per-song output folder. All files are named after the sanitized `<artist> - <title>` base name (see `sanitize_output_name`). The note editor HTML (`{artist} - {title}_editor.html`, generated via `cli/editor.generate_editor` from the fresh `.txt` with the `whisperx_pitch.json` overlay and the `_vocals.mp3` audio hint) is written *before* packaging, so it is always part of the ZIP.
6. **Preview** (`cli/html_preview.py`): HTML with SVG pitch visualization, automatically generated at the end of the generate stage, overlaying `tmp/whisperx_pitch.json` when available (written by alignment when `debug_alignment` is on)

## Configuration

`cli/config.jsonc` (44 keys, supports `//` and `/* */` comments). Defaults below are the committed `config.jsonc` values; `cli/config.py` holds fallback defaults for missing/invalid keys.

| Key | Default | Description |
|---|---|---|
| `device` | `"auto"` | GPU/CPU selection (auto, cuda, cpu) |
| `transcription_backend` | `"faster-whisper"` | Hybrid faster-whisper/WhisperX path; `"whisperx"` keeps the legacy VAD ASR path |
| `whisper_model` | `"medium"` | Whisper ASR model (tiny, base, small, medium, large) |
| `whisper_language` | `"en"` | Language hint; empty enables auto-detection |
| `faster_whisper_compute_type` | `"auto"` | CTranslate2 compute type for standalone faster-whisper |
| `whisperx_batch_size` | `8` | Chunks per WhisperX inference batch (legacy backend) |
| `whisperx_compute_type` | `"default"` | CTranslate2 compute type for WhisperX |
| `whisperx_align_model` | `""` | Optional wav2vec2 alignment model override |
| `whisperx_interpolate_method` | `"linear"` | Missing-character timing policy (nearest, linear, ignore) |
| `whisperx_chunk_pause_ms` | `1000` | Split lyrics/audio at pauses strictly longer than this |
| `whisperx_align_runs` | `5` | WhisperX exact-timing passes (positive odd number) |
| `transcribe_runs` | `7` | Demucs + ASR passes consolidated before exact alignment |
| `demucs_model` | `"htdemucs"` | Demucs source separation model |
| `sample_rate` | `44100` | Audio sample rate |
| `pitch_min_hz` | `65.41` | Pitch floor (C2) |
| `pitch_max_hz` | `1046.5` | Pitch ceiling (C6) |
| `crepe_hop_ms` | `10` | torchcrepe analysis hop length |
| `band_energy_min_hz` | `60.0` | Lower bound of the band used for per-frame amplitude proxy |
| `band_energy_max_hz` | `4000.0` | Upper bound of the amplitude band (harmonics/formants) |
| `pause_min_silence_ms` | `400` | Minimum silence for pause detection |
| `pause_threshold_pct` | `5` | RMS energy threshold (% of 95th percentile) |
| `gap_lead_in_ms` | `500` | Fallback lead-in before first note for `#GAP` |
| `linebreak_beat_offset` | `4` | Beats before next note for line breaks |
| `beat_resolution_multiplier` | `2` | Scales exported BPM for a finer Ultrastar beat grid |
| `activity_quiet_confidence` | `0.2` | Min CREPE confidence for "quiet" frames in activity threshold |
| `activity_voiced_confidence` | `0.5` | Min CREPE confidence for "voiced" frames in activity threshold |
| `activity_noise_percentile` | `0.9` | Percentile of quiet frames for noise floor |
| `activity_noise_fallback_percentile` | `0.1` | Percentile of all frames for noise floor fallback |
| `activity_signal_percentile` | `0.5` | Percentile of voiced frames for signal level |
| `activity_signal_fallback_percentile` | `0.75` | Percentile of all frames for signal fallback |
| `activity_threshold_ratio` | `0.2` | Fraction of (signal - noise) added to noise floor |
| `note_min_confidence` | `0.3` | Min CREPE confidence for vocal activity in note segmentation |
| `note_fallback_confidence` | `0.5` | Fallback confidence when no frame meets the activity threshold |
| `note_dropout_gap_ms` | `50` | Max gap (ms) between active frames in one continuous note |
| `note_smooth_window` | `5` | Median filter window (frames) for pitch smoothing |
| `note_pitch_tolerance` | `1` | Max semitone drift kept within a single note |
| `note_min_duration_ms` | `60` | Min duration (ms) for a pitch-change segment |
| `note_frame_step_ms` | `10` | Fallback frame spacing (ms) for note end times |
| `note_segment_plots` | `false` | Write matplotlib diagnostic plots to `tmp/note_segments_plots/` (slow; debug only) |
| `ffmpeg_audio_bitrate` | `"128k"` | Output MP3 bitrate |
| `output_dir` | `"./output"` | Base output directory (a per-song `<artist> - <title>/` folder is created inside it) |
| `temp_dir` | `"./tmp"` | Intermediate files directory |
| `debug_alignment` | `true` | Write alignment debug JSON/backtrace/pitch HTML to temp dir |
| `bpm_use_accompaniment` | `true` | Use Demucs instrumental stem for BPM detection |

## Artifacts

Temp files in `./tmp/`, generated output in `./output/` (both gitignored):

```
./tmp/
  {stem}.mp3                     ← normalized mono MP3
  {stem}_vocals.mp3              ← separated vocals stem
  {stem}_accompaniment.mp3       ← separated instrumental stem
  {stem}_transcribe.json         ← TranscribeResult (resume file)
  {stem}_faster_whisper_passes.json  ← per-pass ASR word dumps (backend-named)
  align_debug.json, align_backtrace.txt          ← alignment debug (debug_alignment)
  whisperx_pitch.json, whisperx_pitch.html       ← aligned words + pitch for preview
  note_segments.txt, note_segments_plots/        ← note-segmentation diagnostics

./output/
  <artist> - <title>/             ← per-song folder (sanitized)
    <artist> - <title>.txt, .mp3, .zip, .html, _editor.html
    <artist> - <title>.<video_ext>            ← video (if input was one)
    <artist> - <title>_vocals.mp3, _accompaniment.mp3  ← optional stems

./web_jobs/                       ← web UI jobs (gitignored)
  staging/                        ← in-flight uploads, deleted after submit
  <job_id>/job.json               ← manifest (id, title, status, created_at)
  <job_id>/upload/original.*      ← the uploaded audio/video
  <job_id>/tmp/                   ← per-job intermediate files
  <job_id>/output/<artist> - <title>/   ← same package output as the CLI
```

## Testing

```bash
pytest cli/tests/
```

| Test File | Coverage |
|---|---|
| `cli/tests/test_align.py` | `normalize_char()`, `phonetic_score()`, `smith_waterman()`, `align_lyrics()` |
| `cli/tests/test_bpm.py` | `detect_bpm()`, per-chunk estimates, phase stability, `BpmResult` round-trip |
| `cli/tests/test_config.py` | Defaults, frozen dataclass, JSONC loading, invalid-value fallback |
| `cli/tests/test_config_dict.py` | `config_from_dict()` overrides, invalid-value fallback, `load_jsonc()` |
| `cli/tests/test_consensus.py` | `word_similarity()`, transcription + timing consolidation |
| `cli/tests/test_diff.py` | Identical files, BPM/beat tolerances, different titles |
| `cli/tests/test_editor.py` | `extract_raw_header()`, `load_pitch_words()`, `build_payload()` shape (full `pitchWords`), `serialize_editor_txt()` round-trip/idempotency, `embed_audio_b64()`, `generate_editor()` (LUT + data injection, embed, errors) |
| `cli/tests/test_generate.py` | Basic generation, line breaks, overlap prevention, video filename |
| `cli/tests/test_hybrid_transcribe.py` | Approximate lyric alignment, chunk splitting at pauses, boundaries, `slice_audio()`, word offsetting |
| `cli/tests/test_package_intermediates.py` | `collect_intermediates()` selection, ZIP `intermediates/` bundling, `--no-intermediates` |
| `cli/tests/test_pipeline.py` | `run_process()` orchestration, `prepare_lyrics_input()`, `sanitize_filename()`, `sanitize_output_name()`, `--stage`/`--resume` |
| `cli/tests/test_pipeline_types.py` | `TranscribeResult`/`WordTimestamp` round-trips, character alignments, legacy pitch-frame recovery |
| `cli/tests/test_syllabify.py` | `split_word()`, `syllabify_line()`, multi-language, unsupported |
| `cli/tests/test_transcribe_alignment.py` | CREPE/band-energy frame-count parity, exact frame alignment (incl. real torchcrepe) |
| `cli/tests/test_ultrastar.py` | `ms_to_beats()`, `build_ultrastar_txt()`, `parse_ultrastar_txt()`, `extract_lyrics_from_ultrastar()`, round-trip |
| `cli/tests/test_web_app.py` | Layout generation (all 44 controls), input validation, config-override flow, process/upload callbacks |
| `cli/tests/test_web_auth.py` | Flask `test_client`: anon redirect, wrong/right password, logout, `--no-auth` host guard |
| `cli/tests/test_web_jobs.py` | submit→running→succeeded/failed, FIFO ordering, log-handler capture, retention pruning, `snapshot()` shape |
| `cli/tests/test_whisperx_transcribe.py` | Word/character extraction, artifact filters, model loading (monkeypatched), alignment-model caching, faster-whisper paths, Windows DLL registration |

## Other

- Test song used for debugging is committed at the repo root: `test_song_full_audio.mp3` + `test_song_lyrics_only.txt` + `test_song_reference_ultrastar_file.txt/html` (a `.vscode` debug config resumes from `tmp/test_song_full_audio_transcribe.json`)
- `test_song_small_audio.mp3` + `test_song_small_lyrics.txt`: a ~35 s quick-test fileset — the full song cut from the `#GAP` (32.646 s, with 0.5 s lead-in) to the end of line 115 of the reference `.txt` (first "Come on brothers sing with me!", ~66.3 s, with 0.5 s tail), lyrics = the first 13 lines of `test_song_lyrics_only.txt`
- Note: the reference `.txt`'s beat grid runs at 1/4 of its listed BPM (effective 4×121.3 ≈ 485 BPM; verified against the audio with word timestamps), so don't use standard `ms_to_beats` math on it for durations
- `docs/pipeline.html` + `docs/pipeline/stage*.svg`: visual documentation of the 7 pipeline stages
- A `.venv` at the repo root is the dev virtualenv (Python 3.13, CUDA wheels)
