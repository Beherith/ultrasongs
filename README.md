# Ultrasongs

> **Work in progress** — functional but rough around the edges.

Generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from any audio or video file + song lyrics. Pure Python — runs fully local, no external AI APIs. CLI-first, with an optional web UI.

**Pipeline:** FFmpeg extract → Demucs vocal separation → multi-pass faster-whisper consensus → Smith-Waterman lyric alignment → pause-delimited lyric/vocal chunks → multi-pass WhisperX exact word/character timing → BPM + torchcrepe analysis → `.txt` + ZIP export.

## Features

- Upload any audio or video format (MP4, MKV, MP3, FLAC, WAV, …)
- Automatic vocal separation via [Demucs](https://github.com/facebookresearch/demucs) `htdemucs`
- Per-word MIDI pitch via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — neural pitch estimation tuned for singing voices
- Multi-pass [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription for accurate text, followed by WhisperX forced alignment for exact word/character timing
- Smith-Waterman phonetic alignment of transcribed words → user lyrics
- BPM detection via [librosa](https://librosa.org/) + beat-accurate note placement
- Syllable splitting (20+ languages via [pyphen](https://github.com/karpathy/pyphen))
- Re-process from an existing Ultrastar `.txt` — lyrics, title, artist, and audio track are extracted automatically
- Standalone lyrics extraction from Ultrastar files (`lyrics` subcommand or `extract_lyrics.py`)
- Partial pipeline execution and resume from intermediate results
- Built-in `.txt` diff tool with configurable tolerances
- HTML preview with SVG pitch visualization
- ZIP download with `.txt` + separated audio tracks
- Optional web UI (`web` subcommand) — upload + process in the browser, per-song settings, job queue, download results

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and in PATH
- PyTorch with CUDA recommended (CPU works but is slow)

## Installation

```bash
# Install package (creates `ultrasongs` command)
pip install -e cli/

# Or just install dependencies to run directly
pip install -r cli/requirements.txt

# Optional: web UI (Dash)
pip install -e "cli/[web]"
```

**Optional (GPU):** CUDA 12.8 with PyTorch 2.8 for faster Demucs/WhisperX inference.

```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
# Full pipeline
python -m cli process --mp3 song.mp3 --lyrics lyrics.txt --title "Title" --artist "Artist"
python -m cli process ... --no-intermediates   # keep the ZIP to package output only

# With video background
python -m cli process --mp3 song.mp4 --lyrics lyrics.txt --title "Title" --artist "Artist" --video song.mp4

# Re-process from an existing Ultrastar file
# Lyrics, title, artist, and audio (#MP3 tag, relative to the .txt file)
# are taken from the file; --mp3/--title/--artist override when given.
python -m cli process --lyrics output/song.txt

# Import existing Ultrastar file
python -m cli import --txt existing.txt --mp3 existing.mp3

# Compare two .txt files
python -m cli diff --original reference.txt --generated output.txt

# Generate HTML preview
python -m cli preview --txt output.txt --pitch tmp/name_pitch.json

# Extract plain lyrics from an Ultrastar file (to stdout or --output file)
python -m cli lyrics --txt output.txt
python extract_lyrics.py output.txt   # standalone script

# Web UI (optional, needs `pip install 'ultrasongs-cli[web]'` or dash)
# Password from the ULTRASONGS_WEB_PASSWORD env var, or ./.env.local (gitignored)
ULTRASONGS_WEB_PASSWORD=secret python -m cli web [--host 127.0.0.1] [--port 8030] [--no-auth]
```

Global flags:

| Flag | Description |
|---|---|
| `-c, --config <path>` | Custom `config.jsonc` (default: `cli/config.jsonc`) |
| `-o, --override <spec>` | Override config keys (repeatable; takes precedence over `-c`). A JSON object (`'{"whisper_model": "small"}'`) or comma-separated `key=value` pairs (`transcribe_runs=5,whisper_model=small`) |
| `-v, --verbose` | DEBUG logging |
| `-q, --quiet` | WARNING and above only |

Partial execution and resume:

```bash
# Run only up to transcription
python -m cli process ... --stage transcribe

# Resume from saved intermediate results
python -m cli process ... --resume tmp/name_transcribe.json --stage align
```

## Web UI

`python -m cli web` serves a single-page Dash app that runs the same pipeline the CLI uses. One job at a time (FIFO queue); each job shows live status, queue position, log tail, and download links (ZIP, HTML preview, stems).

- **Auth:** password from the `ULTRASONGS_WEB_PASSWORD` environment variable, or from a `ULTRASONGS_WEB_PASSWORD=...` entry in `./.env.local` (gitignored; env var wins) (session cookie, constant-time compare). There is no TLS — bind to `127.0.0.1` (default) or put a reverse proxy in front. `--no-auth` disables login but is only allowed on `127.0.0.1`/`localhost`.
- **Flags:** `--host`, `--port`, `--no-auth`, `--web-config <path>` (defaults from `cli/web_config.jsonc`).
- **Per-job settings:** the form auto-generates one control per `config.jsonc` key; values override the base config for that job only. "Reset settings to file defaults" restores them.
- **Jobs:** stored under `web_jobs/<job_id>/{upload,tmp,output}` with a `job.json` manifest; jobs older than `job_retention_days` (default 7) are pruned at startup and before each submission.

## Configuration

Edit `cli/config.jsonc` (supports `//` and `/* */` comments). 44 configuration keys covering GPU selection, model choices, pitch range, pause detection, BPM, note segmentation, output paths, and more. See the file for detailed per-key documentation.

Individual keys can be overridden on the command line with `-o/--override` (repeatable), e.g. `python -m cli -o transcribe_runs=5,whisper_model=small process ...`.

The web UI auto-generates a settings form from the same keys — no separate web config for pipeline options. Web server settings (host, port, job directory, retention, upload cap, poll interval) live in `cli/web_config.jsonc`.

## Pipeline

### Stage 1 — Extract

FFmpeg normalizes input media to mono 128 kbps MP3. Strips video, downmixes to mono, encodes with `libmp3lame`.

### Stage 2 — Transcribe

The transcription stage performs:

1. **Vocal separation** (Demucs `htdemucs`): splits audio into vocals + accompaniment stems.
2. **Initial ASR**: standalone faster-whisper receives the lyrics as an `initial_prompt` and emits approximate word timestamps.
3. **Consensus**: multiple faster-whisper passes are matched and voted before any forced alignment.
4. **Pause detection**: sliding-window RMS energy analysis; frames below 5% of the 95th percentile are silent; consecutive silence ≥400 ms is retained for note/line handling.
5. **Lyric chunking**: the consensus is aligned to the supplied lyrics; lyrics and the primary vocal waveform are split at vocal pauses longer than one second.
6. **Exact timing** (WhisperX + language-specific wav2vec2): forced-aligns the authoritative lyrics inside each audio chunk an odd, configurable number of times, then selects the coherent per-word result nearest the median timing.
7. **Pitch analysis** (torchcrepe): runs the `full` model on vocals at 16 kHz, 10 ms hop, C2–C6 range, with Viterbi decoding.

Produces `{stem}_transcribe.json` with words, character timestamps, MIDI, pitch frames, language, pauses, and stem paths.

### Stage 3 — BPM Detect

`librosa.feature.tempo()` on the MP3 (or accompaniment stem if configured). Falls back to 120 BPM on failure.

### Stage 4 — Align

Smith-Waterman alignment with phonetic scoring:

- Character normalization: lowercase, NFD decompose, strip diacritics while preserving non-Latin scripts
- Phonetic scoring: exact match, articulation groups (vowels, sibilants, stops), cross-group confusions
- Affine-gap DP matrices with backtrack
- Character-anchored word and syllable timing, with interpolation for unmatched text
- Syllabification via pyphen (20+ languages)
- Line break insertion at lyric boundaries

### Stage 5 — Generate

Converts aligned syllables to Ultrastar `.txt`:

- Beat conversion: `round(((ms - gap) / 1000) * (bpm / 60) * 4)`
- Overlap prevention: `start = max(start, prev_end + 1)`
- Line breaks placed 4 beats before next note

### Stage 6 — Package

Writes everything into a per-song `output/<artist> - <title>/` folder. The `.txt`, MP3, video, stems, note editor, preview, and ZIP are all named after the sanitized `<artist> - <title>` base name. The ZIP also bundles the run's intermediate files under `intermediates/` unless `--no-intermediates` is given.

### Stage 7 — Preview

Generates HTML with SVG pitch visualization, beat grid, and confidence-colored dots.

## Artifacts

```
./tmp/
  {stem}.mp3                    ← normalized mono MP3
  {stem}_vocals.mp3             ← separated vocals stem
  {stem}_accompaniment.mp3      ← separated instrumental stem
  {stem}_transcribe.json        ← words, timestamps, MIDI, pitch frames, pauses
  {stem}_whisperx_passes.json   ← per-pass word and character alignments
  whisperx_pitch.json           ← aligned words and pitch data for preview

./output/
  <artist> - <title>/           ← per-song folder (sanitized)
    <artist> - <title>.txt      ← Ultrastar song file
    <artist> - <title>.mp3      ← source audio
    <artist> - <title>.zip      ← complete bundle (incl. intermediates/ unless --no-intermediates)
    <artist> - <title>.html     ← HTML preview
    <artist> - <title>_editor.html  ← self-contained note editor
    <artist> - <title>_vocals.mp3        ← vocals stem (optional)
    <artist> - <title>_accompaniment.mp3 ← instrumental stem (optional)

./web_jobs/                     ← web UI job dirs (one per submitted song)
  <job_id>/job.json             ← manifest (id, title, status, created_at)
  <job_id>/upload/original.*    ← uploaded audio/video
  <job_id>/tmp/                 ← per-job intermediate files
  <job_id>/output/              ← same package output as the CLI (per-song subfolder)
```

## Ultrastar .txt format

```
#TITLE:Song Title
#ARTIST:Artist Name
#MP3:song.mp3
#BPM:120
#GAP:1200
: 0 4 60 Hel-
: 4 4 62 lo
- 16
: 20 4 60 World
E
```

- `:` normal note · `*` golden note · `-` line break · `E` end of song
- Note: `[type] [start_beat] [duration_beats] [midi_pitch] [syllable]`
- `GAP` = milliseconds before beat 0; beat formula: `((ms - GAP) / 1000) * (BPM / 60) * 4`

## Testing

```bash
pytest cli/tests/ -v
```

Tests cover WhisperX result conversion, character-aware alignment, consensus,
configuration, BPM detection, pitch-frame alignment, generation, and Ultrastar parsing.

## Tech stack

| Component | Technology |
|---|---|
| CLI | argparse (stdlib) |
| Web UI | Dash (optional, `ultrasongs-cli[web]`) |
| Package | setuptools, pyproject.toml |
| Audio separation | Demucs, torchaudio |
| Pitch detection | torchcrepe |
| Transcription | faster-whisper or WhisperX |
| Forced alignment | WhisperX, wav2vec2 |
| BPM detection | librosa |
| Syllabification | pyphen |
| Audio I/O | FFmpeg (subprocess), soundfile, lameenc |
| Numeric | numpy |

## Credits

Built by [Pablo Pramparo](https://github.com/pablopramparo).

Powered by:

- [Demucs](https://github.com/facebookresearch/demucs) — vocal separation (Meta Research)
- [WhisperX](https://github.com/m-bain/whisperX) — speech transcription and forced alignment
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — standalone speech transcription backend
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — pitch estimation

## License

MIT
