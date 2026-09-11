# Ultrasongs

Generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from any audio or video file + song lyrics. Pure Python — runs fully local, no external AI APIs. A command-line pipeline with a web UI, plus a full-featured, self-contained browser note editor for hand-tuning the results.

**Pipeline:** FFmpeg extract → Demucs vocal separation → multi-pass faster-whisper consensus → Smith-Waterman lyric alignment → pause-delimited lyric/vocal chunks → multi-pass WhisperX exact word/character timing → BPM + torchcrepe analysis → `.txt` + ZIP export.

### Note Editor Screenshots

<img width="1897" height="1097" alt="image" src="https://github.com/user-attachments/assets/c0ef578f-0514-427d-91a2-bbb706ba7f45" />


## Web UI
Screenshot:

<img width="904" height="966" alt="image" src="https://github.com/user-attachments/assets/516f5e65-b593-46b6-8592-66c82529ea0e" />


The web UI is the easiest way to run the pipeline: upload media + lyrics in the browser, watch the job run, and download the results (including the note editor).

### Running the server

```bash
pip install -e "cli/[web]"    # adds the optional Dash dependency
ULTRASONGS_WEB_PASSWORD=secret python -m cli web
```

- **Flags:** `--host`, `--port`, `--no-auth`, `--web-config <path>`. Defaults come from `cli/web_config.jsonc` (port 8030).
- **Auth:** password from the `ULTRASONGS_WEB_PASSWORD` environment variable, or from a `ULTRASONGS_WEB_PASSWORD=...` entry in `./.env.local` (gitignored; env var wins). Flask session cookie, constant-time compare. There is no TLS — keep the default localhost bind or put a reverse proxy in front. `--no-auth` disables login but is rejected unless the host is `127.0.0.1`/`localhost`.

### Processing a song

1. **Mode** — pick one:
   - **Full pipeline** (default): builds an Ultrastar `.txt` from lyrics. Needs the media upload, artist, title, and lyrics.
   - **Stems only**: runs just the htdemucs split and returns the vocals and accompaniment MP3s. Only the media upload is used — lyrics, title, artist, and cover fields are ignored (dimmed in the form).
2. **Upload** — drag & drop (or browse) an audio/video file (MP3, MP4, MKV, WEBM, MOV, AVI, FLAC, OGG, M4A, WAV). If the file name looks like `Artist - Title`, the artist and title fields are prefilled for you.
3. **Artist / Title** — required for the full pipeline; edit the prefilled values if needed.
4. **Lyrics** — paste the lyrics, or upload a lyrics `.txt` file. Plain text is the gold standard and should be complete and accurate. An existing Ultrastar `.txt` also works: the plain lyrics are extracted from it, and empty title/artist fields are filled from its `#TITLE`/`#ARTIST` tags.
5. **Cover art** (optional) — upload a JPEG (10 MB cap); it is written as `<name>_cover.jpg` and referenced by a `#COVER` tag in the generated `.txt`.
6. **Settings** (optional) — a collapsible accordion with one control per `config.jsonc` key (40 total), grouped by area. Values override the base config for that job only; **Reset settings to file defaults** restores them.
7. **Process** — submit the job.

### Jobs and downloads

- One job runs at a time (FIFO queue). The job card shows a status pill (`Queued #n` → `Running` → `Succeeded` / `Failed`), elapsed time, and a live log tail.
- On success you get download links:
  - **Download ZIP** — the full bundle: `.txt`, MP3, video (if the input was one), stems, cover, HTML preview, and the note editor (plus the run's `tmp/` artifacts under `intermediates/`).
  - **Open editor** — the self-contained note editor (see below).
  - **Open preview** — the HTML pitch visualization.
  - Individual files (`.txt`, MP3, video, stems, cover) with their sizes.
- Jobs are stored under `web_jobs/<job_id>/` (`upload/`, `tmp/`, `output/`, plus a `job.json` manifest). Jobs older than `job_retention_days` (default 7) are pruned at server startup and before each new submission. Uploads are capped at `max_upload_mb`.

## Note editor

Every processed song ships with a `<name>_editor.html` — a single **self-contained HTML file** (no server, no build step, no CDN). Double-click it in your browser. It renders the song as an SVG note chart (the same visualization as the HTML preview) and lets you hand-tune the generated `.txt`:

- Download it from the web UI (**Open editor**), or
- Generate it for any `.txt` with the CLI:

```bash
python -m cli edit --txt output/song.txt \
  [--pitch tmp/whisperx_pitch.json] \
  [--vocals output/song_vocals.mp3] \
  [--embed-audio] \
  [--output editor.html]
```

`--pitch` embeds the word/pitch overlay; `--vocals` names the suggested audio file; `--embed-audio` base64-embeds it (requires `--vocals`). Without `--output`, the file is written next to the `.txt` as `<stem>_editor.html`.

### What you see

- **Notes** — one bar per syllable, colored by pitch. Blue = normal, yellow = gold note, grey = freestyle note.
- **Whisper words** (green, with start/end arrows) — the words the ASR actually heard in the vocals, with their forced-alignment timing. Use them to check that your notes sit under the syllable that is actually sung.
- **CREPE pitch dots** — one per pitch-detected audio frame; y = detected pitch, color = loudness (green quiet → red loud), opacity = pitch confidence (faint = less trustworthy).
- **FFT background** — a live spectrogram of the loaded vocals (dark = quiet, bright = energy); appears once audio is loaded.
- **Beat grid** — the file's `#BPM` is authoritative. Notes snap to a 16th-note grid (`BPM × 4`); the visual grid lines and the metronome run on quarter notes.

### Getting audio in

The pipeline already points the editor at the song's `_vocals.mp3` (or embeds it with `--embed-audio`). Otherwise drag a vocals audio file onto the drop zone (or click to browse) to enable the FFT background and playback. **Load vocals** in the toolbar swaps the track and recalculates the FFT at any time.

### Editing notes

- **Select** a note by clicking it.
- **Drag** the note body: up/down = pitch, left/right = start. **Drag an edge** = duration. Notes snap to the beat grid.
- **Double-click** a note to edit its lyric; **double-click empty canvas** to insert a `~` rest note at that position.
- Toolbar (with keyboard shortcut):
  - **Split** (`S`) — split the selected note at its midpoint.
  - **Merge** (`M`) — merge the selected note with the next one.
  - **Delete** (`Del`)
  - **Gold** (`G`) — toggle golden note.
  - **Freestyle** (`F`) — toggle freestyle note (sung free, drawn grey, written as `F`).
  - **Lyric** (`L` / double-click)
- **Lines:**
  - **+ Line** / **− Line** — insert a new lyric line after the current one (with a single starter note) / remove the entire current line.
  - **◀ 1 beat** / **1 beat ▶** — shift every note of the current line by one beat (clamped at 0), e.g. to reposition a line after its shape changed.
  - **Copy to repeats** — finds the other lines with identical displayed lyrics (e.g. the other instances of a chorus) and, through a checkbox popover (all pre-checked), overwrites their entire shape (pitches, durations, text, gold/freestyle flags, rests, splits) with the current line's, anchoring each target at its own first note. One **Undo** covers all targets. If an overwritten line now ends after the next line's first note, the status bar warns you to use the 1-beat shift buttons.
- **Undo** (`Ctrl+Z`) — full undo stack for every edit above.

### Playback

- **Play** (`Space`) — plays the current line's notes as MIDI over the vocals. **Play from selected** starts at the selected note.
- Three volume sliders: **Vocals** (audio), **Notes** (MIDI), **Metronome** (quarter-note clicks).

### Loading and saving

- **Load Ultrastar .txt** — import a saved `.txt` (UTF-8, Windows-1252 fallback): replaces the notes, raw header, `#BPM`/`#GAP`, and title/artist. Audio and pitch overlay stay as-is.
- **Load pitch JSON** — replace the overlay with a different `whisperx_pitch.json` (bare `words` array or `{words: [...]}`).
- **Download Ultrastar .txt** (`Ctrl+S`) — browser download of the edited song. Header tags are preserved verbatim, and a `#CREATOR` timestamp is stamped on save. This is the file you give to Ultrastar Deluxe.

## Features

- Upload any audio or video format (MP4, MKV, MP3, FLAC, WAV, …)
- Automatic vocal separation via [Demucs](https://github.com/facebookresearch/demucs) `htdemucs`
- Per-word MIDI pitch via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — neural pitch estimation tuned for singing voices
- Multi-pass [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription for accurate text, followed by WhisperX forced alignment for exact word/character timing
- Smith-Waterman phonetic alignment of transcribed words → user lyrics
- BPM detection via [librosa](https://librosa.org/) + beat-accurate note placement
- Syllable splitting (20+ languages via [pyphen](https://github.com/karpathy/pyphen))
- Self-contained browser **note editor**: drag/edge-drag notes, split/merge/delete, gold + freestyle, lyric editing, line insert/remove/shift, copy-to-repeats, undo, live FFT spectrogram, audio + MIDI + metronome playback
- Re-process from an existing Ultrastar `.txt` — lyrics, title, artist, and audio track are extracted automatically
- Standalone lyrics extraction from Ultrastar files (`lyrics` subcommand or `extract_lyrics.py`)
- Partial pipeline execution and resume from intermediate results
- Built-in `.txt` diff tool with configurable tolerances
- HTML preview with SVG pitch visualization
- ZIP download with `.txt` + separated audio tracks
- Web UI — upload + process in the browser, per-song settings, FIFO job queue, live log tail, one-click downloads

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

## CLI Usage

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

# Generate the self-contained note editor HTML
python -m cli edit --txt output/song.txt [--pitch tmp/whisperx_pitch.json] [--vocals output/song_vocals.mp3] [--embed-audio]

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
| `-o, --override <spec>` | Override config keys (repeatable; takes precedence over `-c`). A JSON object (`'{"whisper_model": "small"}`) or comma-separated `key=value` pairs (`transcribe_runs=5,whisper_model=small`) |
| `-v, --verbose` | DEBUG logging |
| `-q, --quiet` | WARNING and above only |

Partial execution and resume:

```bash
# Run only up to transcription
python -m cli process ... --stage transcribe

# Resume from saved intermediate results
python -m cli process ... --resume tmp/name_transcribe.json --stage align
```

## Configuration

Edit `cli/config.jsonc` (supports `//` and `/* */` comments). 40 configuration keys covering GPU selection, model choices, pitch range, pause detection, BPM, note segmentation, output paths, and more. See the file for detailed per-key documentation.

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
  <job_id>/upload/<name>.<ext>  ← uploaded audio/video (sanitized file name)
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

- `:` normal note · `*` golden note · `F` freestyle note · `-` line break · `E` end of song
- Note: `[type] [start_beat] [duration_beats] [midi_pitch] [syllable]`
- `GAP` = milliseconds before beat 0; beat formula: `((ms - GAP) / 1000) * (BPM / 60) * 4`

## Testing

```bash
pytest cli/tests/ -v
```

On Windows, pass a fresh `--basetemp` per run to avoid `PermissionError` walls from a locked default temp dir (e.g. antivirus scans):

```powershell
pytest cli/tests/ --basetemp "$env:TEMP\pytest-ultrasongs-$([guid]::NewGuid().ToString('N'))"
```

Tests cover WhisperX result conversion, character-aware alignment, consensus, configuration, BPM detection, pitch-frame alignment, generation, Ultrastar parsing, packaging, and the web UI.

## Tech stack

| Component | Technology |
|---|---|
| CLI | argparse (stdlib) |
| Web UI | Dash (optional, `ultrasongs-cli[web]`) |
| Note editor | self-contained HTML/JS (Web Audio, SVG, Canvas) |
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
