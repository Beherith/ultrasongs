# Ultrasongs CLI

Pure Python CLI tool to generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from audio/video + lyrics. No webserver, no UI — command-line processing with stdout logging.

## Pipeline

```
Input media → FFmpeg extract → Demucs separation → multi-pass faster-whisper → lyric alignment + pause chunks → multi-pass WhisperX exact timing → torchcrepe pitch → Ultrastar .txt + ZIP
```

## Installation

```bash
# Ensure FFmpeg is installed and in PATH
ffmpeg -version

# Install Python dependencies
pip install -r cli/requirements.txt
```

**Required Python packages:** `whisperx`, `demucs`, `torchcrepe`, `torchaudio`, `soundfile`, `numpy`, `lameenc`, `librosa`, `pyphen`.

**Optional (GPU):** CUDA 12.8 with the matching PyTorch 2.8 packages for faster
Demucs/WhisperX inference. The exact install command is documented at the top
of `cli/requirements.txt`.

WhisperX downloads its language-specific wav2vec2 alignment model on first
use. Once the model files are cached, processing remains fully local.

## Usage

```bash
python -m cli <command> [options]
```

### Global flags

| Flag | Description |
|---|---|
| `-c, --config <path>` | Path to `config.jsonc` (default: `cli/config.jsonc`) |
| `-v, --verbose` | DEBUG logging |
| `-q, --quiet` | WARNING and above only |

### `process` — Full pipeline

Generate an Ultrastar song from audio/video + lyrics file.

```bash
python -m cli process \
  --mp3 song.mp3 \
  --lyrics lyrics.txt \
  --title "Song Title" \
  --artist "Artist Name" \
  [--video song.mp4] \
  [--output ./my-output] \
  [--stage all] \
  [--resume intermediate.json]
```

`--lyrics` also accepts an existing Ultrastar `.txt` file: the plain lyrics are extracted from it, and `--mp3`/`--title`/`--artist` (optional in that case) fall back to the file's `#MP3` (resolved relative to the .txt file's directory), `#TITLE`, and `#ARTIST` tags. `#BPM` and `#GAP` are ignored; command line arguments always win.

```bash
python -m cli process --lyrics output/song.txt
python -m cli process --lyrics output/song.txt --title "New Title"
```

**Stages** (`--stage`): Run a partial pipeline. Each stage includes all prior stages.

| Stage | What it does |
|---|---|
| `extract` | Normalize input to mono 128kbps MP3 |
| `transcribe` | Demucs + multi-pass faster-whisper + lyric chunking + WhisperX timing + torchcrepe pitch |
| `align` | BPM detection + Smith-Waterman lyric alignment |
| `generate` | Build `.txt` + package output ZIP |
| `all` | Everything (default) |

**Resume** (`--resume`): Skip earlier stages by loading intermediate JSON results.

**Output:** Creates a directory containing `<title>.txt`, `<title>.mp3`, optional video/vocals/accompaniment stems, and a `<title>.zip` bundle.

### `import` — Re-package existing song

Parse an existing Ultrastar `.txt` + MP3 and re-package into the output directory.

```bash
python -m cli import \
  --txt "Existing Song.txt" \
  --mp3 "Existing Song.mp3" \
  [--output ./my-output]
```

### `diff` — Compare two .txt files

Compare a reference Ultrastar file against a generated one. Exits 0 if within tolerances, 1 if not.

```bash
python -m cli diff \
  --original reference.txt \
  --generated generated.txt
```

**Tolerances:** BPM ±2, beat offset ±4, duration ±4, pitch ±3 semitones.

### `lyrics` — Extract plain lyrics from an Ultrastar file

Reassembles note syllables into words and lines (unvoiced `~` notes are skipped, `-` notes become line breaks). Prints to stdout by default.

```bash
python -m cli lyrics \
  --txt "Existing Song.txt" \
  [--output lyrics.txt]
```

A standalone wrapper is available at the repo root: `python extract_lyrics.py song.txt [-o lyrics.txt]`.

## Configuration

Edit `cli/config.jsonc` (supports `//` and `/* */` comments):

| Key | Default | Description |
|---|---|---|
| `transcription_backend` | `"faster-whisper"` | Hybrid faster-whisper/WhisperX path; `"whisperx"` retains the legacy ASR path |
| `whisper_model` | `"medium"` | Initial ASR model |
| `whisper_language` | `"en"` | Language hint; empty enables detection |
| `faster_whisper_compute_type` | `"auto"` | CTranslate2 compute type for standalone faster-whisper |
| `whisperx_batch_size` | `8` | Number of ASR chunks processed per inference batch |
| `whisperx_compute_type` | `"default"` | CTranslate2 compute type (`default`, `float16`, `float32`, `int8`) |
| `whisperx_align_model` | `""` | Optional wav2vec2 alignment model override |
| `whisperx_interpolate_method` | `"nearest"` | Missing-character timing policy (`nearest`, `linear`, `ignore`) |
| `whisperx_chunk_pause_ms` | `1000` | Split lyric/audio chunks at pauses strictly longer than this |
| `whisperx_align_runs` | `3` | Positive odd number of WhisperX exact-timing passes |
| `transcribe_runs` | `3` | Demucs + faster-whisper passes consolidated before exact alignment |
| `demucs_model` | `"htdemucs"` | Demucs model name |
| `sample_rate` | `44100` | Audio sample rate |
| `pitch_min_hz` | `65.41` | Pitch floor (C2) |
| `pitch_max_hz` | `1046.5` | Pitch ceiling (C6) |
| `crepe_hop_ms` | `10` | torchcrepe hop length |
| `pause_min_silence_ms` | `400` | Minimum silence for pause detection |
| `pause_threshold_pct` | `5` | RMS energy threshold (% of 95th percentile) |
| `gap_lead_in_ms` | `500` | Milliseconds before first note for `#GAP` |
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
| `note_fallback_confidence` | `0.5` | Fallback confidence when no frame meets `note_min_confidence` |
| `note_dropout_gap_ms` | `50` | Max gap (ms) between active frames in one continuous note |
| `note_smooth_window` | `5` | Median filter window (frames) for pitch smoothing |
| `note_pitch_tolerance` | `1` | Max semitone drift kept within a single note |
| `note_min_duration_ms` | `60` | Min duration (ms) for a pitch-change segment |
| `note_frame_step_ms` | `10` | Fallback frame spacing (ms) for note end times |
| `ffmpeg_audio_bitrate` | `"128k"` | Output MP3 bitrate |
| `output_dir` | `"./output"` | Output directory |
| `temp_dir` | `"./tmp"` | Intermediate files directory |
| `debug_alignment` | `true` | Write alignment debug JSON to temp dir |
| `bpm_use_accompaniment` | `true` | Use Demucs instrumental track for BPM detection |

## Supported input formats

**Video:** `.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`
**Audio:** `.mp3`, `.ogg`, `.flac`, `.wav`, `.m4a`

## Testing

```bash
pytest cli/tests/ -v
```
