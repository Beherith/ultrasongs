# Ultrasongs CLI

Pure Python CLI tool to generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from audio/video + lyrics. No webserver, no UI — command-line processing with stdout logging.

## Pipeline

```
Input media → FFmpeg extract → Demucs separation → torchcrepe pitch → Whisper transcription → Lyric alignment → Ultrastar .txt + ZIP
```

## Installation

```bash
# Ensure FFmpeg is installed and in PATH
ffmpeg -version

# Install Python dependencies
pip install -r cli/requirements.txt
```

**Required Python packages:** `faster-whisper`, `demucs`, `torchcrepe`, `torchaudio`, `soundfile`, `numpy`, `lameenc`, `librosa`, `pyphen`.

**Optional (GPU):** CUDA-enabled PyTorch for faster Demucs/Whisper inference.

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

**Stages** (`--stage`): Run a partial pipeline. Each stage includes all prior stages.

| Stage | What it does |
|---|---|
| `extract` | Normalize input to mono 128kbps MP3 |
| `transcribe` | Demucs separation + torchcrepe pitch + Whisper transcription |
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

## Configuration

Edit `cli/config.jsonc` (supports `//` and `/* */` comments):

| Key | Default | Description |
|---|---|---|
| `whisper_model` | `"medium"` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `demucs_model` | `"htdemucs"` | Demucs model name |
| `sample_rate` | `44100` | Audio sample rate |
| `pitch_min_hz` | `65.41` | Pitch floor (C2) |
| `pitch_max_hz` | `1046.5` | Pitch ceiling (C6) |
| `crepe_hop_ms` | `10` | torchcrepe hop length |
| `pause_min_silence_ms` | `400` | Minimum silence for pause detection |
| `pause_threshold_pct` | `5` | RMS energy threshold (% of 95th percentile) |
| `gap_lead_in_ms` | `500` | Milliseconds before first note for `#GAP` |
| `linebreak_beat_offset` | `4` | Beats before next note for line breaks |
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
