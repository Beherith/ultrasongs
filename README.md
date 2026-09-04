# Ultrasongs

> **Work in progress** — functional but rough around the edges.

Generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from any audio or video file + song lyrics. Pure Python CLI — runs fully local, no external AI APIs, no web server.

**Pipeline:** FFmpeg extract → Demucs vocal separation → multi-pass faster-whisper consensus → Smith-Waterman lyric alignment → pause-delimited lyric/vocal chunks → multi-pass WhisperX exact word/character timing → BPM + torchcrepe analysis → `.txt` + ZIP export.

## Features

- Upload any audio or video format (MP4, MKV, MP3, FLAC, WAV, …)
- Automatic vocal separation via [Demucs](https://github.com/facebookresearch/demucs) `htdemucs`
- Per-word MIDI pitch via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — neural pitch estimation tuned for singing voices
- Multi-pass [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription for accurate text, followed by WhisperX forced alignment for exact word/character timing
- Smith-Waterman phonetic alignment of transcribed words → user lyrics
- BPM detection via [librosa](https://librosa.org/) + beat-accurate note placement
- Syllable splitting (20+ languages via [pyphen](https://github.com/karpathy/pyphen))
- Partial pipeline execution and resume from intermediate results
- Built-in `.txt` diff tool with configurable tolerances
- HTML preview with SVG pitch visualization
- ZIP download with `.txt` + separated audio tracks

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
```

**Optional (GPU):** CUDA 12.8 with PyTorch 2.8 for faster Demucs/WhisperX inference.

```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
# Full pipeline
python -m cli process --mp3 song.mp3 --lyrics lyrics.txt --title "Title" --artist "Artist"

# With video background
python -m cli process --mp3 song.mp4 --lyrics lyrics.txt --title "Title" --artist "Artist" --video song.mp4

# Import existing Ultrastar file
python -m cli import --txt existing.txt --mp3 existing.mp3

# Compare two .txt files
python -m cli diff --original reference.txt --generated output.txt

# Generate HTML preview
python -m cli preview --txt output.txt --pitch tmp/name_pitch.json
```

Global flags:

| Flag | Description |
|---|---|
| `-c, --config <path>` | Custom `config.jsonc` (default: `cli/config.jsonc`) |
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

Edit `cli/config.jsonc` (supports `//` and `/* */` comments). 18 configuration keys covering GPU selection, model choices, pitch range, pause detection, BPM, output paths, and more. See the file for detailed per-key documentation.

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

Writes `.txt`, copies MP3/video/stems, creates ZIP bundle in `output/`.

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
  {title}.txt                   ← Ultrastar song file
  {title}.mp3                   ← source audio
  {title}.zip                   ← complete bundle
  vocals.mp3                    ← vocals stem (optional)
  accompaniment.mp3             ← instrumental stem (optional)
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
