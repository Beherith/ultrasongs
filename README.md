# Ultrasongs

> **Work in progress** — functional but rough around the edges.

Generate [Ultrastar Deluxe](https://ultrastar-deluxe.org/) compatible `.txt` song files from any audio or video file + song lyrics. Runs fully local — no external AI APIs required.

**Pipeline:** Upload → Demucs vocal separation → torchcrepe pitch detection → faster-whisper transcription → fuzzy lyric alignment → BPM-based beat mapping → `.txt` export with optional visual timeline editor.

**Runs in the browser** — no desktop app to install. Clone the repo, run two commands, open `localhost:3000`.

## Features

- Runs in the browser — no installer, no Electron, no desktop app
- Upload any audio or video format (MP4, MKV, MP3, FLAC, WAV, …)
- Automatic vocal separation via [Demucs](https://github.com/facebookresearch/demucs) `htdemucs`
- Per-word MIDI pitch via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — neural pitch estimation tuned for singing voices
- Transcription via [faster-whisper](https://github.com/guillaumekynast/faster-whisper), biased by your provided lyrics
- Fuzzy Levenshtein alignment of Whisper words → user lyrics
- BPM detection + beat-accurate note placement
- Syllable splitting (20+ languages via TeX hyphenation patterns)
- Visual timeline editor with audio playback, per-note preview, and real-time microphone pitch tracking
- Draft save/load for resuming work later
- ZIP download with `.txt` + separated audio tracks

## Requirements

- Node.js 20+ and [pnpm](https://pnpm.io/)
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (or install via `@ffmpeg-installer/ffmpeg` — already in `package.json`)
- PyTorch with CUDA recommended (CPU works but is slow)

### Python dependencies

```bash
pip install faster-whisper torchcrepe torch torchaudio demucs lameenc soundfile fastapi uvicorn python-multipart
```

## Setup

```bash
# 1. Install Node dependencies
pnpm install

# 2. Configure environment
cp .env.example .env.local
# Edit .env.local if needed (defaults work out of the box)
```

`.env.local` variables:

```
PYTHON_SERVICE_URL=http://localhost:8001  # Python microservice address
TMP_DIR=./tmp                             # temporary processing files
DRAFTS_DIR=./drafts                       # saved draft projects
```

## Running

```bash
pnpm dev:all   # Starts Next.js (port 3000) + Python service (port 8001) together
```

Or start them separately:

```bash
pnpm dev                                                           # Next.js only
cd python && python -m uvicorn transcribe_service:app --port 8001  # Python only
```

`pnpm dev:all` checks if port 8001 is already listening before spawning uvicorn, so it's safe to run repeatedly.

## Architecture

Everything runs locally. Three processes communicate over HTTP:

```
┌──────────────┐     HTTP      ┌──────────────────┐     fetch      ┌──────────────────┐
│   Browser     │ ◄───────────► │  Next.js Server   │ ◄───────────► │  Python Service   │
│  (React UI)   │               │  (Node.js :3000)  │               │  (FastAPI :8001)  │
└──────────────┘               └──────────────────┘               └──────────────────┘
                                     │                                    │
                               ┌─────┴─────┤                          ┌───┴────┐
                               │           │                          │        │
                          ./tmp/      ./drafts/                   GPU/     CPU/
                          (temp)      (saved)                     VRAM     RAM
```

- **Browser** — file upload, UI, timeline editor, audio/video playback
- **Next.js Server** — API routes, FFmpeg audio extraction, lyric alignment, BPM detection, ZIP packaging
- **Python Service** — GPU-accelerated Demucs, torchcrepe, and Whisper models

## Pipeline

### Step 1 — Upload

**Route:** `POST /api/upload`
**Runs in:** Next.js (Node.js)

Accepts an audio file and an optional video file. Validates the file extension against supported formats (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`, `.mp3`, `.ogg`, `.flac`, `.wav`, `.m4a`), writes the raw file to disk, then converts it to a normalized MP3.

**How it works:**

1. The raw file is saved as `{uuid}-raw{ext}` in `TMP_DIR`.
2. FFmpeg (`libmp3lame` codec) extracts audio: strips the video track, downmixes to mono, encodes at 128 kbps constant bitrate. Output is `{uuid}.mp3`.
3. If the uploaded file is a video format, a copy of the raw file is saved as `{uuid}-video{ext}` for later use as an Ultrastar background video.
4. If a separate video file was uploaded explicitly, it overrides any video derived from the audio file.
5. The raw upload file (`{uuid}-raw{ext}`) is deleted.

| Artifact | Location | Description |
|----------|----------|-------------|
| `{uuid}.mp3` | `TMP_DIR/` | Normalized mono 128 kbps MP3 used for all downstream steps |
| `{uuid}-video{ext}` | `TMP_DIR/` | Video file (if uploaded or extracted from a video-format audio source) |

### Step 2 — Transcribe

**Route:** `POST /api/transcribe` (proxied to Python via SSE)
**Runs in:** Python service (`python/transcribe_service.py`)

The Python service receives the MP3 path and runs four sub-steps sequentially, streaming progress events back through SSE. All heavy computation runs in a background `asyncio` task; the generator loop reads from a queue with a 25-second timeout, emitting SSE keepalive comments (`: keepalive\n\n`) to prevent the Node.js HTTP layer from timing out.

#### 2a — Vocal separation (Demucs)

**How it works:**

1. The `htdemucs` model is loaded on-demand (not at startup, to leave VRAM for Whisper).
2. Audio is resampled to 44100 Hz, duplicated to stereo (required by Demucs), and shaped to `(1, 2, samples)`.
3. `apply_model()` runs inference, producing 4 stems: drums, bass, other, vocals.
4. The vocals stem (index 3) is averaged to mono. The remaining 3 stems are summed and averaged to produce the accompaniment.
5. The model is deleted, `gc.collect()` and `torch.cuda.empty_cache()` are called to free VRAM for subsequent steps.
6. Both stems are encoded to 128 kbps mono MP3 via `lameenc`, and the vocals is also saved as a lossless WAV for Whisper.

#### 2b — Pitch analysis (torchcrepe)

**How it works:**

1. Vocals audio is resampled to 16000 Hz.
2. `torchcrepe.predict()` runs the `full` model once on the entire track with a 10 ms hop (160 samples at 16 kHz), frequency range C2–C6 (65.41–1046.5 Hz), and Viterbi decoding.
3. Returns three parallel arrays: `times` (seconds), `pitch` (Hz), `periodicity` (confidence 0–1).
4. These arrays are kept in memory for the transcription step, where each Whisper word's time window is queried to compute its MIDI pitch.

#### 2c — Pause detection (RMS energy)

**How it works:**

1. A sliding window computes RMS energy across the vocals track: 25 ms frames, 10 ms hop.
2. The 95th-percentile RMS value is used as the energy reference. Frames below 5% of this threshold are marked silent.
3. Consecutive silent frames ≥400 ms are recorded as pause regions `{start, end}` in seconds.
4. These pauses are passed to the alignment step as contextual hints.

#### 2d — Transcription (faster-whisper or WhisperX)

**How it works:**

1. `faster-whisper` transcribes the vocals WAV with `word_timestamps=True`. The user's lyrics are passed as `initial_prompt` to bias the model toward the expected text.
2. Alternatively, when `ALIGN_ENGINE=whisperx`, a separate `whisperx_worker.py` subprocess is invoked for word-level timestamp alignment.
3. For each transcribed word, the pitch arrays from step 2b are queried:
   - **Word MIDI**: the median frequency of pitch frames within the word's time window, converted via `round(12 * log2(hz / 440) + 69)`. Confidence thresholds are tried in descending order (0.5 → 0.3 → 0.1) to find enough data points. Falls back to MIDI 60 (C4) if no confident data.
   - **Pitch frames**: all qualifying frames (confidence > 0.1, frequency > 0) are attached as `{time, midi, confidence}` objects for finer-grained syllable-level pitch lookup later.

| Artifact | Location | Description |
|----------|----------|-------------|
| `{base}_vocals.mp3` | `TMP_DIR/` | Separated vocals track (128 kbps mono MP3) |
| `{base}_accompaniment.mp3` | `TMP_DIR/` | Separated instrumental track (128 kbps mono MP3) |
| `{base}_transcribe.json` | `TMP_DIR/` | Transcription result: words with timestamps + MIDI + pitch frames, detected language, pause regions, and paths to stems |
| `{base}_vocals.wav` | `TMP_DIR/` | Temporary lossless WAV for Whisper (deleted after transcription) |

The JSON result is returned to the browser via the SSE stream.

### Step 3 — Align

**Route:** `POST /api/align`
**Runs in:** Next.js (Node.js)

Aligns user-provided lyrics to Whisper word timestamps. This is the most complex step, combining phonetic fuzzy matching, temporal clustering, gap interpolation, syllable splitting, and BPM detection.

#### 3a — Text normalization

Both lyric words and Whisper words are normalized identically:

1. Lowercased.
2. Unicode NFD decomposition applied, then combining diacritical marks stripped (`\u0300-\u036f` range removed).
3. Non-alphanumeric characters removed.

This allows matching across accent differences (e.g., `café` → `cafe`) and punctuation variations.

#### 3b — Phonetic Levenshtein distance

Standard Levenshtein is extended with phonetic-aware substitution costs:

- **Articulation groups**: Characters are grouped by phonetic class (front vowels `a/e/i`, sibilants `s/z/c`, alveolar stops `t/d`, bilabial stops `p/b`, etc.). Substitutions within the same group cost 0.3–0.45 depending on position within the group.
- **Cross-group confusions**: Known ASR confusion pairs (`s`↔`sh`, `f`↔`ph`, `w`↔`u`, `r`↔`l`) cost 0.4.
- **Unrelated characters**: Cost 1.0.

The DP matrix uses these costs for the substitution diagonal, while insertion and deletion remain at cost 1.0. The final distance is normalized by the maximum word length, producing a similarity score from 0 (identical) to ~1 (completely different).

#### 3c — Word scoring

Beyond the phonetic Levenshtein, two heuristics handle common Whisper-vs-lyrics mismatches:

- **Suffix match**: If the normalized lyric word (≥3 chars) is a suffix of the Whisper word, score = 0.03. Handles Whisper adding articles or prefixes (e.g., lyric `love` vs Whisper `in love`).
- **Prefix match**: If the normalized lyric word (≥3 chars) is a prefix of the Whisper word, score = 0.06. Handles Whisper truncating or merging words.
- **Max tolerable score** depends on lyric word length: ≤2 chars → 0.05, 3 chars → 0.35, 4–5 chars → 0.55, 6+ chars → 0.55. Shorter words tolerate less mismatch.

#### 3d — Line matching

For each lyric line, words are matched greedily against Whisper words:

1. For each lyric word, the algorithm scans forward through Whisper words starting from the previous match position.
2. The search window extends 90 seconds from the last match (or from t=0 for the first match).
3. Each candidate Whisper word is scored as: `textScore + timeJumpPenalty`. The time penalty is `max(0, whisperStart - lastMatchTime - 20) × 0.015`, penalizing large temporal jumps beyond a 20-second grace period.
4. The best-scoring candidate (lowest combined score) is selected, provided its text score is within the word-length-dependent threshold.
5. Search position advances past the match; unmatched words are recorded as `null`.

#### 3e — Cluster resolution

When Whisper repeats words or matches span verse boundaries, a single lyric line may produce multiple disjoint match clusters. Resolution:

1. Matched words are grouped into clusters. A new cluster starts when the time gap between consecutive matches exceeds 4 seconds.
2. If multiple clusters exist, only the largest (most matches) is kept. All other matches are discarded.
3. The search position and last-match time are updated to the end of the kept cluster.

#### 3f — Line validation

To prevent false-positive alignments, lines with ≥4 words must pass two checks:

1. At least one significant match (matched word >2 chars after normalization).
2. The time span from first to last match does not exceed `max(12s, wordCount × 3s)`.

Lines failing either check are fully rejected (all words set to `null`), and the search position/last-match-time revert to pre-line values.

#### 3g — Gap interpolation

Unmatched words (`null` entries) receive interpolated timestamps from surrounding anchor points:

- **Before first anchor**: Uniformly spaced backward from the first match's start time, assuming 1 second per missing word. MIDI copies the first anchor's pitch.
- **Between anchors**: Linear interpolation of both time and MIDI pitch. If anchors are at indices `a` and `b`, word `k` between them gets `time = tA + (k/gap) × (tB - tA)` and `midi = round(midiA + (k/gap) × (midiB - midiA))`.
- **After last anchor**: Fallback slots of 300 ms each, starting from the last anchor's end time. MIDI copies the last anchor's pitch.

#### 3h — Syllabification

Each matched word is split into syllables using the `hyphen` library, which provides TeX hyphenation patterns for 20+ languages:

1. The Whisper-detected language code is mapped to a hyphenation module (e.g., `pt-br` → `pt`, `sr-latn` → `en` fallback).
2. Words ≤2 characters are never split.
3. The `hyphenateSync()` function inserts soft-hyphen characters at allowable break points; splitting on these produces syllable parts.
4. The word's total duration (`end - start`) is divided equally among syllables.
5. For each syllable's time sub-window, the median MIDI pitch is extracted from the word's pitch frames, trying confidence thresholds 0.5 → 0.3 → 0.1. Falls back to the word-level MIDI if no confident frames exist.

#### 3i — Line break markers

Between consecutive lyric lines, a line-break marker is inserted. Its timestamp is set to the start time of the first word of the next line (or the last syllable's end if no next words exist).

#### 3j — BPM detection

Runs in parallel with the alignment via `Promise.all`:

1. FFmpeg extracts the MP3 as float32 mono PCM at 44100 Hz.
2. `music-tempo` performs onset detection on the PCM samples to find beat positions.
3. BPM is derived from the median interval between detected beats. Falls back to 120 BPM on any failure.

#### 3k — GAP computation

The `GAP` value (milliseconds before beat 0) is computed as `max(0, firstSyllableStart × 1000 - 500)`. This gives a 500 ms lead-in before the first sung syllable, matching Ultrastar's convention of starting the beat grid slightly before the first note.

| Artifact | Location | Description |
|----------|----------|-------------|
| `{base}_alignment.json` | `TMP_DIR/` | Editor-ready notes array (syllable, startSec, durationSec, pitch, type), BPM, GAP, and track duration |
| `{base}_align_debug.json` | `TMP_DIR/` | Debug data: per-word match details — lyric word, normalized forms, matched Whisper word, timestamps, MIDI, and match confidence |

Returns `notes` (editor-ready syllable objects), `bpm`, `gap`, and `duration`.

### Step 4 — Generate (or Edit)

**Route:** `POST /api/generate`
**Runs in:** Next.js (Node.js)

Two modes:

#### Mode A — Auto (from alignment)

1. **BPM detection**: If not already cached from the align step, runs `detectBpm()` (FFmpeg PCM extraction → `music-tempo` onset detection, fallback 120 BPM).
2. **GAP computation**: `max(0, firstSyllableStart × 1000 - 500)` — 500 ms lead-in before the first note.
3. **Beat conversion**: Each syllable's start/end time is converted to Ultrastar beats via `round(((ms - gap) / 1000) × (bpm / 60) × 4)`. Duration is `endBeat - startBeat`, clamped to ≥1.
4. **Overlap prevention**: Each note's start beat is adjusted to `max(startBeat, prevEnd + 1)`, ensuring notes never overlap.
5. **Line break handling**: For line-break markers, the previous paragraph's last note is capped so it doesn't extend into the gap. The line break is placed at `max(prevEnd + 1, nextNoteBeat - 4)` — 4 beats before the next note, giving Ultrastar time for display transitions.
6. **TXT construction**: Header (`#TITLE`, `#ARTIST`, `#MP3`, optional `#VIDEO`, `#BPM`, `#GAP`) followed by note lines (`: start duration pitch syllable` or `- startBreak`), terminated by `E`.

#### Mode B — Editor (manual notes)

1. BPM and GAP come from the editor (already computed during alignment).
2. Each editor note's start/duration (in seconds) is converted to beats via `msToBeats()`.
3. Overlap prevention applies the same `max(startBeat, prevEnd + 1)` adjustment.
4. TXT construction follows the same format as Mode A.

#### ZIP packaging

Both modes use JSZip to assemble the download:

1. `{title}.txt` — the Ultrastar song file.
2. `{mp3Filename}` — the original normalized MP3.
3. Optional: video file, `vocals.mp3`, `accompaniment.mp3`.
4. DEFLATE compression, streamed to the browser as `application/zip` with `Content-Disposition: attachment`.

| Artifact | Location | Description |
|----------|----------|-------------|
| `{base}_generate_result.json` | `TMP_DIR/` | Generation result: title, artist, BPM, GAP, filename references, beat-mapped notes, and the full `.txt` content |
| `{title}.zip` | streamed to browser | Contains: `{title}.txt`, `{mp3}`, optional video, `vocals.mp3`, `accompaniment.mp3` |

### Step 5 — Save Draft (optional)

**Route:** `POST /api/drafts`, `GET /api/drafts`, `PATCH /api/drafts/{id}`, `DELETE /api/drafts/{id}`
**Runs in:** Next.js (Node.js)

Persists a complete processing state so work can be resumed later. Copies all relevant files into a dedicated draft directory.

| Artifact | Location | Description |
|----------|----------|-------------|
| `draft.json` | `DRAFTS_DIR/{uuid}/` | Full draft metadata: title, artist, lyrics, BPM, GAP, notes, words, language, pauses, and file references |
| `audio.mp3` | `DRAFTS_DIR/{uuid}/` | Copy of the source MP3 |
| `video{ext}` | `DRAFTS_DIR/{uuid}/` | Copy of the video file (if present) |
| `vocals.mp3` | `DRAFTS_DIR/{uuid}/` | Copy of the separated vocals (if present) |
| `accompaniment.mp3` | `DRAFTS_DIR/{uuid}/` | Copy of the separated accompaniment (if present) |

### Step 6 — Import (optional)

**Route:** `POST /api/import`
**Runs in:** Next.js (Node.js)

Imports an existing Ultrastar Deluxe `.txt` file along with its audio and optional video. Parses the `.txt` header and note data, converts beat-based notes back to second-based editor notes, and creates a draft entry.

| Artifact | Location | Description |
|----------|----------|-------------|
| `draft.json` | `DRAFTS_DIR/{uuid}/` | Imported draft metadata |
| `audio.mp3` | `DRAFTS_DIR/{uuid}/` | Uploaded audio file |
| `video{ext}` | `DRAFTS_DIR/{uuid}/` | Uploaded video file (if provided) |

### Artifact summary

```
./tmp/
  {uuid}.mp3                    ← normalized mono MP3 (Step 1)
  {uuid}-video{ext}             ← video file (Step 1)
  {uuid}_vocals.mp3             ← separated vocals stem (Step 2)
  {uuid}_accompaniment.mp3      ← separated instrumental stem (Step 2)
  {uuid}_transcribe.json        ← words, timestamps, MIDI, pitch frames, pauses (Step 2)
  {uuid}_alignment.json         ← editor notes, BPM, GAP, duration (Step 3)
  {uuid}_align_debug.json       ← per-word match details for debugging (Step 3)
  {uuid}_generate_result.json   ← beat-mapped notes, TXT content (Step 4)

./drafts/
  {uuid}/
    draft.json                  ← saved project state (Step 5)
    audio.mp3                   ← source audio copy
    video{ext}                  ← video copy (optional)
    vocals.mp3                  ← vocals copy (optional)
    accompaniment.mp3           ← accompaniment copy (optional)
```

## Tech stack

| Layer            | Technology                                 |
| ---------------- | ------------------------------------------ |
| Framework        | Next.js 16 + App Router, TypeScript strict |
| Styling          | Tailwind CSS v4                            |
| Audio extraction | FFmpeg via `fluent-ffmpeg`                 |
| Vocal separation | Demucs `htdemucs` (Python)                 |
| Transcription    | faster-whisper (Python)                    |
| Pitch detection  | torchcrepe (Python)                        |
| BPM detection    | `music-tempo`                              |
| Syllabification  | `hyphen` (TeX patterns)                    |
| ZIP export       | JSZip                                      |
| Package manager  | pnpm                                       |

## Performance

Tested on an NVIDIA GTX 1080 (8 GB VRAM). A 4-minute song takes roughly 40–60 s (Demucs + torchcrepe + Whisper running sequentially).

Default Whisper model: `medium`. For tighter VRAM budgets, set `WHISPER_MODEL=large-v3` to use `int8_float16` compute.

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
- `GAP` = milliseconds before beat 0; beat formula: `((ms − GAP) / 1000) × (BPM / 60) × 4`

## Credits

Built by [Pablo Pramparo](https://github.com/pablopramparo).

Powered by:

- [Demucs](https://github.com/facebookresearch/demucs) — vocal separation (Meta Research)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — speech transcription
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe) — pitch estimation

## License

MIT
