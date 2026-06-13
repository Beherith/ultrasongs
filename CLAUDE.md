# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## What we're building

A Next.js web app that generates Ultrastar Deluxe compatible song files (.txt) from an uploaded audio/video file + song lyrics. Runs fully local — no external AI APIs required.

## Commands

```bash
pnpm dev          # Next.js dev server only (port 3000)
pnpm dev:all      # Next.js + Python service together (recommended)
pnpm build        # Production build
pnpm lint         # ESLint
```

**Python service (manual start):**
```bash
cd python
python -m uvicorn transcribe_service:app --port 8001 --reload
```

**Python dependencies:**
```bash
pip install faster-whisper torchcrepe torch torchaudio demucs lameenc soundfile fastapi uvicorn python-multipart
```

`pnpm dev:all` uses [scripts/ensure-python.mjs](scripts/ensure-python.mjs) — it checks if port 8001 is already open before spawning `uvicorn`, so it's safe to run repeatedly.

## Environment variables

```
# .env.local
PYTHON_SERVICE_URL=http://localhost:8001
TMP_DIR=./tmp
```

## Tech stack

- Framework: Next.js 16 + App Router, TypeScript strict mode
- Styling: Tailwind CSS v4
- File upload: Next.js native `request.formData()` (no formidable)
- Audio extraction: FFmpeg via `fluent-ffmpeg` + `@ffmpeg-installer/ffmpeg`
- Vocal separation: Demucs `htdemucs` model (Python microservice)
- Transcription: `faster-whisper` — runs on Demucs vocals track for better accuracy
- Pitch detection: `torchcrepe` — runs on same vocals track
- BPM detection: `music-tempo` npm package
- Syllabification: `hyphen` npm package (TeX hyphenation patterns, 20+ languages)
- ZIP packaging: `jszip`
- Package manager: pnpm

**`next.config.ts` marks native modules as server-external** (`fluent-ffmpeg`, `@ffmpeg-installer/ffmpeg`, `hyphen`, `music-tempo`) — keep this list updated when adding native/CJS-only packages.

## Pipeline architecture

```
Upload (/api/upload)
  → FFmpeg: any format → mono 128kbps mp3 → saved to TMP_DIR

Transcribe (/api/transcribe)
  → POST to Python service with mp3_path
  → Python streams SSE events back through Next.js to the browser:
      1. Demucs: separates vocals + accompaniment (saves _vocals.mp3, _accompaniment.mp3)
      2. torchcrepe: pitch analysis on full vocals track (times/freqs/confs arrays)
      3. Pause detection: RMS energy silence regions from vocals
      4. Whisper: transcribes _vocals.wav, uses lyrics as initial_prompt
      5. Per-word MIDI: median confident crepe frequency in that word's time range
  → Returns: { words, language, vocalsPath, accompanimentPath, pauses }

Generate (/api/generate) — two modes:
  Mode A (auto): alignLyrics() → BPM detection → msToBeats → buildUltrastarTxt
  Mode B (editor): uses EditedNotes from TimelineEditor already in beat format

Align (/api/align) — used only when opening TimelineEditor:
  alignLyrics() → BPM detection → produces { notes: EditorNote[], bpm, gap }
```

## Key files

| File | Purpose |
|------|---------|
| [app/lib/align.ts](app/lib/align.ts) | Fuzzy Levenshtein match of Whisper words → user lyrics; interpolates missing timestamps |
| [app/lib/ultrastar.ts](app/lib/ultrastar.ts) | `msToBeats`, `buildUltrastarTxt`, `detectBpm` |
| [app/lib/syllabify.ts](app/lib/syllabify.ts) | `splitWord` via `hyphen` package; falls back to whole word |
| [app/lib/pitch.ts](app/lib/pitch.ts) | `hzToMidi`, `midiToNoteName` |
| [app/lib/ffmpeg.ts](app/lib/ffmpeg.ts) | `extractAudio`, `isSupportedFile`, `isVideoFile` |
| [app/lib/editorNote.ts](app/lib/editorNote.ts) | `EditorNote` interface shared between align API and TimelineEditor |
| [app/components/TimelineEditor.tsx](app/components/TimelineEditor.tsx) | Phase 2 visual editor — drag notes, adjust timing, export |
| [python/transcribe_service.py](python/transcribe_service.py) | FastAPI: Demucs → crepe → Whisper → SSE stream |

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

- `:` = normal note, `*` = golden note, `-` = line break
- Note format: `[type] [start_beat] [duration] [pitch] [syllable]`
- GAP = milliseconds before beat 0; BPM = song BPM
- Pitch = MIDI (60 = middle C); beat formula: `((ms - GAP) / 1000) * (BPM / 60) * 4`

## Key technical decisions

- **Demucs-first pipeline**: vocals are separated before both Whisper and crepe, improving accuracy on mixed tracks vs. running on the full mix.
- **SSE proxy**: `/api/transcribe` proxies the Python SSE stream directly to the browser. Next.js returns `new NextResponse(serviceRes.body, { "Content-Type": "text/event-stream" })`. The Python service sends keepalive comments every 25 s to prevent HTTP timeouts during long GPU jobs.
- **align.ts is monotone**: Levenshtein matching only moves forward through Whisper output (`searchStart` never resets), so repeated choruses the user omitted don't confuse the matcher.
- **Lyrics as Whisper prompt**: the raw lyrics string is passed as `initial_prompt` to Whisper, biasing transcription toward known words.
- **Two generate modes**: Mode A (auto) runs the full align pipeline; Mode B (editor export) bypasses alignment and uses notes already fine-tuned in the timeline.
- **`/tmp` cleanup**: raw uploads are deleted immediately after FFmpeg conversion; Demucs WAV temp file is deleted after Whisper; mp3s persist until the user's session ends.

## Code style

- API routes are thin — all logic in `/lib`
- Async/await, no `.then()` chains
- All error responses: `{ message: string }`
- `hyphen` modules are `require()`'d at module load time (CJS); listed in `serverExternalPackages`

## Local hardware

- GPU: NVIDIA GTX 1080 (8 GB VRAM)
- Whisper model: `medium` (auto compute type). Fallback if VRAM is tight: `large-v3` with `int8_float16` via `WHISPER_MODEL=large-v3` env var.
- Processing time: ~40–60 s for a 4-min song (Demucs + crepe + Whisper sequential)
