# Python CLI Port Plan

Port the Ultrasongs TypeScript/JavaScript pipeline to a pure Python CLI tool.
No webserver, no UI -- just command-line song processing with stdout logging.

---

## Phase 0: Project skeleton

- [x] Create `cli/` directory at repo root
- [x] `cli/pyproject.toml` -- project metadata, dependencies, entry point
- [x] `cli/__init__.py`
- [x] `cli/__main__.py` -- `argparse`-based CLI with subcommands
- [x] `cli/config.jsonc` -- single unified settings file (see Phase 1)
- [x] `cli/__main__.py` loads `config.jsonc` on startup (manual JSONC strip + `json.load`)

## Phase 1: Configuration

- [x] Design `config.jsonc` schema covering all pipeline knobs:
  - `whisper_model` (default: "medium")
  - `demucs_model` (default: "htdemucs")
  - `sample_rate` (default: 44100)
  - `pitch_min_hz`, `pitch_max_hz` (C2=65.41, C6=1046.5)
  - `crepe_hop_ms` (default: 10)
  - `pause_min_silence_ms` (default: 400)
  - `pause_threshold_pct` (default: 5)
  - `gap_lead_in_ms` (default: 500)
  - `linebreak_beat_offset` (default: 4)
  - `ffmpeg_audio_bitrate` (default: "128k")
  - `output_dir` (default: "./output")
  - `temp_dir` (default: "./tmp")
- [x] `cli/config.py` -- load, validate, expose as a frozen dataclass
- [x] CLI `--config` flag to override default path

## Phase 2: Logging

- [x] `cli/logging_setup.py` -- configure Python `logging` module
  - stdout handler, `INFO` level by default
  - `--verbose` / `-v` flag for `DEBUG`
  - `--quiet` / `-q` flag for `WARNING`
  - Format: `[YYYY-MM-DD HH:MM:SS] [module] message`
- [x] Every module gets its own `logger = logging.getLogger(__name__)`
- [x] Progress logging pattern: `logger.info("Step N/X: <description>")` at each pipeline stage

## Phase 3: FFmpeg audio extraction (replaces `app/lib/ffmpeg.ts` + upload route)

- [x] `cli/ffmpeg_extract.py`
  - `extract_audio(input_path: Path, output_path: Path, config: Config) -> Path`
  - Subprocess call to `ffmpeg`: `-i <in> -vn -ac 1 -b:a 128k -codec:a libmp3lame <out>`
  - Validate input extension against supported sets (video: mp4/mkv/webm/mov/avi, audio: mp3/ogg/flac/wav/m4a)
  - Fail early: raise on unsupported extension, missing ffmpeg, FFmpeg exit code != 0
- [x] `cli/ffmpeg_pcm.py`
  - `extract_pcm(mp3_path: Path, sample_rate: int) -> bytes`
  - `ffmpeg -i <mp3> -vn -ac 1 -ar <sr> -acodec pcm_f32le -f f32le -`
  - Returns raw float32 PCM bytes for BPM detection

## Phase 4: BPM detection (replaces `app/lib/ultrastar.ts::detectBpm`)

- [x] `cli/bpm_detect.py`
  - `detect_bpm(mp3_path: Path, config: Config) -> float`
  - Replace `music-tempo` (JS library) with `librosa.beat.track()` or `librosa.feature.tempo()`
  - Load audio via `librosa.load(mp3_path, sr=22050, mono=True)`
  - Call `librosa.feature.tempo(y=audio, sr=22050)` -> take median BPM estimate
  - Fallback to 120.0 on any error
  - Return single float BPM value

## Phase 5: Transcription service integration (existing Python, refactor for CLI)

The existing `python/transcribe_service.py` already contains the heavy pipeline. Refactor it into importable modules rather than a FastAPI service.

- [x] `cli/transcribe.py` -- extract logic from `python/transcribe_service.py`
  - `transcribe(mp3_path: Path, lyrics_prompt: str | None, config: Config) -> TranscribeResult`
  - Returns dataclass:
    ```python
    @dataclass
    class TranscribeResult:
        words: list[WordTimestamp]   # word, start, end, midi, pitch_frames
        language: str
        vocals_path: Path
        accompaniment_path: Path
        pauses: list[Pause]          # start, end
    ```
  - Steps (same as current service):
    1. Load audio with `soundfile.read()`
    2. Demucs separation -> vocals + accompaniment at 44100 Hz
    3. Save stems as MP3 via `lameenc`
    4. torchcrepe pitch on vocals at 16000 Hz, 10ms hop
    5. RMS pause detection on vocals
    6. faster-whisper transcription with word timestamps
    7. Attach pitch frames to words
  - Progress logging at each sub-step
  - Fail early: raise on model load failure, empty transcription, etc.
- [x] Remove FastAPI/uvicorn/SSE dependencies from the transcribe module
- [x] Keep `python/whisperx_worker.py` as-is (optional WhisperX path, subprocess call)

## Phase 6: Syllabification (replaces `app/lib/syllabify.ts`)

- [x] `cli/syllabify.py`
  - Replace `hyphen` npm package with `pyphen` (Python equivalent)
  - `split_word(word: str, lang: str) -> list[str]`
  - `syllabify_line(line: str, lang: str) -> list[list[str]]`
  - Language alias mapping: Whisper ISO codes -> pyphen locale codes (e.g., `en` -> `en_US`, `pt-br` -> `pt_BR`, `de` -> `de_DE`)
  - Words <= 2 chars: return as single syllable
  - Unsupported languages: return whole word

## Phase 7: Lyric alignment (replaces `app/lib/align.ts` -- 704 lines, core logic)

This is the largest port. The Smith-Waterman algorithm with phonetic scoring.

- [x] `cli/align.py`
  - `align_lyrics(lyrics: str, words: list[WordTimestamp], pauses: list[Pause], language: str) -> list[AlignedSyllable]`
  - Port these sub-steps verbatim from TS:
    1. **Character normalization**: lowercase, NFD decompose, strip diacritics
    2. **Phonetic scoring**: `phonetic_score(c1: str, c2: str) -> float`
       - Exact match: 1.0
       - Phonetic groups (e.g., [a,e,i], [s,z,c], [t,d]): 0.6 - 0.1 * within-group distance
       - Cross-group pairs: 0.5
       - Mismatch: -0.3
    3. **Smith-Waterman alignment**: three matrices M, X, Y with affine gaps
       - MATCH_SCORE=4, GAP_OPEN=4, GAP_EXTEND=0.5
       - Backtrack to produce alignment path
    4. **Word-level match extraction**: map lyric words to Whisper word indices
    5. **Timestamp computation**: min/max timestamps for matched words, median MIDI from pitch frames
    6. **Interpolation for unmatched words**: linear interpolation before/between/after anchors
    7. **Syllabification**: split each word into syllables, divide duration evenly
    8. **Line break insertion**: mark syllables at lyric line boundaries
  - Return `list[AlignedSyllable]` dataclass:
    ```python
    @dataclass
    class AlignedSyllable:
        syllable: str
        start: float      # seconds
        end: float        # seconds
        midi: int         # MIDI note
        is_line_break: bool = False
    ```
  - Extensive logging at each sub-step for debugging
  - Optional: write debug JSON to temp dir (controlled by config)

## Phase 8: Ultrastar format (replaces `app/lib/ultrastar.ts`)

- [x] `cli/ultrastar.py`
  - Dataclasses:
    ```python
    @dataclass
    class UltrastarNote:
        note_type: str       # ":", "*", "-"
        start_beat: int
        duration: int        # 0 for line breaks
        pitch: int           # 0 for line breaks
        syllable: str        # "" for line breaks

    @dataclass
    class UltrastarMeta:
        title: str
        artist: str
        mp3: str             # filename within output package
        bpm: float
        gap: int             # milliseconds
        video: str | None = None
    ```
  - `ms_to_beats(ms: float, bpm: float, gap: int) -> int`
    - `round(((ms - gap) / 1000) * (bpm / 60) * 4)`
  - `build_ultrastar_txt(notes: list[UltrastarNote], meta: UltrastarMeta) -> str`
    - Header lines (#TITLE, #ARTIST, #MP3, #VIDEO?, #BPM, #GAP)
    - Body lines (: / * / - per note)
    - Trailing `E`
  - `parse_ultrastar_txt(content: str) -> tuple[UltrastarMeta, list[UltrastarNote]]`
    - For import and comparison testing
    - Handle comma-to-dot BPM normalization

## Phase 9: Note generation (replaces `app/api/generate/route.ts`)

- [x] `cli/generate.py`
  - `generate_ultrastar(
      aligned_syllables: list[AlignedSyllable],
      bpm: float,
      gap_ms: int,
      title: str,
      artist: str,
      mp3_filename: str,
      video_filename: str | None = None,
  ) -> str`
  - Convert seconds to beats via `ms_to_beats()`
  - Overlap prevention: `adj_start = max(start_beat, prev_end + 1)`
  - Line break handling:
    - Cap last note of paragraph before line break
    - Insert line break `linebreak_beat_offset` beats before next note
  - Return the full `.txt` string
- [x] `cli/package.py`
  - `package_output(txt_content: str, mp3_path: Path, output_dir: Path, title: str,
                     video_path: Path | None, vocals_path: Path | None,
                     accompaniment_path: Path | None) -> Path`
  - Create output directory
  - Write `<title>.txt`
  - Copy MP3, optional video, vocals, accompaniment
  - Optionally create ZIP (use `zipfile` stdlib)
  - Return path to output directory or ZIP

## Phase 10: CLI subcommands

- [x] `cli/__main__.py` -- argparse with subcommands:

  ```
  ultrasongs process   --mp3 <file> --lyrics <file> --title <t> --artist <a> [--video <file>] [--output <dir>]
  ultrasongs import    --txt <file> --mp3 <file> [--output <dir>]
  ultrasongs diff      --original <txt> --generated <txt>
  ```

  - `process`: full pipeline (extract -> transcribe -> align -> generate -> package)
    - Accepts `--stage` flag to run partial pipeline: `extract`, `transcribe`, `align`, `generate`, `all`
    - Accepts `--resume <json>` to load intermediate results (skip earlier stages)
  - `import`: parse an existing Ultrastar `.txt` + MP3, re-package into output directory
  - `diff`: compare two Ultrastar `.txt` files (see Phase 11)

## Phase 11: End-to-end testing and comparison

- [x] `cli/diff.py`
  - `diff_ultrastar(original_path: Path, generated_path: Path) -> DiffReport`
  - Parse both files via `parse_ultrastar_txt()`
  - Compare:
    - Meta fields (title, artist, BPM, GAP) -- exact match or tolerance
    - Note count
    - Per-note: syllable match, beat offset (< 4 beats), duration diff, pitch diff (< 3 semitones)
    - Line break positions
  - `DiffReport` dataclass with pass/fail summary + per-note deltas
  - Print human-readable report to stdout
  - Exit code 0 if within tolerances, 1 if not
- [x] `tests/` directory with pytest tests:
  - [x] `test_ffmpeg_extract.py` -- smoke test on a known MP3
  - [x] `test_bpm_detect.py` -- verify BPM on known track (tolerance +/- 5 BPM)
  - [x] `test_syllabify.py` -- unit tests for known words in multiple languages
  - [x] `test_align.py` -- unit tests with synthetic Whisper words + lyrics
  - [x] `test_ultrastar_parse.py` -- parse known `.txt` files, verify structure
  - [x] `test_ultrastar_build.py` -- build `.txt`, re-parse, verify round-trip
  - [x] `test_generate.py` -- verify overlap prevention, line break placement
  - [x] `test_diff.py` -- compare two known `.txt` files
  - [x] `test_e2e.py` -- full pipeline on a small test MP3 + lyrics, then `diff` against a reference `.txt`
- [x] `tests/fixtures/` directory:
  - [x] Small test MP3 file (< 30 seconds)
  - [x] Corresponding lyrics file
  - [x] Reference Ultrastar `.txt` file for comparison

## Phase 12: Cleanup and polish

- [x] Remove old `python/transcribe_service.py` FastAPI wrapper (keep refactored modules)
- [x] Update `python/requirements.txt` or move to `cli/pyproject.toml` dependencies
- [x] Add `cli/` to `.gitignore` only for `__pycache__`, `.pytest_cache`
- [x] Verify `config.jsonc` has sensible defaults for a first run
- [x] Ensure all failures produce clear error messages with `sys.exit(1)`
- [x] Add `--help` to all subcommands with usage examples
- [x] Final smoke test: run `python -m cli process` on a real song end-to-end

---

## Dependencies (new Python packages beyond existing `requirements.txt`)

| Package | Replaces | Purpose |
|---------|----------|---------|
| `pyphen` | `hyphen` (npm) | Syllabification |
| `librosa` | `music-tempo` (npm) | BPM detection |
| `python-json-config` or manual JSONC strip | -- | Config file parsing |

Existing packages already in `python/requirements.txt`:
`faster-whisper`, `demucs`, `librosa`, `lameenc`, `torchcrepe`, `torchaudio`,
`soundfile`, `numpy`, `fastapi`, `uvicorn`, `python-multipart`

Note: `fastapi`, `uvicorn`, `python-multipart` can be dropped if the HTTP service is removed entirely.
