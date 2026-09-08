"""Shared pipeline orchestration used by both the CLI and the web service."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from cli.config import Config
from cli.logging_setup import get_logger
from cli.pipeline_types import TranscribeResult
from cli.ultrastar import UltrastarMeta, extract_lyrics_from_ultrastar, parse_ultrastar_txt

logger = get_logger("cli.pipeline")

STAGES = ("extract", "transcribe", "align", "generate", "all")


@dataclass(frozen=True)
class ProcessRequest:
    title: str
    artist: str
    lyrics_text: str
    input_path: Path
    video_path: Path | None
    config: Config
    output_dir: Path
    stage: str = "all"
    resume_path: Path | None = None
    include_intermediates: bool = True


@dataclass(frozen=True)
class ProcessResult:
    ok: bool
    error: str | None = None
    txt_path: Path | None = None
    zip_path: Path | None = None
    html_path: Path | None = None
    editor_path: Path | None = None
    editor_json_path: Path | None = None
    output_dir: Path | None = None
    temp_dir: Path | None = None


def looks_like_ultrastar(text: str) -> bool:
    """Return True if the text looks like an Ultrastar .txt file (header line first)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("#")
    return False


def prepare_lyrics_input(text: str) -> tuple[str, UltrastarMeta | None]:
    """Split a lyrics input into (plain lyrics, meta).

    If the text is an Ultrastar .txt, returns the extracted plain lyrics and
    the parsed meta (title/artist/mp3 tags); otherwise the text unchanged and
    None.
    """
    if looks_like_ultrastar(text):
        lyrics = extract_lyrics_from_ultrastar(text)
        meta, _ = parse_ultrastar_txt(text)
        return lyrics, meta
    return text, None


_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(title: str) -> str:
    """Make a title safe for use in an output filename (both CLI and web)."""
    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip(" .")
    return cleaned or "untitled"


def sanitize_output_name(artist: str, title: str) -> str:
    """Sanitized ``<artist> - <title>`` base name for the output folder and files.

    Missing artist or title are dropped; if both are missing, falls back to
    ``untitled``.
    """
    parts = [p for p in ((artist or "").strip(), (title or "").strip()) if p]
    if not parts:
        return "untitled"
    return sanitize_filename(" - ".join(parts))


def run_process(req: ProcessRequest) -> ProcessResult:
    """Run the full or partial pipeline. Failures are returned, not raised."""
    config = req.config
    run_started = time.time()
    try:
        return _run_process(req, run_started)
    except Exception as exc:
        logger.exception("Processing failed")
        return ProcessResult(ok=False, error=str(exc))


def _run_process(req: ProcessRequest, run_started: float) -> ProcessResult:
    from cli.ffmpeg_extract import extract_audio, is_video_path
    from cli.transcribe import transcribe
    from cli.bpm_detect import detect_bpm
    from cli.align import align_lyrics
    from cli.generate import generate_ultrastar
    from cli.package import collect_intermediates, package_output
    from cli.html_preview import generate_preview
    from cli.editor import generate_editor

    title = req.title
    artist = req.artist
    input_path = Path(req.input_path)

    if not input_path.exists():
        return ProcessResult(ok=False, error=f"Input audio file not found: {input_path}")

    # A video passed as the input file is also the source video for the
    # package output and the #VIDEO tag, unless an explicit video is given.
    video_path = Path(req.video_path) if req.video_path else (input_path if is_video_path(input_path) else None)

    config = req.config
    config.temp_path.mkdir(parents=True, exist_ok=True)

    audio_out = config.temp_path / f"{input_path.stem}.mp3"
    stage = req.stage

    result: TranscribeResult | None = None

    # ── Resume from saved TranscribeResult ──
    if req.resume_path:
        resume_path = Path(req.resume_path)
        logger.info(f"Resuming from {resume_path}")
        if not resume_path.exists():
            return ProcessResult(ok=False, error=f"Resume file not found: {resume_path}")
        result = TranscribeResult.from_dict(json.loads(resume_path.read_text(encoding="utf-8")))
        logger.info(f"Loaded {len(result.words)} words from resume file")

    # Stage: extract
    if not result and stage in ("extract", "transcribe", "align", "generate", "all"):
        logger.info("Step 1/5: Extracting audio…")
        extract_audio(input_path, audio_out, config)
        logger.info("Step 1/5: Audio extracted")

    # Stage: transcribe
    if not result and stage in ("transcribe", "align", "generate", "all"):
        logger.info("Step 2/5: Transcribing…")
        result = transcribe(audio_out, req.lyrics_text, config)
        logger.info("Step 2/5: Transcription complete")

        # Always persist TranscribeResult for later resume
        transcribe_json = config.temp_path / f"{input_path.stem}_transcribe.json"
        transcribe_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"TranscribeResult saved to {transcribe_json}")

    if not result:
        return ProcessResult(
            ok=True,
            output_dir=Path(req.output_dir),
            temp_dir=config.temp_path,
        )

    # Stage: align
    aligned = None
    bpm_result = None
    if stage in ("align", "generate", "all"):
        if result.bpm_result is None:
            logger.info("Step 3/5: Detecting BPM…")
            bpm_input = Path(result.accompaniment_path) if config.bpm_use_accompaniment else audio_out
            result.bpm_result = detect_bpm(bpm_input, config)
        bpm_result = result.bpm_result
        logger.info(
            f"Step 3/5: BPM {bpm_result.bpm:.5f} "
            f"(first beat at {bpm_result.first_beat_ms:.0f} ms, "
            f"stable={bpm_result.stable})"
        )

        logger.info("Step 4/5: Aligning lyrics…")
        aligned = align_lyrics(
            req.lyrics_text,
            result.words,
            result.language,
            result.pauses,
            config,
            pitch_frames=result.pitch_frames,
            audio_path=Path(result.vocals_path),
        )
        logger.info("Step 4/5: Alignment complete")

    # Stage: generate
    if stage in ("generate", "all"):
        logger.info("Step 5/5: Generating Ultrastar file…")
        safe_name = sanitize_output_name(artist, title)
        txt_content = generate_ultrastar(
            aligned_syllables=aligned,
            bpm=bpm_result.bpm,
            first_beat_ms=bpm_result.first_beat_ms,
            gap_ms=config.gap_lead_in_ms,
            title=title,
            artist=artist,
            mp3_filename=f"{safe_name}.mp3",
            video_filename=f"{safe_name}{video_path.suffix.lower()}" if video_path else None,
            vocals_filename=f"{safe_name}_vocals.mp3" if Path(result.vocals_path).exists() else None,
            instrumental_filename=f"{safe_name}_accompaniment.mp3" if Path(result.accompaniment_path).exists() else None,
            config=config,
        )
        logger.info("Step 5/5: Ultrastar file generated")

        # All outputs of this run go into a per-song folder.
        run_dir = Path(req.output_dir) / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)
        txt_path = run_dir / f"{safe_name}.txt"
        txt_path.write_text(txt_content, encoding="utf-8")

        # The note editor is written before packaging so the ZIP includes it.
        pitch_json = config.temp_path / "whisperx_pitch.json"
        editor_path = run_dir / f"{safe_name}_editor.html"
        editor_json_path = run_dir / f"{safe_name}_editor.json"
        generate_editor(
            txt_path,
            output_html=editor_path,
            output_json=editor_json_path,
            pitch_json_path=pitch_json if pitch_json.exists() else None,
            vocals_hint=f"{safe_name}_vocals.mp3" if Path(result.vocals_path).exists() else None,
        )
        logger.info(f"Note editor written to {editor_path} with data at {editor_json_path}")

        extra_files = (
            collect_intermediates(config.temp_path, since_ts=run_started)
            if req.include_intermediates
            else []
        )
        package_output(
            txt_content=txt_content,
            mp3_path=audio_out,
            output_dir=run_dir,
            name=safe_name,
            video_path=video_path,
            vocals_path=Path(result.vocals_path),
            accompaniment_path=Path(result.accompaniment_path),
            extra_files=extra_files,
        )
        logger.info(f"Output packaged to {run_dir}")

        html_path = run_dir / f"{safe_name}.html"
        if pitch_json.exists():
            generate_preview(txt_path, output_html=html_path, pitch_json_path=pitch_json)
        else:
            generate_preview(txt_path, output_html=html_path)
        logger.info("HTML preview generated")

        return ProcessResult(
            ok=True,
            txt_path=txt_path,
            zip_path=run_dir / f"{safe_name}.zip",
            html_path=html_path,
            editor_path=editor_path,
            editor_json_path=editor_json_path,
            output_dir=run_dir,
            temp_dir=config.temp_path,
        )

    return ProcessResult(
        ok=True,
        output_dir=Path(req.output_dir),
        temp_dir=config.temp_path,
    )
