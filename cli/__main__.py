"""CLI entry point with argparse subcommands."""

import argparse
import json
import sys
from pathlib import Path

from cli.config import load_config
from cli.logging_setup import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultrasongs",
        description="Generate Ultrastar Deluxe song files from audio/video + lyrics",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config.jsonc file (default: cli/config.jsonc)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Only show WARNING and above",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── process ──────────────────────────────────────────────────────────────
    proc = subparsers.add_parser("process", help="Full pipeline: extract, transcribe, align, generate")
    proc.add_argument(
        "--mp3",
        default=None,
        help="Input audio/video file (default: #MP3 tag of the lyrics file, if it is an Ultrastar .txt)",
    )
    proc.add_argument(
        "--lyrics",
        required=True,
        help="Lyrics text file (plain text, or an Ultrastar .txt file whose lyrics will be extracted)",
    )
    proc.add_argument(
        "--title",
        default=None,
        help="Song title (default: #TITLE tag of the lyrics file, if it is an Ultrastar .txt)",
    )
    proc.add_argument(
        "--artist",
        default=None,
        help="Artist name (default: #ARTIST tag of the lyrics file, if it is an Ultrastar .txt)",
    )
    proc.add_argument("--video", default=None, help="Optional video file")
    proc.add_argument("--output", default=None, help="Output directory (overrides config)")
    proc.add_argument(
        "--stage",
        choices=["extract", "transcribe", "align", "generate", "all"],
        default="all",
        help="Run only up to this stage (default: all)",
    )
    proc.add_argument(
        "--resume",
        default=None,
        help="Load intermediate results from JSON file (skips earlier stages)",
    )

    # ── import ───────────────────────────────────────────────────────────────
    imp = subparsers.add_parser("import", help="Import existing Ultrastar .txt + MP3")
    imp.add_argument("--txt", required=True, help="Ultrastar .txt file")
    imp.add_argument("--mp3", required=True, help="MP3 file")
    imp.add_argument("--output", default=None, help="Output directory")

    # ── diff ─────────────────────────────────────────────────────────────────
    diff = subparsers.add_parser("diff", help="Compare two Ultrastar .txt files")
    diff.add_argument("--original", required=True, help="Original .txt file")
    diff.add_argument("--generated", required=True, help="Generated .txt file")

    # ── preview ──────────────────────────────────────────────────────────────
    prev = subparsers.add_parser("preview", help="Generate HTML preview from Ultrastar .txt")
    prev.add_argument("--txt", required=True, help="Ultrastar .txt file")
    prev.add_argument("--output", default=None, help="Output HTML file (default: <title>.html)")
    prev.add_argument("--pitch", default=None, help="Pitch detection JSON to overlay")

    # ── lyrics ───────────────────────────────────────────────────────────────
    lyr = subparsers.add_parser("lyrics", help="Extract plain lyrics from an Ultrastar .txt file")
    lyr.add_argument("--txt", required=True, help="Ultrastar .txt file")
    lyr.add_argument("--output", default=None, help="Output lyrics file (default: print to stdout)")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging before loading config
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Load configuration
    config = load_config(args.config)

    if args.command == "process":
        return _cmd_process(args, config)
    elif args.command == "import":
        return _cmd_import(args, config)
    elif args.command == "diff":
        return _cmd_diff(args, config)
    elif args.command == "preview":
        return _cmd_preview(args, config)
    elif args.command == "lyrics":
        return _cmd_lyrics(args)
    else:
        parser.print_help()
        return 1


def _cmd_process(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Execute the full or partial pipeline."""
    from cli.ffmpeg_extract import extract_audio
    from cli.transcribe import transcribe
    from cli.bpm_detect import detect_bpm
    from cli.align import align_lyrics
    from cli.generate import generate_ultrastar
    from cli.package import package_output
    from cli.pipeline_types import TranscribeResult
    from cli.logging_setup import get_logger

    logger = get_logger("cli.process")

    lyrics_path = Path(args.lyrics)
    lyrics_input = lyrics_path.read_text(encoding="utf-8")

    title = args.title
    artist = args.artist
    mp3_arg = args.mp3
    if _looks_like_ultrastar(lyrics_input):
        from cli.ultrastar import extract_lyrics_from_ultrastar, parse_ultrastar_txt

        lyrics_text = extract_lyrics_from_ultrastar(lyrics_input)
        meta, _ = parse_ultrastar_txt(lyrics_input)
        if title is None:
            title = meta.title
        if artist is None:
            artist = meta.artist
        if mp3_arg is None:
            mp3_arg = meta.mp3
        logger.info(f"Lyrics input {lyrics_path.name} is an Ultrastar file; extracted plain lyrics")
    else:
        lyrics_text = lyrics_input

    if not title or not artist or not mp3_arg:
        missing = [name for name, value in (("--title", title), ("--artist", artist), ("--mp3", mp3_arg)) if not value]
        logger.error(
            f"Missing {', '.join(missing)}: provide them as command line arguments "
            f"or use an Ultrastar .txt lyrics file with matching tags"
        )
        return 1

    if args.mp3 is None:
        mp3_path = lyrics_path.parent / mp3_arg
    else:
        mp3_path = Path(mp3_arg)
    if not mp3_path.exists():
        logger.error(f"Input audio file not found: {mp3_path}")
        return 1

    # Ensure temp directory exists
    config.temp_path.mkdir(parents=True, exist_ok=True)

    audio_out = config.temp_path / f"{mp3_path.stem}.mp3"
    resume_path = Path(args.resume) if args.resume else config.temp_path / f"{mp3_path.stem}_transcribe.json"

    result: TranscribeResult | None = None

    # ── Resume from saved TranscribeResult ──
    if args.resume:
        logger.info(f"Resuming from {resume_path}")
        if not resume_path.exists():
            logger.error(f"Resume file not found: {resume_path}")
            return 1
        result = TranscribeResult.from_dict(json.loads(resume_path.read_text(encoding="utf-8")))
        logger.info(f"Loaded {len(result.words)} words from resume file")

    # Stage: extract
    if not result and args.stage in ("extract", "transcribe", "align", "generate", "all"):
        logger.info("Step 1/5: Extracting audio…")
        extract_audio(mp3_path, audio_out, config)
        logger.info("Step 1/5: Audio extracted")

    # Stage: transcribe
    if not result and args.stage in ("transcribe", "align", "generate", "all"):
        logger.info("Step 2/5: Transcribing…")
        result = transcribe(audio_out, lyrics_text, config)
        logger.info("Step 2/5: Transcription complete")

        # Always persist TranscribeResult for later resume
        transcribe_json = config.temp_path / f"{mp3_path.stem}_transcribe.json"
        transcribe_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"TranscribeResult saved to {transcribe_json}")

    if not result:
        return 0

    # Stage: align
    if args.stage in ("align", "generate", "all"):
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
            lyrics_text,
            result.words,
            result.language,
            result.pauses,
            config,
            pitch_frames=result.pitch_frames,
            audio_path=Path(result.vocals_path),
        )
        logger.info("Step 4/5: Alignment complete")

    # Stage: generate
    if args.stage in ("generate", "all"):
        logger.info("Step 5/5: Generating Ultrastar file…")
        txt_content = generate_ultrastar(
            aligned_syllables=aligned,
            bpm=bpm_result.bpm,
            first_beat_ms=bpm_result.first_beat_ms,
            gap_ms=config.gap_lead_in_ms,
            title=title,
            artist=artist,
            mp3_filename=f"{title}.mp3",
            video_filename=Path(args.video).name if args.video else None,
            config=config,
        )
        logger.info("Step 5/5: Ultrastar file generated")

        # Package output
        output_dir = Path(args.output) if args.output else config.output_path
        package_output(
            txt_content=txt_content,
            mp3_path=audio_out,
            output_dir=output_dir,
            title=title,
            video_path=Path(args.video) if args.video else None,
            vocals_path=Path(result.vocals_path),
            accompaniment_path=Path(result.accompaniment_path),
        )
        logger.info(f"Output packaged to {output_dir}")

        from cli.html_preview import generate_preview

        txt_path = output_dir / f"{title}.txt"
        pitch_json = config.temp_path / "whisperx_pitch.json"
        if pitch_json.exists():
            generate_preview(txt_path, pitch_json_path=pitch_json)
        else:
            generate_preview(txt_path)
        logger.info("HTML preview generated")

    return 0


def _cmd_import(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Import an existing Ultrastar .txt + MP3."""
    from cli.logging_setup import get_logger
    from cli.package import package_output
    from cli.pipeline_types import UltrastarMeta, UltrastarNote
    from cli.ultrastar import build_ultrastar_txt, parse_ultrastar_txt

    logger = get_logger("cli.import")
    txt_path = Path(args.txt)
    mp3_path = Path(args.mp3)

    meta, notes = parse_ultrastar_txt(txt_path.read_text(encoding="utf-8"))
    txt_content = _rebuild_txt(meta, notes)

    output_dir = Path(args.output) if args.output else config.output_path
    package_output(
        txt_content=txt_content,
        mp3_path=mp3_path,
        output_dir=output_dir,
        title=meta.title,
    )
    logger.info(f"Imported to {output_dir}")
    return 0


def _rebuild_txt(meta: "UltrastarMeta", notes: list["UltrastarNote"]) -> str:  # type: ignore[name-defined]
    """Rebuild a .txt string from parsed data."""
    return build_ultrastar_txt(notes, meta)


def _cmd_diff(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Compare two Ultrastar .txt files."""
    from cli.diff import diff_ultrastar

    report = diff_ultrastar(Path(args.original), Path(args.generated))
    report.print()
    return 0 if report.passed else 1


def _cmd_preview(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Generate an HTML preview from an Ultrastar .txt file."""
    from cli.html_preview import generate_preview

    txt_path = Path(args.txt)
    output_html = Path(args.output) if args.output else None
    pitch_json = args.pitch if args.pitch else None
    generate_preview(txt_path, output_html, pitch_json)
    return 0


def _cmd_lyrics(args: argparse.Namespace) -> int:
    """Extract plain lyrics from an Ultrastar .txt file."""
    from cli.logging_setup import get_logger
    from cli.ultrastar import extract_lyrics_from_ultrastar

    logger = get_logger("cli.lyrics")
    content = Path(args.txt).read_text(encoding="utf-8")
    lyrics = extract_lyrics_from_ultrastar(content)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(lyrics, encoding="utf-8")
        logger.info(f"Lyrics written to {output_path}")
    else:
        sys.stdout.write(lyrics)
    return 0


def _looks_like_ultrastar(text: str) -> bool:
    """Return True if the text looks like an Ultrastar .txt file (header line first)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("#")
    return False


if __name__ == "__main__":
    sys.exit(main())
