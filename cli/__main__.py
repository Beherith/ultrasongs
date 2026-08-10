"""CLI entry point with argparse subcommands."""

import argparse
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
    proc.add_argument("--mp3", required=True, help="Input audio/video file")
    proc.add_argument("--lyrics", required=True, help="Lyrics text file")
    proc.add_argument("--title", required=True, help="Song title")
    proc.add_argument("--artist", required=True, help="Artist name")
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
    from cli.logging_setup import get_logger

    logger = get_logger("cli.process")
    logger.info("Step 1/5: Extracting audio…")

    mp3_path = Path(args.mp3)
    lyrics_path = Path(args.lyrics)
    lyrics_text = lyrics_path.read_text(encoding="utf-8")

    # Ensure temp directory exists
    config.temp_path.mkdir(parents=True, exist_ok=True)

    # Stage: extract
    if args.stage in ("extract", "transcribe", "align", "generate", "all"):
        audio_out = config.temp_path / f"{mp3_path.stem}.mp3"
        extract_audio(mp3_path, audio_out, config)
        logger.info("Step 1/5: Audio extracted")

    # Stage: transcribe
    if args.stage in ("transcribe", "align", "generate", "all"):
        logger.info("Step 2/5: Transcribing…")
        result = transcribe(audio_out, lyrics_text, config)
        logger.info("Step 2/5: Transcription complete")

    # Stage: align
    if args.stage in ("align", "generate", "all"):
        logger.info("Step 3/5: Detecting BPM…")
        bpm_input = Path(result.accompaniment_path) if config.bpm_use_accompaniment else audio_out
        bpm = detect_bpm(bpm_input, config)
        logger.info(f"Step 3/5: BPM detected: {bpm}")

        logger.info("Step 4/5: Aligning lyrics…")
        aligned = align_lyrics(lyrics_text, result.words, result.language, result.pauses, config)
        logger.info("Step 4/5: Alignment complete")

    # Stage: generate
    if args.stage in ("generate", "all"):
        logger.info("Step 5/5: Generating Ultrastar file…")
        txt_content = generate_ultrastar(
            aligned_syllables=aligned,
            bpm=bpm,
            gap_ms=config.gap_lead_in_ms,
            title=args.title,
            artist=args.artist,
            mp3_filename=f"{args.title}.mp3",
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
            title=args.title,
            video_path=Path(args.video) if args.video else None,
            vocals_path=Path(result.vocals_path),
            accompaniment_path=Path(result.accompaniment_path),
        )
        logger.info(f"Output packaged to {output_dir}")

        from cli.html_preview import generate_preview

        txt_path = output_dir / f"{args.title}.txt"
        pitch_json = config.temp_path / "whisper_pitch.json"
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


if __name__ == "__main__":
    sys.exit(main())
