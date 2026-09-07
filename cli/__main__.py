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
    proc.add_argument(
        "--no-intermediates",
        action="store_true",
        default=False,
        help="Do not include intermediate temp files in the output ZIP",
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

    # ── edit ─────────────────────────────────────────────────────────────────
    edit = subparsers.add_parser("edit", help="Generate a self-contained HTML note editor from an Ultrastar .txt")
    edit.add_argument("--txt", required=True, help="Ultrastar .txt file to edit")
    edit.add_argument("--pitch", default=None, help="whisperx_pitch.json to overlay (word labels + pitch dots)")
    edit.add_argument("--vocals", default=None, help="Vocals stem (audio source for the FFT background + playback)")
    edit.add_argument(
        "--embed-audio",
        action="store_true",
        default=False,
        help="Base64-embed the --vocals audio into the HTML (truly single-file)",
    )
    edit.add_argument("--output", default=None, help="Output HTML file (default: <txt_stem>_editor.html)")

    # ── web ──────────────────────────────────────────────────────────────────
    web = subparsers.add_parser("web", help="Run the web UI (Dash) for the pipeline")
    web.add_argument("--host", default=None, help="Bind address (default: from web config, 127.0.0.1)")
    web.add_argument("--port", type=int, default=None, help="Bind port (default: from web config, 8080)")
    web.add_argument(
        "--no-auth",
        action="store_true",
        default=False,
        help="Disable password login (only allowed when binding to 127.0.0.1 or localhost)",
    )
    web.add_argument(
        "--web-config",
        default=None,
        help="Path to web_config.jsonc (default: cli/web_config.jsonc)",
    )

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
    elif args.command == "edit":
        return _cmd_edit(args, config)
    elif args.command == "web":
        return _cmd_web(args, config)
    else:
        parser.print_help()
        return 1


def _cmd_process(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Execute the full or partial pipeline (thin adapter over pipeline.run_process)."""
    from cli.logging_setup import get_logger
    from cli.pipeline import ProcessRequest, run_process, prepare_lyrics_input
    from cli.ultrastar import read_text_fallback

    logger = get_logger("cli.process")

    lyrics_path = Path(args.lyrics)
    lyrics_input = read_text_fallback(lyrics_path)

    title = args.title
    artist = args.artist
    mp3_arg = args.mp3
    lyrics_text, meta = prepare_lyrics_input(lyrics_input)
    if meta is not None:
        if title is None:
            title = meta.title
        if artist is None:
            artist = meta.artist
        if mp3_arg is None:
            mp3_arg = meta.mp3
        logger.info(f"Lyrics input {lyrics_path.name} is an Ultrastar file; extracted plain lyrics")

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

    request = ProcessRequest(
        title=title,
        artist=artist,
        lyrics_text=lyrics_text,
        input_path=mp3_path,
        video_path=Path(args.video) if args.video else None,
        config=config,
        output_dir=Path(args.output) if args.output else config.output_path,
        stage=args.stage,
        resume_path=Path(args.resume) if args.resume else None,
        include_intermediates=not args.no_intermediates,
    )
    result = run_process(request)
    if not result.ok:
        logger.error(result.error or "Processing failed")
        return 1
    return 0


def _cmd_import(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Import an existing Ultrastar .txt + MP3."""
    from cli.logging_setup import get_logger
    from cli.package import package_output
    from cli.pipeline_types import UltrastarMeta, UltrastarNote
    from cli.ultrastar import build_ultrastar_txt, parse_ultrastar_txt, read_text_fallback

    logger = get_logger("cli.import")
    txt_path = Path(args.txt)
    mp3_path = Path(args.mp3)

    meta, notes = parse_ultrastar_txt(read_text_fallback(txt_path))
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


def _cmd_web(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Run the Dash web UI."""
    import os

    from cli.logging_setup import get_logger

    logger = get_logger("cli.web")
    try:
        from cli.web.app import load_web_config, run_server
        from cli.web.auth import resolve_auth
    except ImportError:
        logger.error("The web UI requires Dash. Install it with: pip install 'ultrasongs-cli[web]'")
        return 1

    web_cfg = load_web_config(args.web_config)

    host = args.host or web_cfg.host
    port = args.port or web_cfg.port
    password, error = resolve_auth(
        os.environ.get("ULTRASONGS_WEB_PASSWORD"),
        args.no_auth,
        host,
    )
    if error:
        logger.error(error)
        return 1
    return run_server(web_cfg, config, host=host, port=port, password=password)


def _cmd_lyrics(args: argparse.Namespace) -> int:
    """Extract plain lyrics from an Ultrastar .txt file."""
    from cli.logging_setup import get_logger
    from cli.ultrastar import extract_lyrics_from_ultrastar, read_text_fallback

    logger = get_logger("cli.lyrics")
    content = read_text_fallback(Path(args.txt))
    lyrics = extract_lyrics_from_ultrastar(content)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(lyrics, encoding="utf-8")
        logger.info(f"Lyrics written to {output_path}")
    else:
        sys.stdout.write(lyrics)
    return 0


def _cmd_edit(args: argparse.Namespace, config: "Config") -> int:  # type: ignore[name-defined]
    """Generate a self-contained HTML note editor from an Ultrastar .txt file."""
    from cli.editor import generate_editor
    from cli.logging_setup import get_logger

    logger = get_logger("cli.edit")
    try:
        out = generate_editor(
            Path(args.txt),
            Path(args.output) if args.output else None,
            Path(args.pitch) if args.pitch else None,
            Path(args.vocals) if args.vocals else None,
            args.embed_audio,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1
    logger.info(f"Editor written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
