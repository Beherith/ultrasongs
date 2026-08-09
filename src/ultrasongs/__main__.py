"""Command-line entry point for the UltraSongs Dash application."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import load_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ultrasongs")
    parser.add_argument("--config", type=Path, help="JSON or TOML settings file")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the validated effective startup configuration and exit",
    )
    commands = parser.add_subparsers(dest="command")
    repair = commands.add_parser(
        "repair",
        help="reprocess an MP3 and compare the updated chart with an existing UltraStar TXT",
    )
    repair.add_argument(
        "--config",
        dest="repair_config",
        type=Path,
        help="JSON or TOML settings file (may also be placed before the command)",
    )
    repair.add_argument("--audio", type=Path, required=True, help="source MP3/audio file")
    repair.add_argument(
        "--song", type=Path, required=True, help="existing UltraStar TXT reference"
    )
    repair.add_argument(
        "--output-dir",
        type=Path,
        help="directory beneath which a unique repair bundle is created",
    )
    repair.add_argument(
        "--lyrics-file",
        type=Path,
        help="optional corrected UTF-8 lyrics instead of reconstructing them from the chart",
    )
    repair.add_argument("--title", help="override the title from the existing chart")
    repair.add_argument("--artist", help="override the artist from the existing chart")
    repair.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="set a UI-safe pipeline option; VALUE accepts JSON (repeatable)",
    )
    repair.add_argument(
        "--json", action="store_true", help="print the final result as JSON"
    )
    repair.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="exit with status 3 when configured similarity thresholds fail",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load settings once, then start the application factory when available."""
    args = _parser().parse_args(argv)
    settings = load_settings(
        config_file=getattr(args, "repair_config", None) or args.config
    )

    if args.print_config:
        print(settings.effective_snapshot().to_json())
        return 0

    if args.command == "repair":
        from .cli import run_repair_workflow

        try:
            overrides = _parse_cli_overrides(args.set)
            if not args.json:
                print(f"Reprocessing {args.audio} against {args.song} ...", flush=True)
            result = run_repair_workflow(
                settings,
                audio_path=args.audio,
                song_path=args.song,
                output_root=args.output_dir,
                lyrics_path=args.lyrics_file,
                title=args.title,
                artist=args.artist,
                ui_overrides=overrides,
            )
        except Exception as exc:
            print(f"UltraSongs repair failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_repair_summary(result)
        if (
            args.fail_on_threshold
            and result.validation_outcome is not None
            and not result.validation_outcome.passed
        ):
            return 3
        return 0

    try:
        from .app import create_app
    except ModuleNotFoundError as exc:
        if exc.name != "ultrasongs.app":
            raise
        print("UltraSongs configuration is valid; the Dash application is not installed yet.")
        return 0

    app = create_app(settings=settings)
    app.run(
        host=settings.server.host,
        port=settings.server.port,
        debug=settings.server.debug,
    )
    return 0


def _parse_cli_overrides(values: Sequence[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in values:
        path, separator, raw_value = item.partition("=")
        if not separator or not path.strip():
            raise ValueError(f"Invalid --set value {item!r}; expected PATH=VALUE")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[path.strip()] = value
    return overrides


def _print_repair_summary(result: object) -> None:
    similarity = result.similarity
    outcome = result.validation_outcome
    status = "PASSED" if outcome is not None and outcome.passed else "FAILED"
    print(f"Repair complete: {result.export_directory}")
    print(f"Updated song: {result.updated_song_path}")
    print(f"Visual comparison: {result.report_path}")
    print(f"Scores: {result.scores_path}")
    print(
        "Similarity: "
        f"{similarity.matched_notes}/{similarity.reference_notes} reference notes matched; "
        f"coverage={similarity.reference_coverage:.3f}; "
        f"timing RMSE={_metric(similarity.timing_rmse_ms, 'ms')}; "
        f"duration RMSE={_metric(similarity.duration_rmse_ms, 'ms')}; "
        f"pitch distance={_metric(similarity.pitch_distance_semitones, 'semitones')}"
    )
    print(f"Configured validation: {status}")
    if outcome is not None:
        for failure in outcome.failures:
            print(f"  - {failure}")


def _metric(value: float | None, unit: str) -> str:
    return "unavailable" if value is None else f"{value:.3f} {unit}"


if __name__ == "__main__":
    raise SystemExit(main())
