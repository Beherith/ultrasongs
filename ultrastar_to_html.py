#!/usr/bin/env python3
"""Compatibility command for rendering an UltraStar song as HTML."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from ultrasongs.domain.reporting import write_pipeline_report  # noqa: E402
from ultrasongs.domain.ultrastar import parse_ultrastar_file  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("Usage: python ultrastar_to_html.py <song.txt> [output.html]")
        return 1
    source = Path(arguments[0])
    if not source.exists():
        print(f"Error: {source} not found.")
        return 1
    song = parse_ultrastar_file(source)
    output = Path(arguments[1]) if len(arguments) > 1 else Path(source.stem + ".html")
    write_pipeline_report(output, candidate=song, title=song.metadata.title or source.stem)
    print(f"Written: {output}  ({len(song.notes)} notes, {len(tuple(song.verses()))} verses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
