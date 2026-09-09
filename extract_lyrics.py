#!/usr/bin/env python3
"""Standalone script: extract plain lyrics from an Ultrastar .txt file.

Usage:
    python extract_lyrics.py ./output/tit31.txt
    python extract_lyrics.py ./output/tit31.txt -o lyrics.txt
"""

import argparse
import sys
from pathlib import Path

from cli.ultrastar import extract_lyrics_from_ultrastar


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="extract_lyrics",
        description="Extract plain lyrics from an Ultrastar .txt file",
    )
    parser.add_argument("txt", help="Ultrastar .txt file")
    parser.add_argument("-o", "--output", default=None, help="Write lyrics to file (default: stdout)")
    args = parser.parse_args()

    content = Path(args.txt).read_text(encoding="utf-8")
    lyrics = extract_lyrics_from_ultrastar(content)

    if args.output:
        Path(args.output).write_text(lyrics, encoding="utf-8")
    else:
        sys.stdout.write(lyrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
