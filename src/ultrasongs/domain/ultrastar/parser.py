"""Canonical parser for UltraStar Deluxe text files."""

from __future__ import annotations

import re
from pathlib import Path

from .models import LineBreak, NoteType, UltrastarMetadata, UltrastarNote, UltrastarSong

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_NOTE_RE = re.compile(
    rf"^\s*([:*F])\s+({_NUMBER})\s+({_NUMBER})\s+([-+]?\d+)[ \t](.*)$",
    re.IGNORECASE,
)
_BREAK_RE = re.compile(rf"^\s*-\s+({_NUMBER})(?:\s+({_NUMBER}))?\s*$")
_KNOWN_HEADERS = {"TITLE", "ARTIST", "MP3", "BPM", "GAP", "VIDEO"}


class UltrastarParseError(ValueError):
    def __init__(self, message: str, line_number: int | None = None) -> None:
        prefix = f"line {line_number}: " if line_number is not None else ""
        super().__init__(prefix + message)
        self.line_number = line_number


def parse_ultrastar_text(content: str, *, strict: bool = False) -> UltrastarSong:
    """Parse UltraStar content.

    UltraStar timings are interpreted as beat units by definition. Unknown
    headers are retained in ``metadata.extras``. In permissive mode malformed
    body lines are ignored, which matches the legacy application behavior.
    """

    headers: dict[str, str] = {}
    events: list[UltrastarNote | LineBreak] = []

    for line_number, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            key, separator, value = stripped[1:].partition(":")
            if not separator:
                if strict:
                    raise UltrastarParseError("metadata header has no colon", line_number)
                continue
            headers[key.strip().upper()] = value.strip()
            continue
        if stripped.upper() == "E":
            break

        note_match = _NOTE_RE.match(line)
        if note_match:
            marker, start, duration, pitch, lyric = note_match.groups()
            marker = marker.upper()
            events.append(
                UltrastarNote(
                    note_type=NoteType(marker),
                    start_beat=float(start),
                    duration_beats=float(duration),
                    pitch=int(pitch),
                    lyric=lyric,
                )
            )
            continue

        break_match = _BREAK_RE.match(line)
        if break_match:
            start, end = break_match.groups()
            events.append(LineBreak(float(start), float(end) if end is not None else None))
            continue

        if strict:
            raise UltrastarParseError(f"unrecognized content: {line!r}", line_number)

    try:
        bpm = _parse_decimal(headers.get("BPM", "120"), "BPM")
        gap = _parse_decimal(headers.get("GAP", "0"), "GAP")
        metadata = UltrastarMetadata(
            title=headers.get("TITLE", ""),
            artist=headers.get("ARTIST", ""),
            mp3=headers.get("MP3", ""),
            video=headers.get("VIDEO"),
            bpm=bpm,
            gap_ms=gap,
            extras={key: value for key, value in headers.items() if key not in _KNOWN_HEADERS},
        )
    except ValueError as exc:
        raise UltrastarParseError(str(exc)) from exc
    return UltrastarSong(metadata=metadata, events=tuple(events))


def parse_ultrastar_file(path: str | Path, *, strict: bool = False) -> UltrastarSong:
    return parse_ultrastar_text(Path(path).read_text(encoding="utf-8-sig"), strict=strict)


def _parse_decimal(value: str, name: str) -> float:
    try:
        return float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
