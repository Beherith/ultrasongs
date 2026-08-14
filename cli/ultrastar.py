"""Ultrastar format: parser and builder for .txt song files."""

import re

from cli.pipeline_types import UltrastarMeta, UltrastarNote


_NOTE_RE = re.compile(r"^([:*])\s+(-?\d+)\s+(\d+)\s+(-?\d+)(?: (.*))?$")


def ms_to_beats(ms: float, bpm: float, gap: int) -> int:
    """Convert milliseconds to Ultrastar beats.

    beat = ((ms - GAP) / 1000) * (BPM / 60) * 4
    """
    return round(((ms - gap) / 1000) * (bpm / 60) * 4)


def build_ultrastar_txt(notes: list[UltrastarNote], meta: UltrastarMeta) -> str:
    """Build an Ultrastar .txt string from notes and metadata.

    Format:
        #TITLE:Title
        #ARTIST:Artist
        #MP3:file.mp3
        #VIDEO:file.mp4  (optional)
        #BPM:120.00
        #GAP:500

        : 0 4 60 hello
        * 5 4 62 world
        - 10
        E
    """
    header_lines = [
        f"#TITLE:{meta.title}",
        f"#ARTIST:{meta.artist}",
        f"#MP3:{meta.mp3}",
        f"#BPM:{meta.bpm:.2f}",
        f"#GAP:{round(meta.gap)}",
    ]
    if meta.video:
        header_lines.insert(3, f"#VIDEO:{meta.video}")

    header = "\n".join(header_lines)

    body_lines = []
    for n in notes:
        if n.note_type == "-":
            body_lines.append(f"- {n.start_beat}")
        else:
            body_lines.append(f"{n.note_type} {n.start_beat} {n.duration} {n.pitch} {n.syllable}")

    body = "\n".join(body_lines)
    return f"{header}\n\n{body}\nE\n"


def parse_ultrastar_txt(content: str) -> tuple[UltrastarMeta, list[UltrastarNote]]:
    """Parse an Ultrastar .txt file into structured data.

    Args:
        content: Raw .txt file content.

    Returns:
        Tuple of (UltrastarMeta, list of UltrastarNote).
    """
    lines = content.split("\n")
    meta: dict[str, str] = {}
    notes: list[UltrastarNote] = []

    for line in lines:
        trimmed = line.strip()

        if trimmed.startswith("#"):
            colon = trimmed.index(":") if ":" in trimmed else -1
            if colon > 0:
                key = trimmed[1:colon].upper()
                val = trimmed[colon + 1:].strip()
                meta[key] = val

        elif match := _NOTE_RE.match(trimmed):
            notes.append(UltrastarNote(
                note_type=match.group(1),
                start_beat=int(match.group(2)),
                duration=int(match.group(3)),
                pitch=int(match.group(4)),
                syllable=match.group(5) or "",
            ))

        elif trimmed.startswith("- "):
            notes.append(UltrastarNote(
                note_type="-",
                start_beat=int(trimmed[2:]),
                duration=0,
                pitch=0,
                syllable="",
            ))

    bpm_str = meta.get("BPM", "120").replace(",", ".")
    bpm = float(bpm_str)

    return (
        UltrastarMeta(
            title=meta.get("TITLE", ""),
            artist=meta.get("ARTIST", ""),
            mp3=meta.get("MP3", ""),
            bpm=bpm,
            gap=int(meta.get("GAP", "0")),
            video=meta.get("VIDEO"),
        ),
        notes,
    )
