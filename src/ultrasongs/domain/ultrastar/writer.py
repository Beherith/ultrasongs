"""Serialization of canonical UltraStar songs."""

from __future__ import annotations

from pathlib import Path

from .models import LineBreak, UltrastarSong, format_number


def write_ultrastar_text(song: UltrastarSong) -> str:
    lines = [f"#{key}:{value}" for key, value in song.metadata.as_headers().items()]
    lines.append("")
    for event in song.events:
        if isinstance(event, LineBreak):
            line = f"- {format_number(event.start_beat)}"
            if event.end_beat is not None:
                line += f" {format_number(event.end_beat)}"
            lines.append(line)
        else:
            lines.append(
                f"{event.note_type.value} {format_number(event.start_beat)} "
                f"{format_number(event.duration_beats)} {event.pitch} {event.lyric}"
            )
    lines.append("E")
    return "\n".join(lines) + "\n"


def write_ultrastar_file(song: UltrastarSong, path: str | Path) -> None:
    Path(path).write_text(write_ultrastar_text(song), encoding="utf-8", newline="\n")
