"""Build downloadable UltraStar ZIP bundles with safe member names."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .models import UltrastarSong
from .writer import write_ultrastar_text

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(value: str, *, fallback: str = "song") -> str:
    name = _UNSAFE_FILENAME.sub("_", value).strip(" ._")
    return name or fallback


def build_export_zip(
    song: UltrastarSong,
    *,
    audio_path: str | Path,
    video_path: str | Path | None = None,
    vocals_path: str | Path | None = None,
    accompaniment_path: str | Path | None = None,
    text_filename: str | None = None,
    text_encoding: str = "utf-8",
    include_audio: bool = True,
) -> bytes:
    """Return an in-memory ZIP matching the legacy download contents."""

    audio = _required_file(audio_path, "audio") if include_audio else None
    optional_files = (
        (video_path, safe_filename(song.metadata.video or "video.mp4")),
        (vocals_path, "vocals.mp3"),
        (accompaniment_path, "accompaniment.mp3"),
    )
    txt_name = safe_filename(text_filename or f"{song.metadata.title or 'song'}.txt")
    audio_name = safe_filename(song.metadata.mp3 or audio.name, fallback="audio.mp3")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(txt_name, write_ultrastar_text(song).encode(text_encoding))
        if audio is not None:
            archive.write(audio, audio_name)
        for source, member_name in optional_files:
            if source is None:
                continue
            path = _required_file(source, member_name)
            archive.write(path, member_name)
    return output.getvalue()


def _required_file(value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path
