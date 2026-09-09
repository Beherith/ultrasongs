from __future__ import annotations

import io
import zipfile
from pathlib import Path

from ultrasongs.domain.ultrastar import (
    NoteType,
    UltrastarMetadata,
    UltrastarNote,
    UltrastarSong,
    build_export_zip,
    parse_ultrastar_text,
    safe_filename,
)


def test_builds_expected_export_members(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    vocals = tmp_path / "stem.mp3"
    audio.write_bytes(b"audio")
    vocals.write_bytes(b"vocals")
    song = UltrastarSong(
        UltrastarMetadata(title="A/B", artist="Artist", mp3="audio.mp3"),
        (UltrastarNote(NoteType.NORMAL, 0, 4, 60, "Hello"),),
    )

    payload = build_export_zip(song, audio_path=audio, vocals_path=vocals)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {"A_B.txt", "audio.mp3", "vocals.mp3"}
        parsed = parse_ultrastar_text(archive.read("A_B.txt").decode("utf-8"))
        assert parsed.metadata.title == "A/B"
        assert archive.read("audio.mp3") == b"audio"


def test_safe_filename_removes_path_and_windows_metacharacters() -> None:
    assert safe_filename('../bad:name?.txt') == "bad_name_.txt"


def test_missing_optional_file_is_not_silently_ignored(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    song = UltrastarSong(UltrastarMetadata(mp3="audio.mp3"))

    try:
        build_export_zip(song, audio_path=audio, vocals_path=tmp_path / "missing.mp3")
    except FileNotFoundError as exc:
        assert "vocals.mp3" in str(exc)
    else:
        raise AssertionError("missing optional artifact should fail export")


def test_export_can_exclude_audio_and_use_bom_encoding(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    song = UltrastarSong(UltrastarMetadata(title="Song", mp3="audio.mp3"))

    payload = build_export_zip(
        song,
        audio_path=audio,
        include_audio=False,
        text_encoding="utf-8-sig",
    )

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.namelist() == ["Song.txt"]
        assert archive.read("Song.txt").startswith(b"\xef\xbb\xbf")
