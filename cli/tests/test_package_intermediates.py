"""Tests for package_output intermediates and collect_intermediates."""

import os
import time
import zipfile
from pathlib import Path

from cli.package import collect_intermediates, package_output


def _make_mp3(path: Path) -> Path:
    path.write_bytes(b"ID3fake")
    return path


class TestCollectIntermediates:
    def test_includes_json_html_txt(self, tmp_path: Path):
        (tmp_path / "a_transcribe.json").write_text("{}", encoding="utf-8")
        (tmp_path / "whisperx_pitch.html").write_text("<html></html>", encoding="utf-8")
        (tmp_path / "note_segments.txt").write_text("notes", encoding="utf-8")
        found = collect_intermediates(tmp_path)
        names = {f.name for f in found}
        assert names == {"a_transcribe.json", "whisperx_pitch.html", "note_segments.txt"}

    def test_skips_mp3(self, tmp_path: Path):
        (tmp_path / "song.mp3").write_bytes(b"x")
        (tmp_path / "vocals.mp3").write_bytes(b"x")
        assert collect_intermediates(tmp_path) == []

    def test_skips_plots_directory(self, tmp_path: Path):
        plots = tmp_path / "note_segments_plots"
        plots.mkdir()
        (plots / "fig1.txt").write_text("t", encoding="utf-8")
        (plots / "fig2.json").write_text("{}", encoding="utf-8")
        assert collect_intermediates(tmp_path) == []

    def test_since_ts_filters_stale_files(self, tmp_path: Path):
        stale = tmp_path / "stale.json"
        stale.write_text("{}", encoding="utf-8")
        old = time.time() - 100
        time.sleep(0.01)
        os.utime(stale, (old, old))
        fresh = tmp_path / "fresh.json"
        fresh.write_text("{}", encoding="utf-8")

        started = time.time() - 10
        found = collect_intermediates(tmp_path, since_ts=started)
        assert [f.name for f in found] == ["fresh.json"]

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert collect_intermediates(tmp_path / "nope") == []


class TestPackageOutputZip:
    def test_basic_zip_layout_without_extras(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
        )
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert names == {"T.txt", "T.mp3"}

    def test_extras_written_under_intermediates(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        tmp = tmp_path / "tmp"
        tmp.mkdir()
        transcribe = tmp / "song_transcribe.json"
        transcribe.write_text("{}", encoding="utf-8")
        pitch = tmp / "whisperx_pitch.json"
        pitch.write_text("{}", encoding="utf-8")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
            extra_files=[transcribe, pitch],
        )
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert "intermediates/song_transcribe.json" in names
        assert "intermediates/whisperx_pitch.json" in names
        assert "T.txt" in names and "T.mp3" in names

    def test_extras_with_stems_and_video(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        vocals = _make_mp3(tmp_path / "vocals.mp3")
        acc = _make_mp3(tmp_path / "acc.mp3")
        video = tmp_path / "song.mp4"
        video.write_bytes(b"vid")
        extra = tmp_path / "align_debug.json"
        extra.write_text("{}", encoding="utf-8")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
            video_path=video,
            vocals_path=vocals,
            accompaniment_path=acc,
            extra_files=[extra],
        )
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert names == {
            "T.txt", "T.mp3", "T.mp4", "T_vocals.mp3", "T_accompaniment.mp3",
            "intermediates/align_debug.json",
        }

    def test_cover_copied_and_zipped(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        cover = tmp_path / "album_cover.jpg"
        cover.write_bytes(b"jpegdata")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
            cover_path=cover,
        )
        cover_out = out / "T_cover.jpg"
        assert cover_out.is_file()
        assert cover_out.read_bytes() == b"jpegdata"
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert names == {"T.txt", "T.mp3", "T_cover.jpg"}

    def test_missing_cover_file_skipped(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
            cover_path=tmp_path / "ghost.jpg",
        )
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert names == {"T.txt", "T.mp3"}

    def test_missing_extra_file_skipped(self, tmp_path: Path):
        out = tmp_path / "out"
        mp3 = _make_mp3(tmp_path / "song.mp3")
        package_output(
            txt_content="#TITLE: T\n",
            mp3_path=mp3,
            output_dir=out,
            name="T",
            extra_files=[tmp_path / "ghost.json"],
        )
        with zipfile.ZipFile(out / "T.zip") as zf:
            names = set(zf.namelist())
        assert not any(n.startswith("intermediates/") for n in names)
