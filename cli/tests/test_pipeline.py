"""Tests for cli/pipeline.py: lyrics prep, filename sanitizing, run_process orchestration."""

import json
from pathlib import Path

import pytest
from cli.config import Config
from cli.pipeline import (
    ProcessRequest,
    looks_like_ultrastar,
    prepare_lyrics_input,
    run_process,
    sanitize_filename,
    sanitize_output_name,
)
from cli.pipeline_types import BpmResult, TranscribeResult, WordTimestamp


ULTRASTAR_TEXT = (
    "#TITLE:Test Song\n"
    "#ARTIST:Tester\n"
    "#MP3:song.mp3\n"
    "#BPM:120.00\n"
    "#GAP:500\n"
    "\n"
    ": 0 4 60 Hello \n"
    "* 1 4 62  World \n"
    "- 4\n"
    "E\n"
)


class TestLooksLikeUltrastar:
    def test_plain_text(self):
        assert not looks_like_ultrastar("hello\nworld\n")

    def test_ultrastar_header(self):
        assert looks_like_ultrastar(ULTRASTAR_TEXT)

    def test_blank_lines_before_header(self):
        assert looks_like_ultrastar("\n\n#TITLE:X\n")

    def test_empty(self):
        assert not looks_like_ultrastar("")


class TestPrepareLyricsInput:
    def test_plain_text_passthrough(self):
        lyrics, meta = prepare_lyrics_input("hello\nworld\n")
        assert lyrics == "hello\nworld\n"
        assert meta is None

    def test_ultrastar_extracts_lyrics_and_meta(self):
        lyrics, meta = prepare_lyrics_input(ULTRASTAR_TEXT)
        assert meta is not None
        assert meta.title == "Test Song"
        assert meta.artist == "Tester"
        assert meta.mp3 == "song.mp3"
        assert lyrics.strip() == "Hello World"

    def test_empty_text(self):
        lyrics, meta = prepare_lyrics_input("")
        assert lyrics == ""
        assert meta is None


class TestSanitizeFilename:
    def test_plain_title_unchanged(self):
        assert sanitize_filename("Hello World") == "Hello World"

    def test_illegal_chars_replaced(self):
        assert sanitize_filename('A<B>C:"D/E\\F|G?H*I') == "A B C D E F G H I"

    def test_whitespace_collapsed(self):
        assert sanitize_filename("  a \t\n b  ") == "a b"

    def test_long_title_capped(self):
        out = sanitize_filename("x" * 200)
        assert len(out) <= 80

    def test_empty_falls_back(self):
        assert sanitize_filename("   ") == "untitled"
        assert sanitize_filename('??*') == "untitled"

    def test_trailing_dots_stripped(self):
        assert sanitize_filename("Song...") == "Song"


class TestSanitizeOutputName:
    def test_artist_and_title(self):
        assert sanitize_output_name("Tester", "Test Song") == "Tester - Test Song"

    def test_empty_artist_title_only(self):
        assert sanitize_output_name("", "Test Song") == "Test Song"

    def test_empty_title_artist_only(self):
        assert sanitize_output_name("Tester", "  ") == "Tester"

    def test_both_empty_falls_back(self):
        assert sanitize_output_name("", "") == "untitled"

    def test_illegal_chars_sanitized(self):
        assert sanitize_output_name("A/B", "C?D") == "A B - C D"

    def test_long_name_capped(self):
        out = sanitize_output_name("x" * 100, "y" * 100)
        assert len(out) <= 80


def _make_transcribe_result(tmp: Path) -> TranscribeResult:
    return TranscribeResult(
        words=[WordTimestamp(word="hello", start=1.0, end=2.0, midi=60)],
        language="en",
        vocals_path=str(tmp / "song_vocals.mp3"),
        accompaniment_path=str(tmp / "song_accompaniment.mp3"),
        pauses=[],
        bpm_result=None,
    )


def _make_request(tmp: Path, input_name: str = "song.mp3", **overrides) -> ProcessRequest:
    input_path = tmp / input_name
    input_path.write_bytes(b"ID3fake")
    config = Config(temp_dir=str(tmp / "tmp"), output_dir=str(tmp / "out"))
    base = dict(
        title="Test Song",
        artist="Tester",
        lyrics_text="Hello World",
        input_path=input_path,
        video_path=None,
        cover_path=None,
        config=config,
        output_dir=tmp / "out",
    )
    base.update(overrides)
    return ProcessRequest(**base)


class _Calls:
    def __init__(self):
        self.order: list[str] = []
        self.kwargs: dict[str, dict] = {}


class TestRunProcess:
    def _patch_stages(self, monkeypatch, tmp: Path, calls: _Calls, stage_error: Exception | None = None):
        def fake_extract(src, dst, config):
            calls.order.append("extract")
            dst.write_bytes(b"ID3mono")

        def fake_transcribe(audio, lyrics, config):
            calls.order.append("transcribe")
            calls.kwargs["transcribe"] = {"lyrics": lyrics}
            if stage_error is not None:
                raise stage_error
            (config.temp_path / "song_transcribe.json").write_text("{}", encoding="utf-8")
            return _make_transcribe_result(config.temp_path)

        def fake_detect_bpm(path, config):
            calls.order.append("bpm")
            return BpmResult(bpm=120.0, first_beat_ms=100.0, stable=True)

        def fake_align(lyrics, words, language, pauses, config, pitch_frames=None, audio_path=None):
            calls.order.append("align")
            return []

        def fake_generate(**kwargs):
            calls.order.append("generate")
            calls.kwargs["generate"] = kwargs
            return "#TITLE:Test Song\n"

        def fake_package(**kwargs):
            calls.order.append("package")
            calls.kwargs["package"] = kwargs
            kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
            (kwargs["output_dir"] / "Test Song.txt").write_text("x", encoding="utf-8")
            (kwargs["output_dir"] / "Test Song.zip").write_bytes(b"zip")
            return kwargs["output_dir"]

        def fake_editor(txt_path, output_html=None, output_json=None, pitch_json_path=None,
                        vocals_hint=None, embed_audio=False):
            calls.order.append("editor")
            calls.kwargs["editor"] = {
                "pitch_json_path": pitch_json_path,
                "vocals_hint": vocals_hint,
                "output_json": output_json,
            }
            assert output_html is not None
            output_html.write_text("<html></html>", encoding="utf-8")
            assert output_json is not None
            output_json.write_text("{}", encoding="utf-8")
            return output_html

        def fake_preview(txt_path, output_html=None, pitch_json_path=None):
            calls.order.append("preview")
            calls.kwargs["preview"] = {"pitch_json_path": pitch_json_path}
            assert output_html is not None
            output_html.write_text("<html></html>", encoding="utf-8")
            return output_html

        monkeypatch.setattr("cli.editor.generate_editor", fake_editor)
        monkeypatch.setattr("cli.ffmpeg_extract.extract_audio", fake_extract)
        monkeypatch.setattr("cli.ffmpeg_extract.is_video_path", lambda p: False)
        monkeypatch.setattr("cli.transcribe.transcribe", fake_transcribe)
        monkeypatch.setattr("cli.bpm_detect.detect_bpm", fake_detect_bpm)
        monkeypatch.setattr("cli.align.align_lyrics", fake_align)
        monkeypatch.setattr("cli.generate.generate_ultrastar", fake_generate)
        monkeypatch.setattr("cli.package.package_output", fake_package)
        monkeypatch.setattr("cli.html_preview.generate_preview", fake_preview)

    def test_full_stage_call_sequence(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        req = _make_request(tmp_path)
        result = run_process(req)
        assert result.ok
        assert calls.order == ["extract", "transcribe", "bpm", "align", "generate", "editor", "package", "preview"]
        safe = "Tester - Test Song"
        run_dir = tmp_path / "out" / safe
        assert result.output_dir == run_dir
        assert result.txt_path == run_dir / f"{safe}.txt"
        assert result.zip_path == run_dir / f"{safe}.zip"
        assert result.html_path == run_dir / f"{safe}.html"
        assert result.editor_path == run_dir / f"{safe}_editor.html"
        assert result.editor_json_path == run_dir / f"{safe}_editor.json"
        assert result.temp_dir == tmp_path / "tmp"
        # BPM from the transcribe result (None) triggered detection
        assert calls.kwargs["generate"]["bpm"] == 120.0
        assert calls.kwargs["generate"]["mp3_filename"] == f"{safe}.mp3"

    def test_editor_html_written_and_zipped(self, monkeypatch, tmp_path):
        import zipfile
        from cli.package import package_output

        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        # A parseable Ultrastar txt so the real editor generator can run
        monkeypatch.setattr("cli.generate.generate_ultrastar", lambda **kw: ULTRASTAR_TEXT)
        monkeypatch.setattr("cli.package.package_output", package_output)
        # Stems present so the package copies (and renames) them
        tmp = tmp_path / "tmp"
        tmp.mkdir()
        (tmp / "song_vocals.mp3").write_bytes(b"v")
        (tmp / "song_accompaniment.mp3").write_bytes(b"a")
        result = run_process(_make_request(tmp_path))
        assert result.ok
        safe = "Tester - Test Song"
        run_dir = tmp_path / "out" / safe
        editor = run_dir / f"{safe}_editor.html"
        editor_json = run_dir / f"{safe}_editor.json"
        assert editor.is_file()
        assert editor_json.is_file()
        assert result.editor_path == editor
        assert result.editor_json_path == editor_json
        assert calls.kwargs["editor"]["vocals_hint"] == f"{safe}_vocals.mp3"
        with zipfile.ZipFile(run_dir / f"{safe}.zip") as zf:
            names = set(zf.namelist())
        assert f"{safe}_editor.html" in names
        assert f"{safe}_editor.json" in names
        assert f"{safe}.txt" in names
        assert f"{safe}.mp3" in names
        assert f"{safe}_vocals.mp3" in names
        assert f"{safe}_accompaniment.mp3" in names
        assert "vocals.mp3" not in names
        assert "accompaniment.mp3" not in names

    def test_title_sanitized_for_filenames(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        req = _make_request(tmp_path, title="A/B: C?")
        result = run_process(req)
        assert result.ok
        safe = "Tester - A B C"
        assert calls.kwargs["generate"]["mp3_filename"] == f"{safe}.mp3"
        # #TITLE tag keeps the original title
        assert calls.kwargs["generate"]["title"] == "A/B: C?"
        assert result.output_dir == tmp_path / "out" / safe
        assert result.txt_path.name == f"{safe}.txt"

    def test_resume_skips_extract_and_transcribe(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        req = _make_request(tmp_path)
        (tmp_path / "tmp").mkdir()
        resume_file = tmp_path / "resume.json"
        resume_file.write_text(json.dumps(_make_transcribe_result(tmp_path).to_dict()), encoding="utf-8")
        result = run_process(_make_request(tmp_path, resume_path=resume_file))
        assert result.ok
        assert "extract" not in calls.order
        assert "transcribe" not in calls.order
        assert calls.order == ["bpm", "align", "generate", "editor", "package", "preview"]

    def test_missing_resume_file(self, tmp_path):
        req = _make_request(tmp_path, resume_path=tmp_path / "nope.json")
        result = run_process(req)
        assert not result.ok
        assert "Resume file not found" in (result.error or "")

    def test_missing_input_file(self, tmp_path):
        req = _make_request(tmp_path)
        req.input_path.unlink()
        result = run_process(req)
        assert not result.ok
        assert "not found" in (result.error or "")

    def test_stage_error_returns_ok_false(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls, stage_error=RuntimeError("boom"))
        result = run_process(_make_request(tmp_path))
        assert not result.ok
        assert "boom" in (result.error or "")

    def test_video_from_input(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        monkeypatch.setattr("cli.ffmpeg_extract.is_video_path", lambda p: True)
        req = _make_request(tmp_path, input_name="song.mp4")
        result = run_process(req)
        assert result.ok
        # The video is renamed to the artist-title base name (same extension)
        assert calls.kwargs["generate"]["video_filename"] == "Tester - Test Song.mp4"
        assert calls.kwargs["package"]["video_path"] == tmp_path / "song.mp4"

    def test_explicit_video_wins(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        video = tmp_path / "other.mov"
        video.write_bytes(b"v")
        req = _make_request(tmp_path, video_path=video)
        run_process(req)
        assert calls.kwargs["generate"]["video_filename"] == "Tester - Test Song.mov"
        assert calls.kwargs["package"]["video_path"] == video

    def test_cover_filename_and_path(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        cover = tmp_path / "cover.jpeg"
        cover.write_bytes(b"jpeg")
        run_process(_make_request(tmp_path, cover_path=cover))
        safe = "Tester - Test Song"
        assert calls.kwargs["generate"]["cover_filename"] == f"{safe}_cover.jpeg"
        assert calls.kwargs["package"]["cover_path"] == cover

    def test_missing_cover_file_ignored(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        run_process(_make_request(tmp_path, cover_path=tmp_path / "nope.jpeg"))
        assert calls.kwargs["generate"]["cover_filename"] is None
        assert calls.kwargs["package"]["cover_path"] is None

    def test_stage_extract_only(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        result = run_process(_make_request(tmp_path, stage="extract"))
        assert result.ok
        assert calls.order == ["extract"]
        assert result.txt_path is None

    def test_include_intermediates_toggle(self, monkeypatch, tmp_path):
        calls = _Calls()
        self._patch_stages(monkeypatch, tmp_path, calls)
        run_process(_make_request(tmp_path, include_intermediates=True))
        assert calls.kwargs["package"]["extra_files"], "intermediates should be collected by default"
        calls.kwargs.clear()
        run_process(_make_request(tmp_path, include_intermediates=False))
        assert calls.kwargs["package"]["extra_files"] == []
