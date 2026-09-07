"""Tests for cli/web/app.py: layout generation, validation, config override flow."""

import base64
from pathlib import Path

import pytest
from dash import no_update
from cli.config import Config
from cli.web import app as webapp
from cli.web import settings_meta
from cli.web.app import (
    WebConfig,
    build_config_overrides,
    create_app,
    load_web_config,
    lyrics_prefill,
    parse_setting_value,
    validate_process_inputs,
)
from cli.web.jobs import JobManager


def _walk_components(node, out: dict):
    if getattr(node, "id", None) is not None:
        out[node.id] = node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        _walk_components(child, out)


class TestLayout:
    def test_layout_builds_all_setting_controls(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        manager = JobManager(web_dir=tmp_path / "jobs")
        app = create_app(WebConfig(), config, manager, password=None)
        nodes: dict = {}
        _walk_components(app.layout, nodes)
        expected = {webapp.SETTING_ID.format(key=m.key) for m in settings_meta.SETTINGS}
        ids = set(nodes)
        assert expected <= ids
        assert len(expected) == 44
        # Core form controls exist
        for cid in ("upload", "title", "artist", "lyrics", "process-btn",
                    "reset-btn", "job-status", "job-log", "job-result",
                    "upload-store", "active-job-store", "poll-interval"):
            assert cid in ids, cid

    def test_settings_cover_all_config_keys(self):
        keys = {m.key for m in settings_meta.SETTINGS}
        config_keys = {f.name for f in Config.__dataclass_fields__.values()
                       if not f.name.startswith("_")}
        assert keys == config_keys

    def test_setting_defaults_match_config(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        manager = JobManager(web_dir=tmp_path / "jobs")
        app = create_app(WebConfig(), config, manager, password=None)
        nodes: dict = {}
        _walk_components(app.layout, nodes)

        for meta in settings_meta.SETTINGS:
            node = nodes.get(webapp.SETTING_ID.format(key=meta.key))
            assert node is not None, meta.key
            if meta.kind == "checkbox":
                assert node.value == bool(getattr(config, meta.key))
            else:
                assert node.value == getattr(config, meta.key)


class TestHelpers:
    def test_validate_missing_upload(self):
        assert validate_process_inputs(None, "T", "A", "ly") is not None

    def test_validate_missing_title(self):
        assert validate_process_inputs({"name": "x.mp3"}, "  ", "A", "ly") is not None

    def test_validate_missing_artist(self):
        assert validate_process_inputs({"name": "x.mp3"}, "T", "", "ly") is not None

    def test_validate_missing_lyrics(self):
        assert validate_process_inputs({"name": "x.mp3"}, "T", "A", "  ") is not None

    def test_validate_ok(self):
        assert validate_process_inputs({"name": "x.mp3"}, "T", "A", "ly") is None

    def test_parse_setting_value_kinds(self):
        num = settings_meta.SETTINGS_BY_KEY["whisperx_batch_size"]
        assert parse_setting_value(num, "4") == 4
        f = settings_meta.SETTINGS_BY_KEY["pause_threshold_pct"]
        assert parse_setting_value(f, "5.5") == 5.5
        cb = settings_meta.SETTINGS_BY_KEY["debug_alignment"]
        assert parse_setting_value(cb, True) is True
        sel = settings_meta.SETTINGS_BY_KEY["whisper_model"]
        assert parse_setting_value(sel, "large") == "large"

    def test_parse_number_out_of_range(self):
        meta = settings_meta.SETTINGS_BY_KEY["whisperx_batch_size"]
        with pytest.raises(ValueError):
            parse_setting_value(meta, "0")

    def test_build_config_overrides(self):
        overrides = build_config_overrides({
            "whisper_model": "small",
            "transcribe_runs": "7",
            "debug_alignment": True,
            "pause_threshold_pct": "3.5",
            "whisper_language": "",
        })
        assert overrides == {
            "whisper_model": "small",
            "transcribe_runs": 7,
            "debug_alignment": True,
            "pause_threshold_pct": 3.5,
            "whisper_language": "",
        }

    def test_lyrics_prefill_from_ultrastar_headers(self):
        text = "#TITLE:Song\n#ARTIST:Who\n\n: 0 4 60 Hey \nE\n"
        assert lyrics_prefill(text, "", "") == ("Song", "Who")

    def test_lyrics_prefill_keeps_existing(self):
        text = "#TITLE:Song\n#ARTIST:Who\n"
        assert lyrics_prefill(text, "Keep", "Me") == ("Keep", "Me")

    def test_lyrics_prefill_plain_text_noop(self):
        assert lyrics_prefill("hello\nworld", "T", "A") == ("T", "A")


class TestLoadWebConfig:
    def test_defaults_when_missing(self, tmp_path):
        cfg = load_web_config(tmp_path / "nope.jsonc")
        assert cfg == WebConfig()

    def test_values_from_file(self, tmp_path):
        p = tmp_path / "web_config.jsonc"
        p.write_text('''
{
  // comment
  "host": "0.0.0.0",
  "port": 9090,
  "web_dir": "/custom/jobs",
  "job_retention_days": 3,
  "max_upload_mb": 512,
  "poll_interval_s": 2.5,
}
''', encoding="utf-8")
        cfg = load_web_config(p)
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9090
        assert cfg.web_dir == "/custom/jobs"
        assert cfg.job_retention_days == 3.0
        assert cfg.max_upload_mb == 512.0
        assert cfg.poll_interval_s == 2.5

    def test_invalid_values_fall_back(self, tmp_path):
        p = tmp_path / "web_config.jsonc"
        p.write_text('{"port": "not_a_port"}', encoding="utf-8")
        cfg = load_web_config(p)
        assert cfg.port == 8030
        assert cfg.host == "127.0.0.1"


class _FakeJob:
    def __init__(self, job_id="fakejob123"):
        self.id = job_id
        self.status = "queued"


class _FakeManager:
    def __init__(self, web_dir, max_upload_mb=2048.0):
        self.web_dir = Path(web_dir)
        self.max_upload_bytes = int(max_upload_mb * 1024 * 1024)
        self.submitted = []

    def submit(self, req, upload_bytes, upload_name):
        self.submitted.append((req, upload_bytes, upload_name))
        return _FakeJob()

    def job(self, job_id):
        return None

    def snapshot(self, job_id):
        return None


def _find_callback(app, name):
    for cb in app.callback_map.values():
        fn = cb.get("callback") if isinstance(cb, dict) else getattr(cb, "callback", None)
        if getattr(fn, "__name__", None) == name:
            return getattr(fn, "__wrapped__", fn)
    raise AssertionError(f"callback {name} not found")


def _text(node):
    if node is None:
        return ""
    if hasattr(node, "children"):
        return str(node.children)
    return str(node)


class TestProcessClickCallback:
    def _make_app(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        manager = _FakeManager(tmp_path / "jobs")
        app = create_app(WebConfig(), config, manager, password=None)
        return app, manager

    def _setting_values(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        return [getattr(config, m.key) for m in settings_meta.SETTINGS]

    def test_rejects_missing_title_without_submitting(self, tmp_path):
        app, manager = self._make_app(tmp_path)
        cb = _find_callback(app, "process_click")
        staged = tmp_path / "staged.mp3"
        staged.write_bytes(b"x")
        upload_info = {"path": str(staged), "name": "song.mp3", "size": 1}
        job_id, error = cb(1, upload_info, "", "Artist", "lyrics", *self._setting_values(tmp_path))
        assert job_id is no_update
        assert "title" in _text(error)
        assert manager.submitted == []

    def test_rejects_missing_upload_without_submitting(self, tmp_path):
        app, manager = self._make_app(tmp_path)
        cb = _find_callback(app, "process_click")
        job_id, error = cb(1, None, "Title", "Artist", "lyrics", *self._setting_values(tmp_path))
        assert error is not None
        assert "upload" in _text(error)
        assert manager.submitted == []

    def test_submits_valid_job(self, tmp_path):
        app, manager = self._make_app(tmp_path)
        cb = _find_callback(app, "process_click")
        staged = tmp_path / "staged.mp3"
        staged.write_bytes(b"ID3data")
        upload_info = {"path": str(staged), "name": "song.mp3", "size": 7}
        job_id, error = cb(1, upload_info, "Title", "Artist", "Hey you",
                           *self._setting_values(tmp_path))
        assert error is None
        assert job_id == "fakejob123"
        assert len(manager.submitted) == 1
        req, upload_bytes, name = manager.submitted[0]
        assert req.title == "Title"
        assert req.artist == "Artist"
        assert req.lyrics_text == "Hey you"
        assert upload_bytes == b"ID3data"
        assert name == "song.mp3"
        assert not staged.exists()  # staging cleaned up

    def test_invalid_setting_value_rejected(self, tmp_path):
        app, manager = self._make_app(tmp_path)
        cb = _find_callback(app, "process_click")
        staged = tmp_path / "staged.mp3"
        staged.write_bytes(b"x")
        upload_info = {"path": str(staged), "name": "song.mp3", "size": 1}
        values = self._setting_values(tmp_path)
        # whisperx_batch_size is index 5 in SETTINGS
        idx = [m.key for m in settings_meta.SETTINGS].index("whisperx_batch_size")
        values[idx] = "0"
        job_id, error = cb(1, upload_info, "Title", "Artist", "Hey", *values)
        assert error is not None
        assert "Invalid setting" in _text(error)
        assert manager.submitted == []


class TestUploadCallback:
    def test_rejects_bad_extension(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        manager = _FakeManager(tmp_path / "jobs")
        app = create_app(WebConfig(), config, manager, password=None)
        cb = _find_callback(app, "upload_picked")
        content = "data:application/octet-stream;base64," + base64.b64encode(b"x").decode()
        store, info = cb(content, "evil.exe")
        assert info is not None
        assert "Unsupported file type" in _text(info)

    def test_accepts_mp3(self, tmp_path):
        config = Config(temp_dir=str(tmp_path / "tmp"), output_dir=str(tmp_path / "out"))
        manager = _FakeManager(tmp_path / "jobs")
        app = create_app(WebConfig(), config, manager, password=None)
        cb = _find_callback(app, "upload_picked")
        payload = b"ID3fake"
        content = "data:audio/mpeg;base64," + base64.b64encode(payload).decode()
        store, info = cb(content, "song.mp3")
        assert store is not None
        assert store["name"] == "song.mp3"
        staged = Path(store["path"])
        assert staged.read_bytes() == payload
        assert staged.parent.name == webapp.STAGING_DIRNAME
