"""Tests for config_from_dict and load_jsonc."""

import json
from pathlib import Path

from cli.config import Config, config_from_dict, load_config, load_jsonc


class TestConfigFromDict:
    def test_empty_dict_gives_defaults(self):
        cfg = config_from_dict({})
        expected = Config()
        assert cfg.whisper_model == expected.whisper_model
        assert cfg.transcribe_runs == expected.transcribe_runs

    def test_valid_values_applied(self):
        cfg = config_from_dict({
            "transcription_backend": "whisperx",
            "whisper_model": "large",
            "whisperx_batch_size": 4,
            "whisperx_chunk_pause_ms": 1500,
            "whisperx_align_runs": 5,
            "sample_rate": 48000,
            "output_dir": "/custom/output",
            "temp_dir": "/custom/tmp",
        })
        assert cfg.transcription_backend == "whisperx"
        assert cfg.whisper_model == "large"
        assert cfg.whisperx_batch_size == 4
        assert cfg.whisperx_chunk_pause_ms == 1500
        assert cfg.whisperx_align_runs == 5
        assert cfg.sample_rate == 48000
        assert cfg.output_dir == "/custom/output"
        assert cfg.temp_dir == "/custom/tmp"
        # Defaults preserved
        assert cfg.demucs_model == "htdemucs"

    def test_empty_language_survives_round_trip(self):
        cfg = config_from_dict({"whisper_language": ""})
        assert cfg.whisper_language == ""

    def test_source_path_recorded(self):
        cfg = config_from_dict({}, Path("/some/config.jsonc"))
        assert cfg._config_path == Path("/some/config.jsonc")
        cfg_none = config_from_dict({})
        assert cfg_none._config_path is None

    def test_invalid_values_use_defaults(self):
        cfg = config_from_dict({
            "sample_rate": "not_a_number",
            "whisperx_batch_size": 0,
            "whisperx_compute_type": "half-ish",
            "whisperx_interpolate_method": "guess",
            "whisperx_chunk_pause_ms": -1,
            "whisperx_align_runs": 4,
            "transcription_backend": "magic",
            "faster_whisper_compute_type": "half-ish",
        })
        defaults = Config()
        assert cfg.sample_rate == defaults.sample_rate
        assert cfg.whisperx_batch_size == defaults.whisperx_batch_size
        assert cfg.whisperx_compute_type == defaults.whisperx_compute_type
        assert cfg.whisperx_interpolate_method == defaults.whisperx_interpolate_method
        assert cfg.whisperx_chunk_pause_ms == defaults.whisperx_chunk_pause_ms
        assert cfg.whisperx_align_runs == defaults.whisperx_align_runs
        assert cfg.transcription_backend == defaults.transcription_backend
        assert cfg.faster_whisper_compute_type == defaults.faster_whisper_compute_type

    def test_unknown_keys_ignored(self):
        cfg = config_from_dict({"not_a_key": 123, "whisper_model": "small"})
        assert cfg.whisper_model == "small"


class TestLoadJsonc:
    def test_loads_comments_and_trailing_commas(self, tmp_path: Path):
        p = tmp_path / "cfg.jsonc"
        p.write_text('''
{
  // comment
  "a": 1,
  /* block
     comment */
  "b": "x",
}
''', encoding="utf-8")
        assert load_jsonc(p) == {"a": 1, "b": "x"}


class TestLoadConfigUnchanged:
    def test_load_config_matches_config_from_dict(self, tmp_path: Path):
        p = tmp_path / "cfg.jsonc"
        p.write_text(json.dumps({"whisper_model": "base"}), encoding="utf-8")
        from_file = load_config(str(p))
        from_dict = config_from_dict(json.loads(p.read_text(encoding="utf-8")), p)
        assert from_file.whisper_model == from_dict.whisper_model
        assert from_file._config_path == p
