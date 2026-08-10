"""Tests for configuration module."""

import json
import tempfile
from pathlib import Path

import pytest
from cli.config import Config, load_config


class TestConfigDefaults:
    def test_default_values(self):
        cfg = Config()
        assert cfg.whisper_model == "medium"
        assert cfg.demucs_model == "htdemucs"
        assert cfg.sample_rate == 44100
        assert cfg.pitch_min_hz == 65.41
        assert cfg.pitch_max_hz == 1046.5
        assert cfg.crepe_hop_ms == 10
        assert cfg.pause_min_silence_ms == 400
        assert cfg.pause_threshold_pct == 5.0
        assert cfg.gap_lead_in_ms == 500
        assert cfg.linebreak_beat_offset == 4
        assert cfg.ffmpeg_audio_bitrate == "128k"
        assert cfg.output_dir == "./output"
        assert cfg.temp_dir == "./tmp"
        assert cfg.debug_alignment is False
        assert cfg.bpm_use_accompaniment is False

    def test_frozen(self):
        cfg = Config()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.whisper_model = "large"


class TestLoadConfig:
    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonc", delete=False) as f:
            json.dump({
                "whisper_model": "large",
                "sample_rate": 48000,
                "output_dir": "/custom/output",
            }, f)
            f.flush()
            cfg = load_config(f.name)
            assert cfg.whisper_model == "large"
            assert cfg.sample_rate == 48000
            assert cfg.output_dir == "/custom/output"
            # Defaults preserved
            assert cfg.demucs_model == "htdemucs"

    def test_load_with_comments(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonc", delete=False) as f:
            f.write('''
{
  // This is a comment
  "whisper_model": "tiny",
  /* multi-line
     comment */
  "sample_rate": 22050,
}
''')
            f.flush()
            cfg = load_config(f.name)
            assert cfg.whisper_model == "tiny"
            assert cfg.sample_rate == 22050

    def test_missing_file_returns_defaults(self):
        cfg = load_config("/nonexistent/path/config.jsonc")
        assert cfg.whisper_model == "medium"

    def test_invalid_value_uses_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonc", delete=False) as f:
            json.dump({
                "sample_rate": "not_a_number",
            }, f)
            f.flush()
            cfg = load_config(f.name)
            assert cfg.sample_rate == 44100  # default
