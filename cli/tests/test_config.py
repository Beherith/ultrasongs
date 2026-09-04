"""Tests for configuration module."""

import json
import tempfile
from pathlib import Path

import pytest
from cli.config import Config, load_config


class TestConfigDefaults:
    def test_default_values(self):
        cfg = Config()
        assert cfg.transcription_backend == "faster-whisper"
        assert cfg.whisper_model == "medium"
        assert cfg.whisper_language == "en"
        assert cfg.faster_whisper_compute_type == "auto"
        assert cfg.whisperx_batch_size == 8
        assert cfg.whisperx_compute_type == "default"
        assert cfg.whisperx_align_model == ""
        assert cfg.whisperx_interpolate_method == "nearest"
        assert cfg.whisperx_chunk_pause_ms == 1000
        assert cfg.whisperx_align_runs == 3
        assert cfg.transcribe_runs == 3
        assert cfg.demucs_model == "htdemucs"
        assert cfg.sample_rate == 44100
        assert cfg.pitch_min_hz == 65.41
        assert cfg.pitch_max_hz == 1046.5
        assert cfg.crepe_hop_ms == 10
        assert cfg.band_energy_min_hz == 60.0
        assert cfg.band_energy_max_hz == 4000.0
        assert cfg.pause_min_silence_ms == 400
        assert cfg.pause_threshold_pct == 5.0
        assert cfg.gap_lead_in_ms == 500
        assert cfg.linebreak_beat_offset == 4
        assert cfg.beat_resolution_multiplier == 2
        assert cfg.activity_quiet_confidence == 0.2
        assert cfg.activity_voiced_confidence == 0.5
        assert cfg.activity_noise_percentile == 0.9
        assert cfg.activity_noise_fallback_percentile == 0.1
        assert cfg.activity_signal_percentile == 0.5
        assert cfg.activity_signal_fallback_percentile == 0.75
        assert cfg.activity_threshold_ratio == 0.2
        assert cfg.note_min_confidence == 0.3
        assert cfg.note_fallback_confidence == 0.5
        assert cfg.note_dropout_gap_ms == 50
        assert cfg.note_smooth_window == 5
        assert cfg.note_pitch_tolerance == 1
        assert cfg.note_min_duration_ms == 60
        assert cfg.note_frame_step_ms == 10
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
                "transcription_backend": "whisperx",
                "whisper_model": "large",
                "whisperx_batch_size": 4,
                "whisperx_chunk_pause_ms": 1500,
                "whisperx_align_runs": 5,
                "sample_rate": 48000,
                "output_dir": "/custom/output",
            }, f)
            f.flush()
            cfg = load_config(f.name)
            assert cfg.transcription_backend == "whisperx"
            assert cfg.whisper_model == "large"
            assert cfg.whisperx_batch_size == 4
            assert cfg.whisperx_chunk_pause_ms == 1500
            assert cfg.whisperx_align_runs == 5
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

    def test_invalid_whisperx_options_use_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonc", delete=False) as f:
            json.dump({
                "whisperx_batch_size": 0,
                "whisperx_compute_type": "half-ish",
                "whisperx_interpolate_method": "guess",
                "whisperx_chunk_pause_ms": -1,
                "whisperx_align_runs": 4,
                "transcription_backend": "magic",
                "faster_whisper_compute_type": "half-ish",
            }, f)
            f.flush()
            cfg = load_config(f.name)

        assert cfg.whisperx_batch_size == 8
        assert cfg.whisperx_compute_type == "default"
        assert cfg.whisperx_interpolate_method == "nearest"
        assert cfg.whisperx_chunk_pause_ms == 1000
        assert cfg.whisperx_align_runs == 3
        assert cfg.transcription_backend == "faster-whisper"
        assert cfg.faster_whisper_compute_type == "auto"
