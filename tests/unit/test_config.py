from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ultrasongs.config import AppSettings, EffectiveSettingsSnapshot, load_settings


def test_default_settings_are_complete_and_immutable() -> None:
    settings = AppSettings()

    assert settings.server.host == "127.0.0.1"
    assert settings.transcription.model == "medium"
    assert settings.separation.model == "htdemucs"
    assert settings.ffmpeg.executable == "ffmpeg"

    with pytest.raises(ValidationError):
        settings.server.port = 9000  # type: ignore[misc]


def test_json_config_and_environment_precedence(tmp_path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"port": 8100},
                "transcription": {"model": "small", "beam_size": 3},
                "paths": {"temp_dir": "configured-tmp"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        config_path,
        environ={
            "ULTRASONGS_SERVER__PORT": "8200",
            "ULTRASONGS_TRANSCRIPTION__MODEL": "large-v3",
        },
    )

    assert settings.server.port == 8200
    assert settings.transcription.model == "large-v3"
    assert settings.transcription.beam_size == 3
    assert str(settings.paths.temp_dir) == "configured-tmp"


def test_toml_table_and_legacy_environment_mapping(tmp_path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        "[ultrasongs.tempo]\nfallback_bpm = 100.0\n",
        encoding="utf-8",
    )

    settings = load_settings(
        config_path,
        environ={"TMP_DIR": "legacy-tmp", "WHISPER_MODEL": "base"},
    )

    assert settings.tempo.fallback_bpm == 100.0
    assert str(settings.paths.temp_dir) == "legacy-tmp"
    assert settings.transcription.model == "base"


def test_safe_ui_overrides_take_final_precedence() -> None:
    startup = load_settings(
        environ={
            "ULTRASONGS_TRANSCRIPTION__MODEL": "small",
            "ULTRASONGS_TEMPO__FALLBACK_BPM": "110",
        }
    )

    effective = startup.apply_ui_overrides(
        {
            "transcription": {"model": "large-v3", "language": "HU"},
            "tempo.fallback_bpm": 128,
        }
    )

    assert startup.transcription.model == "small"
    assert effective.transcription.model == "large-v3"
    assert effective.transcription.language == "hu"
    assert effective.tempo.fallback_bpm == 128


@pytest.mark.parametrize(
    "override",
    [
        {"server.port": 9000},
        {"paths.temp_dir": "somewhere-else"},
        {"ffmpeg.executable": "custom-ffmpeg"},
        {"transcription.device": "cuda"},
        {"not_a_real_group.value": 1},
    ],
)
def test_startup_only_and_unknown_ui_overrides_are_rejected(override) -> None:
    with pytest.raises(ValueError, match="not UI-overridable"):
        AppSettings().apply_ui_overrides(override)


def test_ui_schema_is_generated_from_whitelist_metadata() -> None:
    settings = AppSettings()
    schema = settings.ui_override_schema()

    assert schema["transcription.model"]["default"] == "medium"
    assert schema["tempo.fallback_bpm"]["default"] == 120.0
    assert "validation.minimum_match_ratio" in schema
    assert "server.port" not in schema
    assert set(schema) == settings.ui_override_paths()


def test_invalid_models_thresholds_and_device_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_settings(environ={"ULTRASONGS_TRANSCRIPTION__MODEL": "imaginary"})
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "pitch": {"confidence_thresholds": [0.3, 0.5]},
            }
        )
    with pytest.raises(ValidationError, match="requires a CUDA device"):
        AppSettings.model_validate(
            {
                "transcription": {"device": "cpu", "compute_type": "float16"},
            }
        )


def test_effective_snapshot_round_trip_and_file_persistence(tmp_path) -> None:
    startup = AppSettings()
    snapshot = startup.effective_snapshot(
        {"transcription.model": "small", "report.include_pitch_frames": False}
    )

    restored = EffectiveSettingsSnapshot.from_json(snapshot.to_json())
    assert restored == snapshot
    assert restored.settings.transcription.model == "small"
    assert restored.settings.report.include_pitch_frames is False
    assert restored.ui_overrides == {
        "transcription.model": "small",
        "report.include_pitch_frames": False,
    }

    destination = snapshot.write(tmp_path / "run" / "effective-settings.json")
    assert EffectiveSettingsSnapshot.read(destination) == snapshot


def test_unknown_file_settings_and_missing_files_fail_loudly(tmp_path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text('{"server": {"mystery": true}}', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_path, environ={})
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.json", environ={})
