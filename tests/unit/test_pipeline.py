from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ultrasongs.config import AppSettings, PathSettings
from ultrasongs.domain.models import PipelineStageStatus
from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.processing.pipeline import (
    PipelineRunError,
    PipelineRunner,
    ValidationInput,
)
from ultrasongs.processing.pitch_detection import PitchTrack
from ultrasongs.processing.separation import SeparationResult
from ultrasongs.processing.tempo import TempoResult
from ultrasongs.storage import ArtifactRepository, ProjectRepository

REFERENCE = b"""#TITLE:Reference
#ARTIST:Tester
#MP3:reference.mp3
#BPM:120
#GAP:0

: 0 4 69 Hel
: 4 4 71 lo
- 9
: 12 4 72 world
E
"""


class FakeMediaService:
    def normalize_audio(self, source: Path, destination: Path) -> Path:
        destination.write_bytes(b"normalized:" + source.read_bytes())
        return destination


class FakeSeparationService:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def separate(self, audio: np.ndarray[Any, Any], sample_rate: int) -> SeparationResult:
        if self.fail:
            raise RuntimeError("demucs exploded")
        assert sample_rate == 16_000
        return SeparationResult(
            vocals=np.asarray(audio, dtype=np.float32),
            accompaniment=np.asarray(audio * 0.5, dtype=np.float32),
            sample_rate_hz=sample_rate,
        )


class FakePitchService:
    def analyze(self, vocals: np.ndarray[Any, Any], sample_rate: int) -> PitchTrack:
        assert len(vocals) == 16_000
        assert sample_rate == 16_000
        return PitchTrack(
            times=np.asarray([0.1, 0.5, 1.1, 1.5], dtype=np.float32),
            frequencies=np.asarray([440, 440, 523.25, 523.25], dtype=np.float32),
            confidences=np.asarray([0.9, 0.8, 0.9, 0.8], dtype=np.float32),
        )

    def attach_to_words(
        self, words: list[dict[str, Any]], track: PitchTrack
    ) -> list[dict[str, Any]]:
        assert len(track.times) == 4
        return [
            {
                **word,
                "midi": 69 if index == 0 else 72,
                "pitchFrames": [
                    {
                        "time": word["start"],
                        "midi": 69 if index == 0 else 72,
                        "confidence": 0.9,
                    }
                ],
            }
            for index, word in enumerate(words)
        ]


class FakeTranscriptionService:
    language = "en"

    def __init__(self) -> None:
        self.closed = False

    def transcribe(self, path: Path, *, prompt: str | None = None) -> Any:
        assert path.name == "vocals.wav"
        assert prompt == "Hello world"
        return SimpleNamespace(
            language="en",
            words_as_dicts=lambda: [
                {"word": "hello", "start": 0.1, "end": 0.8},
                {"word": "world", "start": 1.1, "end": 1.8},
            ],
        )

    def close(self) -> None:
        self.closed = True


class FakeTempoService:
    def detect(self, path: Path) -> TempoResult:
        assert path.name == "normalized.mp3"
        return TempoResult(120.0, (0.0, 0.5, 1.0), False)


def make_runner(
    tmp_path: Path, *, separation_fails: bool = False
) -> tuple[PipelineRunner, ProjectRepository, ArtifactRepository, FakeTranscriptionService]:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    settings = AppSettings(paths=PathSettings(temp_dir=work_root))
    projects = ProjectRepository(data_root)
    artifacts = ArtifactRepository(data_root, projects=projects)
    transcriber = FakeTranscriptionService()

    def audio_loader(path: Path) -> tuple[np.ndarray[Any, Any], int]:
        assert path.read_bytes().startswith(b"normalized:")
        return np.linspace(-0.2, 0.2, 16_000, dtype=np.float32), 16_000

    def wav_writer(path: Path, audio: np.ndarray[Any, Any], sample_rate: int) -> None:
        assert sample_rate == 16_000
        path.write_bytes(np.asarray(audio, dtype=np.float32).tobytes())

    similarity = SimilarityResult(
        reference_notes=3,
        candidate_notes=2,
        matched_notes=2,
        matched_ratio=1.0,
        reference_coverage=2 / 3,
        candidate_coverage=1.0,
        timing_rmse_ms=10.0,
        duration_rmse_ms=20.0,
        pitch_distance_semitones=0.5,
        timing_median_error_ms=10.0,
        timing_max_error_ms=15.0,
        duration_median_error_ms=20.0,
        duration_max_error_ms=25.0,
        pitch_median_distance_semitones=0.0,
        pitch_max_distance_semitones=1.0,
    )
    runner = PipelineRunner(
        settings,
        projects,
        artifacts,
        media_factory=lambda _: FakeMediaService(),
        separation_factory=lambda _: FakeSeparationService(fail=separation_fails),
        pitch_factory=lambda _: FakePitchService(),
        transcription_factory=lambda _: transcriber,
        tempo_factory=lambda _: FakeTempoService(),
        audio_loader=audio_loader,
        wav_writer=wav_writer,
        pause_detector=lambda *args, **kwargs: [{"start": 0.9, "end": 1.0}],
        scorer=lambda reference, candidate: similarity,
        report_builder=lambda **kwargs: (
            f"<html>{kwargs['title']}:{kwargs['similarity'].matched_notes}</html>"
        ),
    )
    return runner, projects, artifacts, transcriber


def test_pipeline_runs_end_to_end_and_persists_validation_artifacts(tmp_path: Path) -> None:
    runner, projects, artifacts, transcriber = make_runner(tmp_path)
    project = projects.create(title="Hello", artist="Tester")
    source = tmp_path / "upload.mp3"
    source.write_bytes(b"uploaded song")

    result = runner.run(
        project_id=project.project_id,
        source_path=source,
        title="Hello Song",
        artist="Tester",
        lyrics="Hello world",
        ui_overrides={"transcription": {"model": "small"}},
        validation=ValidationInput(REFERENCE, "reference.txt"),
    )

    manifest = artifacts.get_manifest(project.project_id, result.run_id)
    assert result.manifest == manifest
    assert result.similarity is not None
    assert result.validation_outcome is not None
    assert result.report_artifact_id is not None
    assert transcriber.closed is True
    assert projects.get(project.project_id).latest_run_id == result.run_id
    assert projects.get(project.project_id).reference_artifact_id is not None

    stage_names = [stage.stage for stage in manifest.stages]
    assert stage_names == [
        "intake",
        "normalize_audio",
        "load_audio",
        "separate",
        "pitch",
        "pauses",
        "transcribe",
        "tempo",
        "align",
        "generate",
        "package",
        "score",
        "report",
    ]
    assert all(stage.status is PipelineStageStatus.SUCCEEDED for stage in manifest.stages)
    kinds = {artifact.kind for artifact in manifest.artifacts}
    assert kinds == {
        "effective_config",
        "run_inputs",
        "source_media",
        "reference_ultrastar",
        "normalized_audio",
        "vocals",
        "accompaniment",
        "pitch",
        "pauses",
        "transcription",
        "tempo",
        "alignment",
        "candidate_ultrastar",
        "export_zip",
        "similarity",
        "validation_outcome",
        "pipeline_report",
    }
    reference = next(item for item in manifest.artifacts if item.kind == "reference_ultrastar")
    assert (
        artifacts.read_bytes(project.project_id, result.run_id, reference.artifact_id) == REFERENCE
    )
    config_id = manifest.effective_config_artifact_id
    assert config_id is not None
    config = json.loads(artifacts.read_bytes(project.project_id, result.run_id, config_id))
    assert config["settings"]["transcription"]["model"] == "small"
    inputs = next(item for item in manifest.artifacts if item.kind == "run_inputs")
    submitted = json.loads(
        artifacts.read_bytes(project.project_id, result.run_id, inputs.artifact_id)
    )
    assert submitted["lyrics"] == "Hello world"
    assert not any((tmp_path / "work").iterdir())


def test_pipeline_records_failed_stage_and_cleans_temporary_files(tmp_path: Path) -> None:
    runner, projects, artifacts, _ = make_runner(tmp_path, separation_fails=True)
    project = projects.create()
    source = tmp_path / "upload.mp3"
    source.write_bytes(b"uploaded song")

    with pytest.raises(PipelineRunError, match="separate: demucs exploded") as captured:
        runner.run(
            project_id=project.project_id,
            source_path=source,
            title="Failure",
            artist="Tester",
            lyrics="Hello world",
        )

    manifest = artifacts.get_manifest(project.project_id, captured.value.run_id)
    assert [stage.status for stage in manifest.stages] == [
        PipelineStageStatus.SUCCEEDED,
        PipelineStageStatus.SUCCEEDED,
        PipelineStageStatus.SUCCEEDED,
        PipelineStageStatus.FAILED,
    ]
    assert manifest.stages[-1].stage == "separate"
    assert manifest.stages[-1].message == "demucs exploded"
    assert not any((tmp_path / "work").iterdir())


def test_pipeline_rejects_bad_reference_before_creating_run(tmp_path: Path) -> None:
    runner, projects, artifacts, _ = make_runner(tmp_path)
    project = projects.create()
    source = tmp_path / "upload.mp3"
    source.write_bytes(b"uploaded song")

    with pytest.raises(ValueError, match="UTF-8"):
        runner.run(
            project_id=project.project_id,
            source_path=source,
            title="Song",
            artist="Artist",
            lyrics="Hello",
            validation=ValidationInput(b"\xff\xfe\x00", "bad.txt"),
        )

    assert artifacts.list_manifests(project.project_id) == []
