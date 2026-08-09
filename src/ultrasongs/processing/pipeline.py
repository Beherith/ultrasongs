"""Automatic, artifact-producing UltraSongs pipeline orchestration."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ultrasongs.config import AppSettings
from ultrasongs.domain.alignment import AlignmentResult, align_lyrics_with_debug
from ultrasongs.domain.models import (
    ArtifactManifest,
    ArtifactRecord,
    PipelineStageResult,
    PipelineStageStatus,
    utc_now_iso,
)
from ultrasongs.domain.reporting import build_pipeline_report
from ultrasongs.domain.scoring import SimilarityResult, compare_songs
from ultrasongs.domain.ultrastar import (
    GenerationResult,
    build_export_zip,
    generate_song_from_alignment,
    safe_filename,
    write_ultrastar_text,
)
from ultrasongs.domain.validation import (
    ValidationOutcome,
    evaluate_similarity,
    inspect_reference_bytes,
)
from ultrasongs.storage import ArtifactRepository, ProjectRepository

from .media import MediaService, is_video_media
from .pauses import detect_pauses
from .pitch_detection import PitchTrack, TorchCrepePitchService
from .separation import DemucsSeparationService, SeparationResult
from .tempo import TempoResult, TempoService
from .transcription import FasterWhisperService
from .whisperx import WhisperXService

JsonMapping = Mapping[str, Any]
AudioArray = NDArray[np.floating[Any]]


class TranscriptionResultLike(Protocol):
    language: str

    def words_as_dicts(self) -> list[dict[str, Any]]: ...


class TranscriptionServiceLike(Protocol):
    def transcribe(
        self, audio_path: str | Path, *, prompt: str | None = None
    ) -> TranscriptionResultLike: ...


@dataclass(frozen=True, slots=True)
class ValidationInput:
    """Exact reference UltraStar bytes and their display-only original name."""

    content: bytes
    original_name: str = "reference.txt"

    @classmethod
    def from_path(cls, path: str | Path) -> ValidationInput:
        source = Path(path)
        return cls(source.read_bytes(), source.name)


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    project_id: str
    run_id: str
    manifest: ArtifactManifest
    candidate_artifact_id: str
    archive_artifact_id: str
    report_artifact_id: str | None = None
    similarity: SimilarityResult | None = None
    validation_outcome: ValidationOutcome | None = None


class PipelineRunError(RuntimeError):
    """A named pipeline stage failed after its state was persisted."""

    def __init__(self, run_id: str, stage: str, cause: Exception) -> None:
        super().__init__(f"Pipeline run {run_id} failed during {stage}: {cause}")
        self.run_id = run_id
        self.stage = stage
        self.cause = cause


@dataclass(slots=True)
class _StageState:
    name: str
    started_at: str
    artifact_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


def _default_media_factory(settings: AppSettings) -> MediaService:
    return MediaService(settings.ffmpeg)


def _default_separation_factory(settings: AppSettings) -> DemucsSeparationService:
    return DemucsSeparationService(settings.separation)


def _default_pitch_factory(settings: AppSettings) -> TorchCrepePitchService:
    return TorchCrepePitchService(settings.pitch)


def _default_transcription_factory(settings: AppSettings) -> TranscriptionServiceLike:
    if settings.transcription.engine == "whisperx":
        return WhisperXService(settings.whisperx, settings.transcription)
    return FasterWhisperService(settings.transcription)


def _default_tempo_factory(settings: AppSettings) -> TempoService:
    return TempoService(settings.tempo)


def _load_mono_audio(path: Path) -> tuple[AudioArray, int]:
    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Loading normalized audio requires soundfile") from exc
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1:
        raise ValueError("Normalized audio could not be converted to mono")
    return values, int(sample_rate)


def _write_wav(path: Path, audio: AudioArray, sample_rate_hz: int) -> None:
    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Writing separated WAV stems requires soundfile") from exc
    soundfile.write(path, np.asarray(audio, dtype=np.float32), sample_rate_hz, format="WAV")


class PipelineRunner:
    """Run the complete non-editor pipeline and persist each observable stage."""

    def __init__(
        self,
        settings: AppSettings,
        projects: ProjectRepository,
        artifacts: ArtifactRepository,
        *,
        media_factory: Callable[[AppSettings], Any] = _default_media_factory,
        separation_factory: Callable[[AppSettings], Any] = _default_separation_factory,
        pitch_factory: Callable[[AppSettings], Any] = _default_pitch_factory,
        transcription_factory: Callable[[AppSettings], Any] = _default_transcription_factory,
        tempo_factory: Callable[[AppSettings], Any] = _default_tempo_factory,
        audio_loader: Callable[[Path], tuple[AudioArray, int]] = _load_mono_audio,
        wav_writer: Callable[[Path, AudioArray, int], None] = _write_wav,
        pause_detector: Callable[..., list[dict[str, float]]] = detect_pauses,
        aligner: Callable[..., AlignmentResult] = align_lyrics_with_debug,
        generator: Callable[..., GenerationResult] = generate_song_from_alignment,
        text_writer: Callable[[Any], str] = write_ultrastar_text,
        archive_builder: Callable[..., bytes] = build_export_zip,
        scorer: Callable[[Any, Any], SimilarityResult] | None = compare_songs,
        report_builder: Callable[..., str] | None = build_pipeline_report,
    ) -> None:
        self.settings = settings
        self.projects = projects
        self.artifacts = artifacts
        self._media_factory = media_factory
        self._separation_factory = separation_factory
        self._pitch_factory = pitch_factory
        self._transcription_factory = transcription_factory
        self._tempo_factory = tempo_factory
        self._audio_loader = audio_loader
        self._wav_writer = wav_writer
        self._pause_detector = pause_detector
        self._aligner = aligner
        self._generator = generator
        self._text_writer = text_writer
        self._archive_builder = archive_builder
        self._scorer = scorer
        self._report_builder = report_builder

    def run(
        self,
        *,
        project_id: str,
        source_path: str | Path,
        video_path: str | Path | None = None,
        title: str,
        artist: str,
        lyrics: str,
        ui_overrides: JsonMapping | None = None,
        validation: ValidationInput | None = None,
    ) -> PipelineRunResult:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Uploaded source does not exist: {source}")
        explicit_video = Path(video_path).resolve() if video_path is not None else None
        if explicit_video is not None and not explicit_video.is_file():
            raise FileNotFoundError(f"Uploaded video does not exist: {explicit_video}")
        video_source = explicit_video or (source if is_video_media(source) else None)
        if not lyrics.strip():
            raise ValueError("Lyrics cannot be empty")

        # Fail malformed references before any model or media work begins.
        reference_inspection = (
            inspect_reference_bytes(validation.content) if validation is not None else None
        )

        snapshot = self.settings.effective_snapshot(ui_overrides)
        effective = snapshot.settings
        snapshot_payload = snapshot.model_dump(mode="json")
        manifest = self.artifacts.create_manifest(project_id, effective_config=snapshot_payload)
        run_id = manifest.run_id
        self.projects.update(
            project_id,
            title=title,
            artist=artist,
            latest_run_id=run_id,
        )
        current_stage = "intake"

        temp_root = Path(effective.paths.temp_dir).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"{run_id}_", dir=temp_root) as temporary:
                work = Path(temporary)
                with self._stage(project_id, run_id, "intake") as stage:
                    inputs_record = self._json_artifact(
                        project_id,
                        run_id,
                        {
                            "title": title,
                            "artist": artist,
                            "lyrics": lyrics,
                            "source_original_name": source.name,
                            "video_original_name": (
                                video_source.name if video_source is not None else None
                            ),
                            "validation_mode": validation is not None,
                            "reference_original_name": (
                                validation.original_name if validation is not None else None
                            ),
                        },
                        "run_inputs",
                        "run-inputs.json",
                    )
                    stage.artifact_ids.append(inputs_record.artifact_id)
                    upload = self.artifacts.register_file(
                        project_id,
                        run_id,
                        source,
                        kind="source_media",
                        original_name=source.name,
                    )
                    stage.artifact_ids.append(upload.artifact_id)
                    if explicit_video is not None:
                        video_record = self.artifacts.register_file(
                            project_id,
                            run_id,
                            explicit_video,
                            kind="source_video",
                            original_name=explicit_video.name,
                        )
                        stage.artifact_ids.append(video_record.artifact_id)
                    if validation is not None:
                        reference_record = self.artifacts.register_reference(
                            project_id,
                            run_id,
                            validation.content,
                            original_name=validation.original_name,
                        )
                        self.projects.update(
                            project_id, reference_artifact_id=reference_record.artifact_id
                        )
                        stage.artifact_ids.append(reference_record.artifact_id)

                current_stage = "normalize_audio"
                normalized_path = work / "normalized.mp3"
                with self._stage(project_id, run_id, current_stage) as stage:
                    media = self._media_factory(effective)
                    media.normalize_audio(source, normalized_path)
                    normalized_record = self.artifacts.register_file(
                        project_id,
                        run_id,
                        normalized_path,
                        kind="normalized_audio",
                        original_name=f"{safe_filename(title)}.mp3",
                        media_type="audio/mpeg",
                    )
                    stage.artifact_ids.append(normalized_record.artifact_id)

                current_stage = "load_audio"
                with self._stage(project_id, run_id, current_stage) as stage:
                    audio, sample_rate_hz = self._audio_loader(normalized_path)
                    stage.metrics.update(
                        sample_rate_hz=sample_rate_hz,
                        samples=len(audio),
                        duration_seconds=(len(audio) / sample_rate_hz),
                    )

                current_stage = "separate"
                vocals_path = work / "vocals.wav"
                accompaniment_path = work / "accompaniment.wav"
                vocals_mp3_path = work / "vocals.mp3"
                accompaniment_mp3_path = work / "accompaniment.mp3"
                with self._stage(project_id, run_id, current_stage) as stage:
                    separated: SeparationResult = self._separation_factory(effective).separate(
                        audio, sample_rate_hz
                    )
                    self._wav_writer(vocals_path, separated.vocals, separated.sample_rate_hz)
                    self._wav_writer(
                        accompaniment_path,
                        separated.accompaniment,
                        separated.sample_rate_hz,
                    )
                    media.normalize_audio(vocals_path, vocals_mp3_path)
                    media.normalize_audio(accompaniment_path, accompaniment_mp3_path)
                    for path, kind in (
                        (vocals_mp3_path, "vocals"),
                        (accompaniment_mp3_path, "accompaniment"),
                    ):
                        record = self.artifacts.register_file(
                            project_id,
                            run_id,
                            path,
                            kind=kind,
                            original_name=path.name,
                            media_type="audio/mpeg",
                        )
                        stage.artifact_ids.append(record.artifact_id)
                    stage.metrics["sample_rate_hz"] = separated.sample_rate_hz

                current_stage = "pitch"
                with self._stage(project_id, run_id, current_stage) as stage:
                    pitch_service = self._pitch_factory(effective)
                    pitch: PitchTrack = pitch_service.analyze(
                        separated.vocals, separated.sample_rate_hz
                    )
                    pitch_payload = {
                        "times": pitch.times.tolist(),
                        "frequencies": pitch.frequencies.tolist(),
                        "confidences": pitch.confidences.tolist(),
                    }
                    pitch_record = self._json_artifact(
                        project_id, run_id, pitch_payload, "pitch", "pitch.json"
                    )
                    stage.artifact_ids.append(pitch_record.artifact_id)
                    stage.metrics["frames"] = len(pitch.times)

                current_stage = "pauses"
                with self._stage(project_id, run_id, current_stage) as stage:
                    pauses = self._pause_detector(
                        separated.vocals,
                        separated.sample_rate_hz,
                        frame_ms=effective.pauses.frame_milliseconds,
                        hop_ms=effective.pauses.hop_milliseconds,
                        minimum_silence_ms=effective.pauses.minimum_duration_milliseconds,
                        threshold_ratio=effective.pauses.threshold_ratio,
                    )
                    pause_record = self._json_artifact(
                        project_id,
                        run_id,
                        {"pauses": pauses},
                        "pauses",
                        "pauses.json",
                    )
                    stage.artifact_ids.append(pause_record.artifact_id)
                    stage.metrics["pauses"] = len(pauses)

                current_stage = "transcribe"
                with self._stage(project_id, run_id, current_stage) as stage:
                    transcription_service = self._transcription_factory(effective)
                    try:
                        transcription = transcription_service.transcribe(
                            vocals_path, prompt=lyrics.strip()
                        )
                    finally:
                        close = getattr(transcription_service, "close", None)
                        if close is not None:
                            close()
                    raw_words = transcription.words_as_dicts()
                    words = pitch_service.attach_to_words(raw_words, pitch)
                    transcription_payload = {
                        "language": transcription.language,
                        "words": words,
                        "pauses": pauses,
                    }
                    transcription_record = self._json_artifact(
                        project_id,
                        run_id,
                        transcription_payload,
                        "transcription",
                        "transcription.json",
                    )
                    stage.artifact_ids.append(transcription_record.artifact_id)
                    stage.metrics["words"] = len(words)
                    stage.metrics["language"] = transcription.language

                current_stage = "tempo"
                with self._stage(project_id, run_id, current_stage) as stage:
                    tempo: TempoResult = self._tempo_factory(effective).detect(normalized_path)
                    tempo_record = self._json_artifact(
                        project_id,
                        run_id,
                        asdict(tempo),
                        "tempo",
                        "tempo.json",
                    )
                    stage.artifact_ids.append(tempo_record.artifact_id)
                    stage.metrics.update(bpm=tempo.bpm, used_fallback=tempo.used_fallback)
                    stage.message = tempo.warning

                current_stage = "align"
                with self._stage(project_id, run_id, current_stage) as stage:
                    alignment = self._aligner(
                        lyrics,
                        words,
                        transcription.language,
                        pauses,
                        match_score=effective.alignment.match_score,
                        gap_open_penalty=effective.alignment.gap_open_penalty,
                        gap_extend_penalty=effective.alignment.gap_extend_penalty,
                    )
                    alignment_record = self._json_artifact(
                        project_id,
                        run_id,
                        asdict(alignment),
                        "alignment",
                        "alignment.json",
                    )
                    stage.artifact_ids.append(alignment_record.artifact_id)
                    stage.metrics["syllables"] = len(alignment.syllables)

                current_stage = "generate"
                with self._stage(project_id, run_id, current_stage) as stage:
                    generated = self._generator(
                        alignment.syllables,
                        title=title,
                        artist=artist,
                        mp3_filename=f"{safe_filename(title)}.mp3",
                        bpm=tempo.bpm,
                        video_filename=(
                            safe_filename(video_source.name)
                            if video_source is not None
                            else None
                        ),
                    )
                    candidate_text = self._text_writer(generated.song)
                    candidate_record = self.artifacts.register_bytes(
                        project_id,
                        run_id,
                        candidate_text.encode(effective.export.text_encoding),
                        kind="candidate_ultrastar",
                        original_name=f"{safe_filename(title)}.txt",
                        media_type="text/plain",
                    )
                    stage.artifact_ids.append(candidate_record.artifact_id)
                    stage.metrics.update(
                        notes=len(generated.song.notes),
                        bpm=generated.bpm,
                        gap_ms=generated.gap_ms,
                    )

                current_stage = "package"
                with self._stage(project_id, run_id, current_stage) as stage:
                    archive_bytes = self._archive_builder(
                        generated.song,
                        audio_path=normalized_path,
                        text_filename=f"{safe_filename(title)}.txt",
                        text_encoding=effective.export.text_encoding,
                        include_audio=effective.export.include_audio_in_zip,
                        video_path=video_source,
                        vocals_path=vocals_mp3_path,
                        accompaniment_path=accompaniment_mp3_path,
                    )
                    archive_record = self.artifacts.register_bytes(
                        project_id,
                        run_id,
                        archive_bytes,
                        kind="export_zip",
                        original_name=f"{safe_filename(title)}.zip",
                        media_type="application/zip",
                    )
                    stage.artifact_ids.append(archive_record.artifact_id)

                similarity = None
                validation_outcome = None
                reference_song = (
                    reference_inspection.song if reference_inspection is not None else None
                )
                if validation is not None:
                    current_stage = "score"
                    with self._stage(project_id, run_id, current_stage) as stage:
                        if self._scorer is not None:
                            similarity = self._scorer(reference_song, generated.song)
                            score_record = self._json_artifact(
                                project_id,
                                run_id,
                                similarity.to_dict(),
                                "similarity",
                                "similarity.json",
                            )
                            stage.artifact_ids.append(score_record.artifact_id)
                            validation_outcome = evaluate_similarity(
                                similarity, effective.validation
                            )
                            outcome_record = self._json_artifact(
                                project_id,
                                run_id,
                                validation_outcome.to_dict(),
                                "validation_outcome",
                                "validation-outcome.json",
                            )
                            stage.artifact_ids.append(outcome_record.artifact_id)
                            stage.metrics.update(
                                matched_notes=similarity.matched_notes,
                                matched_ratio=similarity.matched_ratio,
                                reference_coverage=similarity.reference_coverage,
                                passed=validation_outcome.passed,
                                failures=len(validation_outcome.failures),
                            )

                report_record = None
                if self._report_builder is not None:
                    current_stage = "report"
                    with self._stage(project_id, run_id, current_stage) as stage:
                        report = self._report_builder(
                            candidate=generated.song,
                            reference=reference_song,
                            transcription=transcription_payload,
                            similarity=similarity,
                            validation_outcome=validation_outcome,
                            title=title,
                            effective_config=snapshot_payload,
                            report_options=effective.report.model_dump(mode="json"),
                        )
                        report_record = self.artifacts.register_bytes(
                            project_id,
                            run_id,
                            report.encode("utf-8"),
                            kind="pipeline_report",
                            original_name=f"{safe_filename(title)}.html",
                            media_type="text/html",
                        )
                        stage.artifact_ids.append(report_record.artifact_id)

            return PipelineRunResult(
                project_id=project_id,
                run_id=run_id,
                manifest=self.artifacts.get_manifest(project_id, run_id),
                candidate_artifact_id=candidate_record.artifact_id,
                archive_artifact_id=archive_record.artifact_id,
                report_artifact_id=(
                    report_record.artifact_id if report_record is not None else None
                ),
                similarity=similarity,
                validation_outcome=validation_outcome,
            )
        except Exception as exc:
            if isinstance(exc, PipelineRunError):
                raise
            raise PipelineRunError(run_id, current_stage, exc) from exc

    @contextmanager
    def _stage(self, project_id: str, run_id: str, name: str):
        state = _StageState(name, utc_now_iso())
        self.artifacts.record_stage(
            project_id,
            run_id,
            PipelineStageResult(
                stage=name,
                status=PipelineStageStatus.RUNNING,
                started_at=state.started_at,
            ),
        )
        try:
            yield state
        except Exception as exc:
            self.artifacts.record_stage(
                project_id,
                run_id,
                PipelineStageResult(
                    stage=name,
                    status=PipelineStageStatus.FAILED,
                    started_at=state.started_at,
                    finished_at=utc_now_iso(),
                    artifact_ids=tuple(state.artifact_ids),
                    metrics=state.metrics,
                    message=str(exc),
                ),
            )
            raise
        else:
            self.artifacts.record_stage(
                project_id,
                run_id,
                PipelineStageResult(
                    stage=name,
                    status=PipelineStageStatus.SUCCEEDED,
                    started_at=state.started_at,
                    finished_at=utc_now_iso(),
                    artifact_ids=tuple(state.artifact_ids),
                    metrics=state.metrics,
                    message=state.message,
                ),
            )

    def _json_artifact(
        self,
        project_id: str,
        run_id: str,
        payload: JsonMapping,
        kind: str,
        original_name: str,
    ) -> ArtifactRecord:
        content = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        return self.artifacts.register_bytes(
            project_id,
            run_id,
            content,
            kind=kind,
            original_name=original_name,
            media_type="application/json",
        )
