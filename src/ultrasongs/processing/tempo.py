"""Configured tempo detection with the legacy 120 BPM fallback behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ultrasongs.config import TempoSettings


@dataclass(frozen=True, slots=True)
class TempoResult:
    bpm: float
    beat_times: tuple[float, ...] = ()
    used_fallback: bool = False
    warning: str | None = None


class TempoService:
    def __init__(
        self,
        settings: TempoSettings,
        *,
        analyzer: Callable[[Path], tuple[float, tuple[float, ...]]] | None = None,
    ) -> None:
        self.settings = settings
        self._analyzer = analyzer or self._analyze_with_librosa

    def detect(self, audio_path: str | Path) -> TempoResult:
        source = Path(audio_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")
        try:
            bpm, beat_times = self._analyzer(source)
            bpm = float(np.asarray(bpm).reshape(-1)[0])
            if not np.isfinite(bpm) or bpm <= 0:
                raise ValueError(f"invalid detected BPM: {bpm}")
            while bpm < self.settings.minimum_bpm:
                bpm *= 2
            while bpm > self.settings.maximum_bpm:
                bpm /= 2
            return TempoResult(bpm=bpm, beat_times=tuple(float(value) for value in beat_times))
        except Exception as exc:
            return TempoResult(
                bpm=self.settings.fallback_bpm,
                used_fallback=True,
                warning=f"Tempo detection failed; using configured fallback: {exc}",
            )

    @staticmethod
    def _analyze_with_librosa(source: Path) -> tuple[float, tuple[float, ...]]:
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - optional runtime extra
            raise RuntimeError("Tempo detection requires the 'runtime' dependency group") from exc

        audio, sample_rate = librosa.load(source, sr=44_100, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        return float(np.asarray(tempo).reshape(-1)[0]), tuple(float(value) for value in beat_times)
