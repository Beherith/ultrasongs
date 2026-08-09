"""Lazy torchcrepe inference service."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .pitch import attach_pitch_to_words


class PitchSettingsLike(Protocol):
    model: str
    sample_rate_hz: int
    hop_length: int
    min_frequency_hz: float
    max_frequency_hz: float
    batch_size: int
    confidence_thresholds: tuple[float, ...]
    device: str


@dataclass(frozen=True, slots=True)
class PitchTrack:
    times: NDArray[np.float32]
    frequencies: NDArray[np.float32]
    confidences: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _PitchBackend:
    torch: Any
    torchaudio: Any
    torchcrepe: Any


def _load_backend() -> _PitchBackend:
    try:
        import torch
        import torchaudio
        import torchcrepe
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Pitch detection requires torch, torchaudio, and torchcrepe") from exc
    return _PitchBackend(torch, torchaudio, torchcrepe)


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA pitch detection was requested but CUDA is unavailable")
    return requested


class TorchCrepePitchService:
    """Estimate pitch once over a complete vocals track."""

    def __init__(
        self,
        settings: PitchSettingsLike,
        *,
        backend_loader: Callable[[], _PitchBackend] | None = None,
    ) -> None:
        self.settings = settings
        self._backend_loader = backend_loader or _load_backend

    def analyze(self, vocals: NDArray[np.floating[Any]], sample_rate_hz: int) -> PitchTrack:
        if vocals.ndim != 1:
            raise ValueError("Pitch input must be a mono, one-dimensional audio array")
        if sample_rate_hz <= 0:
            raise ValueError("Audio sample rate must be positive")

        backend = self._backend_loader()
        torch = backend.torch
        device = _resolve_device(torch, self.settings.device)
        audio_tensor = torch.from_numpy(np.asarray(vocals, dtype=np.float32))
        pitch = None
        periodicity = None
        try:
            if sample_rate_hz != self.settings.sample_rate_hz:
                audio_tensor = backend.torchaudio.functional.resample(
                    audio_tensor.unsqueeze(0),
                    sample_rate_hz,
                    self.settings.sample_rate_hz,
                ).squeeze(0)
            pitch, periodicity = backend.torchcrepe.predict(
                audio_tensor.unsqueeze(0),
                sample_rate=self.settings.sample_rate_hz,
                hop_length=self.settings.hop_length,
                fmin=self.settings.min_frequency_hz,
                fmax=self.settings.max_frequency_hz,
                model=self.settings.model,
                decoder=backend.torchcrepe.decode.viterbi,
                return_periodicity=True,
                batch_size=self.settings.batch_size,
                device=device,
                pad=True,
            )
            frequencies = np.asarray(pitch[0].cpu().numpy(), dtype=np.float32)
            confidences = np.asarray(periodicity[0].cpu().numpy(), dtype=np.float32)
            times = np.arange(len(frequencies), dtype=np.float32) * (
                self.settings.hop_length / self.settings.sample_rate_hz
            )
            return PitchTrack(times, frequencies, confidences)
        finally:
            del periodicity, pitch, audio_tensor
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    def attach_to_words(
        self,
        words: Iterable[Mapping[str, Any]],
        track: PitchTrack,
    ) -> list[dict[str, Any]]:
        """Attach aggregate MIDI notes and pitch frames using shared pure logic."""

        return attach_pitch_to_words(
            words,
            track.times,
            track.frequencies,
            track.confidences,
            confidence_thresholds=self.settings.confidence_thresholds,
            frame_minimum_confidence=min(self.settings.confidence_thresholds),
        )
