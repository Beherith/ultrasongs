"""Lazy Demucs source-separation adapter.

Heavy dependencies are imported only when :meth:`DemucsSeparationService.separate`
is called.  The Demucs model is intentionally scoped to one call so its GPU
memory is released before pitch and transcription stages begin.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[np.float32]


class SeparationSettingsLike(Protocol):
    model: str
    shifts: int
    overlap: float
    device: str


@dataclass(frozen=True, slots=True)
class SeparationResult:
    vocals: FloatAudio
    accompaniment: FloatAudio
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class _DemucsBackend:
    torch: Any
    torchaudio: Any
    get_model: Callable[[str], Any]
    apply_model: Callable[..., Any]


def _load_backend() -> _DemucsBackend:
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Demucs separation requires the 'gpu' runtime dependencies") from exc
    return _DemucsBackend(torch, torchaudio, get_model, apply_model)


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA separation was requested but CUDA is unavailable")
    return requested


class DemucsSeparationService:
    """Separate a mono track into vocals and summed accompaniment stems."""

    def __init__(
        self,
        settings: SeparationSettingsLike,
        *,
        backend_loader: Callable[[], _DemucsBackend] | None = None,
    ) -> None:
        self.settings = settings
        self._backend_loader = backend_loader or _load_backend

    def separate(self, audio: NDArray[np.floating[Any]], sample_rate_hz: int) -> SeparationResult:
        if audio.ndim != 1:
            raise ValueError("Demucs input must be a mono, one-dimensional audio array")
        if sample_rate_hz <= 0:
            raise ValueError("Audio sample rate must be positive")

        backend = self._backend_loader()
        torch = backend.torch
        device = _resolve_device(torch, self.settings.device)
        model = backend.get_model(self.settings.model)
        model.eval()
        model_sample_rate = int(getattr(model, "samplerate", 44_100))
        if device == "cuda":
            model = model.cuda()

        mono = None
        wave = None
        sources = None
        try:
            mono = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            if sample_rate_hz != model_sample_rate:
                mono = backend.torchaudio.functional.resample(
                    mono.unsqueeze(0), sample_rate_hz, model_sample_rate
                ).squeeze(0)
            wave = mono.unsqueeze(0).expand(2, -1).unsqueeze(0)
            if device == "cuda":
                wave = wave.cuda()
            with torch.no_grad():
                sources = backend.apply_model(
                    model,
                    wave,
                    device=device,
                    shifts=self.settings.shifts,
                    overlap=self.settings.overlap,
                )

            source_names = list(getattr(model, "sources", ()))
            vocals_index = source_names.index("vocals") if "vocals" in source_names else 3
            if vocals_index >= sources.shape[1]:
                raise RuntimeError("Demucs output does not contain a vocals stem")
            vocals = sources[0, vocals_index].mean(0).cpu().numpy().astype(np.float32)
            accompaniment_stems = [
                sources[0, index] for index in range(sources.shape[1]) if index != vocals_index
            ]
            accompaniment = (
                torch.stack(accompaniment_stems).sum(0).mean(0).cpu().numpy().astype(np.float32)
            )
            return SeparationResult(vocals, accompaniment, model_sample_rate)
        finally:
            del sources, wave, mono, model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
