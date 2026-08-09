"""Lazy faster-whisper transcription adapter."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TranscriptionSettingsLike(Protocol):
    model: str
    language: str | None
    beam_size: int
    vad_filter: bool
    device: str
    compute_type: str


@dataclass(frozen=True, slots=True)
class TranscribedWord:
    word: str
    start: float
    end: float

    def to_dict(self) -> dict[str, str | float]:
        return {"word": self.word, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    words: tuple[TranscribedWord, ...]
    language: str

    def words_as_dicts(self) -> list[dict[str, str | float]]:
        return [word.to_dict() for word in self.words]


def _load_model_factory() -> Callable[..., Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Transcription requires the faster-whisper runtime dependency") from exc
    return WhisperModel


class FasterWhisperService:
    """Create a faster-whisper model on first use and reuse it until closed."""

    def __init__(
        self,
        settings: TranscriptionSettingsLike,
        *,
        model_factory_loader: Callable[[], Callable[..., Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory_loader = model_factory_loader or _load_model_factory
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            model_factory = self._model_factory_loader()
            self._model = model_factory(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Transcription audio does not exist: {path}")
        model = self._get_model()
        segments, info = model.transcribe(
            str(path),
            word_timestamps=True,
            initial_prompt=prompt.strip() if prompt and prompt.strip() else None,
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
        )
        words: list[TranscribedWord] = []
        for segment in segments:
            for raw_word in getattr(segment, "words", ()) or ():
                text = str(getattr(raw_word, "word", "")).strip()
                start = getattr(raw_word, "start", None)
                end = getattr(raw_word, "end", None)
                if not text or start is None or end is None:
                    continue
                words.append(TranscribedWord(text, float(start), float(end)))
        language = str(getattr(info, "language", None) or self.settings.language or "en")
        return TranscriptionResult(tuple(words), language)

    def close(self) -> None:
        """Release the cached model and any already-loaded CUDA allocator cache."""

        self._model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> FasterWhisperService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
