"""WhisperX subprocess adapter and isolated worker entry point."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class WhisperXSettingsLike(Protocol):
    model: str
    batch_size: int
    device: str
    compute_type: str
    python_executable: str


class LanguageSettingsLike(Protocol):
    language: str | None


@dataclass(frozen=True, slots=True)
class WhisperXWord:
    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class WhisperXResult:
    words: tuple[WhisperXWord, ...]
    language: str

    def words_as_dicts(self) -> list[dict[str, str | float]]:
        return [{"word": word.word, "start": word.start, "end": word.end} for word in self.words]


class WhisperXService:
    """Run WhisperX out of process to isolate its dependency and GPU lifecycle."""

    def __init__(
        self,
        settings: WhisperXSettingsLike,
        language_settings: LanguageSettingsLike,
        *,
        worker_path: str | Path | None = None,
        working_directory: str | Path | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.language_settings = language_settings
        self.worker_path = Path(worker_path or __file__).resolve()
        self.working_directory = Path(working_directory or Path.cwd()).resolve()
        self._runner = runner or subprocess.run

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str | None = None,
    ) -> WhisperXResult:
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"WhisperX audio does not exist: {audio_path}")
        if not self.worker_path.is_file():
            raise FileNotFoundError(f"WhisperX worker does not exist: {self.worker_path}")
        python_executable = self._resolve_python_executable()
        command = [
            python_executable,
            str(self.worker_path),
            "--worker",
            "--audio",
            str(audio_path),
            "--model",
            self.settings.model,
            "--batch-size",
            str(self.settings.batch_size),
            "--device",
            self.settings.device,
            "--compute-type",
            self.settings.compute_type,
        ]
        language = self.language_settings.language
        if language:
            command.extend(["--language", language])
        if prompt and prompt.strip():
            command.extend(["--prompt", prompt.strip()])

        completed = self._runner(
            command,
            cwd=self.working_directory,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = str(completed.stderr or "").strip()
            raise RuntimeError(
                f"WhisperX failed with code {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("WhisperX returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("WhisperX returned a non-object JSON payload")
        words = tuple(_words_from_payload(payload.get("words", ())))
        detected_language = str(payload.get("language") or language or "en")
        return WhisperXResult(words, detected_language)

    def _resolve_python_executable(self) -> str:
        configured = self.settings.python_executable
        candidate = Path(configured)
        if candidate.is_absolute() or candidate.parent != Path("."):
            if not candidate.is_absolute():
                candidate = self.working_directory / candidate
            if not candidate.is_file():
                raise FileNotFoundError(f"WhisperX Python does not exist: {candidate}")
            return str(candidate.resolve())
        resolved = shutil.which(configured)
        if resolved is None:
            raise FileNotFoundError(f"WhisperX Python command was not found: {configured}")
        return resolved


def _words_from_payload(raw_words: Any) -> Sequence[WhisperXWord]:
    if not isinstance(raw_words, list | tuple):
        raise RuntimeError("WhisperX words payload is not a list")
    words: list[WhisperXWord] = []
    for value in raw_words:
        if not isinstance(value, dict):
            continue
        text = str(value.get("word", "")).strip()
        if not text or value.get("start") is None or value.get("end") is None:
            continue
        words.append(WhisperXWord(text, float(value["start"]), float(value["end"])))
    return words


def _worker_main(arguments: argparse.Namespace) -> None:
    try:
        # Direct script execution puts this file's directory first on sys.path.
        # Remove it before importing the top-level dependency, otherwise this
        # worker module can shadow the external package with the same name.
        worker_directory = Path(__file__).resolve().parent
        if __name__ == "__main__":
            sys.path = [
                entry for entry in sys.path if Path(entry or ".").resolve() != worker_directory
            ]
        whisperx = importlib.import_module("whisperx")
    except ImportError as exc:  # pragma: no cover - runs in optional worker environment
        raise RuntimeError("The WhisperX worker environment is missing whisperx") from exc

    language = arguments.language or None
    with contextlib.redirect_stdout(sys.stderr):
        audio = whisperx.load_audio(arguments.audio)
        model = whisperx.load_model(
            arguments.model,
            arguments.device,
            compute_type=arguments.compute_type,
            language=language,
        )
        transcription_arguments: dict[str, Any] = {
            "batch_size": arguments.batch_size,
            "language": language,
        }
        if arguments.prompt:
            try:
                result = model.transcribe(
                    audio,
                    **transcription_arguments,
                    initial_prompt=arguments.prompt,
                )
            except TypeError:
                result = model.transcribe(audio, **transcription_arguments)
        else:
            result = model.transcribe(audio, **transcription_arguments)
        detected_language = result["language"]
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=arguments.device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            arguments.device,
            return_char_alignments=False,
        )
    words = [
        {"word": word.word, "start": word.start, "end": word.end}
        for word in _words_from_payload(
            [word for segment in result.get("segments", ()) for word in segment.get("words", ())]
        )
    ]
    json.dump(
        {"language": result.get("language") or detected_language or language, "words": words},
        sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--prompt", default="")
    arguments = parser.parse_args(argv)
    if not arguments.worker:
        parser.error("This module is an internal WhisperX worker")
    _worker_main(arguments)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke tests
    main()
