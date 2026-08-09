from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ultrasongs.processing.pitch_detection import (
    TorchCrepePitchService,
    _PitchBackend,
)
from ultrasongs.processing.separation import (
    DemucsSeparationService,
    _DemucsBackend,
)
from ultrasongs.processing.transcription import FasterWhisperService
from ultrasongs.processing.whisperx import WhisperXService, _worker_main


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def unsqueeze(self, axis: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self.values, axis))

    def squeeze(self, axis: int) -> FakeTensor:
        return FakeTensor(np.squeeze(self.values, axis))

    def expand(self, *shape: int) -> FakeTensor:
        resolved = tuple(
            current if requested == -1 else requested
            for current, requested in zip(self.values.shape, shape, strict=True)
        )
        return FakeTensor(np.broadcast_to(self.values, resolved))

    def mean(self, axis: int) -> FakeTensor:
        return FakeTensor(self.values.mean(axis))

    def sum(self, axis: int) -> FakeTensor:
        return FakeTensor(self.values.sum(axis))

    def cuda(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values

    def __getitem__(self, key: Any) -> FakeTensor:
        return FakeTensor(self.values[key])


class FakeCuda:
    def __init__(self, available: bool = False) -> None:
        self.available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self, *, cuda: bool = False) -> None:
        self.cuda = FakeCuda(cuda)

    @staticmethod
    def from_numpy(values: np.ndarray[Any, Any]) -> FakeTensor:
        return FakeTensor(values)

    @staticmethod
    def stack(values: list[FakeTensor]) -> FakeTensor:
        return FakeTensor(np.stack([value.values for value in values]))

    @staticmethod
    def no_grad() -> nullcontext[None]:
        return nullcontext()


class FakeResampler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def resample(self, tensor: FakeTensor, source_rate: int, target_rate: int) -> FakeTensor:
        self.calls.append((source_rate, target_rate))
        return tensor


class FakeDemucsModel:
    samplerate = 44_100
    sources = ["drums", "bass", "other", "vocals"]

    def __init__(self) -> None:
        self.evaluated = False
        self.on_cuda = False

    def eval(self) -> None:
        self.evaluated = True

    def cuda(self) -> FakeDemucsModel:
        self.on_cuda = True
        return self


def test_demucs_is_lazy_uses_settings_and_unloads_cuda() -> None:
    torch = FakeTorch(cuda=True)
    resampler = FakeResampler()
    model = FakeDemucsModel()
    calls: list[dict[str, Any]] = []
    loader_calls = 0

    def apply_model(_model: Any, wave: FakeTensor, **kwargs: Any) -> FakeTensor:
        calls.append({"shape": wave.shape, **kwargs})
        stems = np.stack(
            [np.full((2, wave.shape[-1]), value, dtype=np.float32) for value in (1, 2, 3, 4)]
        )
        return FakeTensor(stems[np.newaxis, ...])

    def load_backend() -> _DemucsBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _DemucsBackend(
            torch,
            SimpleNamespace(functional=resampler),
            lambda name: model if name == "htdemucs" else None,
            apply_model,
        )

    settings = SimpleNamespace(model="htdemucs", shifts=2, overlap=0.4, device="auto")
    service = DemucsSeparationService(settings, backend_loader=load_backend)
    assert loader_calls == 0

    result = service.separate(np.array([0.1, -0.1], dtype=np.float32), 48_000)

    assert loader_calls == 1
    assert model.evaluated and model.on_cuda
    assert resampler.calls == [(48_000, 44_100)]
    assert calls == [{"shape": (1, 2, 2), "device": "cuda", "shifts": 2, "overlap": 0.4}]
    np.testing.assert_array_equal(result.vocals, [4, 4])
    np.testing.assert_array_equal(result.accompaniment, [6, 6])
    assert result.sample_rate_hz == 44_100
    assert torch.cuda.empty_cache_calls == 1


def test_demucs_rejects_non_mono_input_without_loading_backend() -> None:
    loaded = False

    def load_backend() -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError

    settings = SimpleNamespace(model="htdemucs", shifts=1, overlap=0.25, device="cpu")
    service = DemucsSeparationService(settings, backend_loader=load_backend)

    with pytest.raises(ValueError, match="mono"):
        service.separate(np.zeros((2, 4), dtype=np.float32), 44_100)
    assert loaded is False


def test_torchcrepe_is_lazy_and_attaches_shared_pitch_output() -> None:
    torch = FakeTorch()
    resampler = FakeResampler()
    predict_calls: list[dict[str, Any]] = []

    def predict(tensor: FakeTensor, **kwargs: Any) -> tuple[FakeTensor, FakeTensor]:
        predict_calls.append({"shape": tensor.shape, **kwargs})
        return (
            FakeTensor([[440.0, 0.0, 880.0]]),
            FakeTensor([[0.9, 0.2, 0.8]]),
        )

    crepe = SimpleNamespace(
        predict=predict,
        decode=SimpleNamespace(viterbi="viterbi-decoder"),
    )
    loader_calls = 0

    def load_backend() -> _PitchBackend:
        nonlocal loader_calls
        loader_calls += 1
        return _PitchBackend(torch, SimpleNamespace(functional=resampler), crepe)

    settings = SimpleNamespace(
        model="full",
        sample_rate_hz=16_000,
        hop_length=160,
        min_frequency_hz=65.41,
        max_frequency_hz=1_046.5,
        batch_size=2048,
        confidence_thresholds=(0.5, 0.3, 0.1),
        device="auto",
    )
    service = TorchCrepePitchService(settings, backend_loader=load_backend)
    assert loader_calls == 0

    track = service.analyze(np.ones(100, dtype=np.float32), 44_100)
    words = service.attach_to_words([{"word": " sing ", "start": 0.0, "end": 0.02}], track)

    assert loader_calls == 1
    assert resampler.calls == [(44_100, 16_000)]
    assert predict_calls[0]["shape"] == (1, 100)
    assert predict_calls[0]["sample_rate"] == 16_000
    assert predict_calls[0]["decoder"] == "viterbi-decoder"
    np.testing.assert_allclose(track.times, [0.0, 0.01, 0.02])
    assert words[0]["word"] == "sing"
    assert words[0]["midi"] == 76  # legacy behavior converts median frequency (660 Hz)
    assert len(words[0]["pitchFrames"]) == 2


def test_faster_whisper_model_is_lazy_cached_and_extracts_words(tmp_path: Path) -> None:
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"wave")
    factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    transcribe_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            transcribe_calls.append((path, kwargs))
            words = [
                SimpleNamespace(word=" hello ", start=0.1, end=0.3),
                SimpleNamespace(word="", start=0.3, end=0.4),
            ]
            return [SimpleNamespace(words=words)], SimpleNamespace(language="hu")

    def factory(*args: Any, **kwargs: Any) -> FakeModel:
        factory_calls.append((args, kwargs))
        return FakeModel()

    loader_calls = 0

    def factory_loader() -> Any:
        nonlocal loader_calls
        loader_calls += 1
        return factory

    settings = SimpleNamespace(
        model="medium",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=5,
        vad_filter=True,
    )
    service = FasterWhisperService(settings, model_factory_loader=factory_loader)
    assert loader_calls == 0

    first = service.transcribe(audio, prompt=" lyrics ")
    second = service.transcribe(audio)

    assert loader_calls == 1
    assert factory_calls == [(("medium",), {"device": "cpu", "compute_type": "int8"})]
    assert first.words_as_dicts() == [{"word": "hello", "start": 0.1, "end": 0.3}]
    assert first.language == "hu"
    assert second.language == "hu"
    assert transcribe_calls[0][1] == {
        "word_timestamps": True,
        "initial_prompt": "lyrics",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
    }
    service.close()
    service.transcribe(audio)
    assert loader_calls == 2


def test_whisperx_command_uses_injected_settings_and_parses_words(tmp_path: Path) -> None:
    audio = tmp_path / "vocals.wav"
    worker = tmp_path / "worker.py"
    audio.write_bytes(b"wave")
    worker.write_text("# worker", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        payload = {
            "language": "de",
            "words": [
                {"word": " Glück ", "start": 1, "end": 1.5},
                {"word": "", "start": 2, "end": 3},
            ],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="diagnostic")

    settings = SimpleNamespace(
        model="small",
        batch_size=4,
        device="cpu",
        compute_type="int8",
        python_executable=sys.executable,
    )
    language_settings = SimpleNamespace(language="de")
    service = WhisperXService(
        settings,
        language_settings,
        worker_path=worker,
        working_directory=tmp_path,
        runner=runner,
    )

    result = service.transcribe(audio, prompt=" song lyrics ")

    command, options = calls[0]
    assert command == [
        sys.executable,
        str(worker.resolve()),
        "--worker",
        "--audio",
        str(audio.resolve()),
        "--model",
        "small",
        "--batch-size",
        "4",
        "--device",
        "cpu",
        "--compute-type",
        "int8",
        "--language",
        "de",
        "--prompt",
        "song lyrics",
    ]
    assert options == {
        "cwd": tmp_path.resolve(),
        "text": True,
        "capture_output": True,
        "check": False,
    }
    assert result.language == "de"
    assert result.words_as_dicts() == [{"word": "Glück", "start": 1.0, "end": 1.5}]


def test_whisperx_reports_process_and_json_failures(tmp_path: Path) -> None:
    audio = tmp_path / "vocals.wav"
    worker = tmp_path / "worker.py"
    audio.write_bytes(b"wave")
    worker.write_text("# worker", encoding="utf-8")
    settings = SimpleNamespace(
        model="small",
        batch_size=4,
        device="cpu",
        compute_type="int8",
        python_executable=sys.executable,
    )
    language_settings = SimpleNamespace(language=None)

    failing = WhisperXService(
        settings,
        language_settings,
        worker_path=worker,
        runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=7, stdout="", stderr="out of memory"
        ),
    )
    with pytest.raises(RuntimeError, match="code 7: out of memory"):
        failing.transcribe(audio)

    invalid = WhisperXService(
        settings,
        language_settings,
        worker_path=worker,
        runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not json", stderr=""),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        invalid.transcribe(audio)


def test_whisperx_worker_aligns_words_and_falls_back_when_prompt_is_unsupported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transcription_calls: list[dict[str, Any]] = []

    class FakeModel:
        def transcribe(self, _audio: Any, **kwargs: Any) -> dict[str, Any]:
            transcription_calls.append(kwargs)
            if "initial_prompt" in kwargs:
                raise TypeError("unsupported")
            return {"language": "hu", "segments": [{"text": "dal"}]}

    fake_whisperx = SimpleNamespace(
        load_audio=lambda path: f"loaded:{path}",
        load_model=lambda *args, **kwargs: FakeModel(),
        load_align_model=lambda **kwargs: ("align-model", "metadata"),
        align=lambda *args, **kwargs: {
            "segments": [
                {
                    "words": [
                        {"word": " dal ", "start": 0.25, "end": 0.75},
                        {"word": "missing timing"},
                    ]
                }
            ]
        },
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    arguments = SimpleNamespace(
        audio="vocals.wav",
        model="small",
        batch_size=4,
        device="cpu",
        compute_type="int8",
        language="",
        prompt="known lyrics",
    )

    _worker_main(arguments)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "language": "hu",
        "words": [{"word": "dal", "start": 0.25, "end": 0.75}],
    }
    assert transcription_calls == [
        {"batch_size": 4, "language": None, "initial_prompt": "known lyrics"},
        {"batch_size": 4, "language": None},
    ]
