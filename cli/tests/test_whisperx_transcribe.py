"""Tests for converting WhisperX forced-alignment results."""

import sys
from types import SimpleNamespace

import cli.whisperx_transcribe as whisperx_adapter
from cli.whisperx_transcribe import (
    align_segments,
    extract_aligned_words,
    filter_known_text_artifacts,
    filter_repeated_character_runs,
    load_faster_whisper_model,
    load_whisperx_asr_model,
    transcribe_with_faster_whisper,
    transcribe_with_whisperx,
)


def test_extract_aligned_words_retains_characters():
    aligned = {
        "segments": [{
            "text": " hello world",
            "words": [
                {"word": "hello", "start": 0.1, "end": 0.5, "score": 0.9},
                {"word": "world", "start": 0.6, "end": 1.0, "score": 0.8},
            ],
            "chars": [
                {"char": " ", "start": 0.0, "end": 0.1, "score": 1.0},
                *[
                    {"char": char, "start": 0.1 + i * 0.08, "end": 0.18 + i * 0.08, "score": 0.9}
                    for i, char in enumerate("hello")
                ],
                {"char": " ", "start": 0.5, "end": 0.6, "score": 1.0},
                *[
                    {"char": char, "start": 0.6 + i * 0.08, "end": 0.68 + i * 0.08, "score": 0.8}
                    for i, char in enumerate("world")
                ],
            ],
        }],
    }

    words = extract_aligned_words(aligned)

    assert [word["word"] for word in words] == ["hello", "world"]
    assert "".join(char["char"] for char in words[0]["characters"]) == "hello"
    assert words[1]["characters"][-1]["end"] == 1.0


def test_extract_skips_word_without_forced_timestamps():
    aligned = {
        "segments": [{
            "words": [{"word": "missing"}],
            "chars": [],
        }],
    }

    assert extract_aligned_words(aligned) == []


def test_filter_repeated_character_runs_logs_and_removes_artifacts(caplog):
    segments = [
        {"text": "before " + "𝘤" * 12 + "\ufffd after", "start": 1.0, "end": 2.0},
        {"text": "!" * 8},
    ]

    filtered = filter_repeated_character_runs(segments)

    assert filtered == [{"text": "before   after", "start": 1.0, "end": 2.0}]
    assert segments[0]["text"] == "before " + "𝘤" * 12 + "\ufffd after"
    assert "character='𝘤' count=12 followed by 1 Unicode replacement character(s)" in caplog.text
    assert "character='!' count=8" in caplog.text
    assert "Dropping WhisperX segment 2" in caplog.text


def test_filter_known_text_artifacts_handles_attached_fragments_and_logs(caplog):
    text = (
        "store issippi born stoneissippi Raised homeissippiSkin "
        "holenamonDiggy holeissippi, Mississippi cinnamon"
    )

    filtered = filter_known_text_artifacts([{"text": text, "start": 1.0}])

    assert filtered == [{
        "text": (
            "store  born stone Raised home Skin hole Diggy hole, "
            "Mississippi cinnamon"
        ),
        "start": 1.0,
    }]
    assert caplog.text.count("Filtered known transcription artifact") == 5
    assert "artifact='issippi'" in caplog.text
    assert "artifact='namon'" in caplog.text


def test_load_whisperx_asr_model_passes_lyrics_as_initial_prompt(monkeypatch):
    calls = []
    fake_whisperx = SimpleNamespace(load_model=lambda *args, **kwargs: calls.append((args, kwargs)) or object())
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    load_whisperx_asr_model("medium", "cuda", "float16", "en", "known lyrics")

    assert calls[0][0] == ("medium", "cuda")
    assert calls[0][1]["language"] == "en"
    assert calls[0][1]["asr_options"] == {"initial_prompt": "known lyrics"}


def test_load_faster_whisper_model_uses_standalone_decoder(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(
        WhisperModel=lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    load_faster_whisper_model("large-v3", "cuda", "auto")

    assert calls == [(('large-v3',), {"device": "cuda", "compute_type": "auto"})]


def test_align_segments_reuses_language_alignment_model(monkeypatch):
    aligned = {
        "segments": [{
            "words": [{"word": "hi", "start": 0.1, "end": 0.3, "score": 0.9}],
            "chars": [
                {"char": "h", "start": 0.1, "end": 0.2, "score": 0.9},
                {"char": "i", "start": 0.2, "end": 0.3, "score": 0.9},
            ],
        }],
    }
    load_calls = []
    fake_whisperx = SimpleNamespace(
        load_align_model=lambda **kwargs: load_calls.append(kwargs) or (object(), {"language": "en"}),
        align=lambda *args, **kwargs: aligned,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    cache = {}

    first, language = align_segments(
        [{"text": "hi", "start": 0.1, "end": 0.3}],
        "en", object(), "cuda", None, "nearest", cache,
    )
    second, _ = align_segments(
        [{"text": "hi", "start": 0.1, "end": 0.3}],
        "en", object(), "cuda", None, "nearest", cache,
    )

    assert language == "en"
    assert first == second
    assert len(load_calls) == 1
    assert first[0]["characters"][1]["char"] == "i"


def test_align_segments_filters_artifacts_before_alignment(monkeypatch):
    align_calls = []
    fake_whisperx = SimpleNamespace(
        load_align_model=lambda **kwargs: (object(), {"language": "en"}),
        align=lambda segments, *args, **kwargs: align_calls.append(segments) or {"segments": []},
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    align_segments(
        [{"text": "sing " + "x" * 20 + " holenamonDiggy with me"}],
        "en", object(), "cpu", None, "nearest", {},
    )

    assert align_calls == [[{"text": "sing   hole Diggy with me"}]]


def test_faster_whisper_segments_keep_original_decoding_options():
    calls = []
    segments = [SimpleNamespace(
        text=" hello", start=0.1, end=0.5, avg_logprob=-0.2,
    )]
    model = SimpleNamespace(
        transcribe=lambda path, **kwargs: calls.append((path, kwargs))
        or (iter(segments), SimpleNamespace(language="en")),
    )

    transcript, language = transcribe_with_faster_whisper(
        model, "vocals.wav", "en", "known lyrics",
    )

    assert language == "en"
    assert transcript == [{
        "text": " hello", "start": 0.1, "end": 0.5, "avg_logprob": -0.2,
    }]
    assert calls == [("vocals.wav", {
        "word_timestamps": True,
        "initial_prompt": "known lyrics",
        "language": "en",
    })]


def test_whisperx_transcription_returns_segments_and_language():
    model = SimpleNamespace(
        transcribe=lambda audio, **kwargs: {
            "language": "de", "segments": [{"text": " hallo"}],
        },
    )

    segments, language = transcribe_with_whisperx(model, object(), 8, "de")

    assert segments == [{"text": " hallo"}]
    assert language == "de"


def test_windows_ffmpeg_directory_is_registered_once(monkeypatch):
    handles = []
    monkeypatch.setattr(whisperx_adapter.os, "name", "nt")
    monkeypatch.setattr(whisperx_adapter.shutil, "which", lambda name: r"C:\ffmpeg\bin\ffmpeg.exe")
    monkeypatch.setattr(
        whisperx_adapter.os,
        "add_dll_directory",
        lambda path: handles.append(path) or object(),
        raising=False,
    )
    monkeypatch.setattr(whisperx_adapter, "_DLL_DIRECTORY_HANDLES", [])

    whisperx_adapter._prepare_windows_dll_search_path()
    whisperx_adapter._prepare_windows_dll_search_path()

    assert handles == [r"C:\ffmpeg\bin"]
