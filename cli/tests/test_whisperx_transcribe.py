"""Tests for converting WhisperX forced-alignment results."""

import sys
from types import SimpleNamespace

from cli.whisperx_transcribe import extract_aligned_words, load_asr_model, transcribe_and_align


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


def test_load_asr_model_passes_lyrics_as_initial_prompt(monkeypatch):
    calls = []
    fake_whisperx = SimpleNamespace(load_model=lambda *args, **kwargs: calls.append((args, kwargs)) or object())
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    load_asr_model("medium", "cuda", "float16", "en", "known lyrics")

    assert calls[0][0] == ("medium", "cuda")
    assert calls[0][1]["language"] == "en"
    assert calls[0][1]["asr_options"] == {"initial_prompt": "known lyrics"}


def test_transcribe_and_align_reuses_language_alignment_model(monkeypatch):
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
    model = SimpleNamespace(
        transcribe=lambda audio, **kwargs: {"language": "en", "segments": [{"text": "hi"}]},
    )
    cache = {}

    first, language = transcribe_and_align(
        model, object(), "cuda", 8, "en", None, "nearest", cache,
    )
    second, _ = transcribe_and_align(
        model, object(), "cuda", 8, "en", None, "nearest", cache,
    )

    assert language == "en"
    assert first == second
    assert len(load_calls) == 1
    assert first[0]["characters"][1]["char"] == "i"
