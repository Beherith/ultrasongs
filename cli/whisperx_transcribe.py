"""WhisperX transcription and forced-alignment adapter."""

from __future__ import annotations

from typing import Any

from cli.logging_setup import get_logger

logger = get_logger("cli.whisperx")


def load_asr_model(
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    initial_prompt: str | None,
):
    """Load a WhisperX ASR model configured for one song."""
    import whisperx

    asr_options = {"initial_prompt": initial_prompt} if initial_prompt else None
    return whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
    )


def load_audio(path: str):
    """Decode an audio file into WhisperX's 16 kHz waveform format."""
    import whisperx

    return whisperx.load_audio(path)


def transcribe_and_align(
    model,
    audio,
    device: str,
    batch_size: int,
    language_hint: str | None,
    align_model_name: str | None,
    interpolate_method: str,
    align_model_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str]:
    """Run WhisperX ASR and forced alignment, returning internal word dicts."""
    import whisperx

    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language_hint,
    )
    language = str(result.get("language") or language_hint or "en")
    segments = result.get("segments", [])
    if not segments:
        return [], language

    cache_key = f"{language}:{align_model_name or ''}"
    if cache_key not in align_model_cache:
        try:
            align_model_cache[cache_key] = whisperx.load_align_model(
                language_code=language,
                device=device,
                model_name=align_model_name,
            )
        except ValueError as exc:
            override = " Set whisperx_align_model to a compatible wav2vec2 model."
            raise RuntimeError(
                f"WhisperX has no usable alignment model for language {language!r}.{override}"
            ) from exc

    align_model, metadata = align_model_cache[cache_key]
    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        interpolate_method=interpolate_method,
        return_char_alignments=True,
    )
    return extract_aligned_words(aligned), language


def extract_aligned_words(aligned: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten WhisperX segments while retaining per-character timestamps."""
    output: list[dict[str, Any]] = []
    for segment in aligned.get("segments", []):
        words = segment.get("words", [])
        char_groups = _group_characters(words, segment.get("chars") or [])
        for index, word in enumerate(words):
            if "start" not in word or "end" not in word:
                logger.warning("Skipping WhisperX word without timestamps: %r", word.get("word"))
                continue
            output.append({
                "word": str(word.get("word", "")).strip(),
                "start": float(word["start"]),
                "end": float(word["end"]),
                "score": _optional_float(word.get("score")),
                "characters": char_groups[index] if index < len(char_groups) else [],
            })
    return [word for word in output if word["word"]]


def _group_characters(
    words: list[dict[str, Any]],
    characters: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Assign WhisperX segment characters to its words in text order."""
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for word in words:
        while cursor < len(characters) and str(characters[cursor].get("char", "")).isspace():
            cursor += 1

        target = str(word.get("word", "")).strip()
        group: list[dict[str, Any]] = []
        consumed = ""
        while cursor < len(characters) and len(consumed) < len(target):
            raw = characters[cursor]
            cursor += 1
            char = str(raw.get("char", ""))
            if char.isspace() and not consumed:
                continue
            if char.isspace():
                break
            consumed += char
            group.append({
                "char": char,
                "start": _optional_float(raw.get("start")),
                "end": _optional_float(raw.get("end")),
                "score": _optional_float(raw.get("score")),
            })

        if consumed != target:
            logger.debug("WhisperX word/character mismatch: word=%r chars=%r", target, consumed)
        groups.append(group)
    return groups


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
