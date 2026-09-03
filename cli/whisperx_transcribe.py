"""WhisperX transcription and forced-alignment adapter."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from cli.logging_setup import get_logger

logger = get_logger("cli.whisperx")

_DLL_DIRECTORY_HANDLES: list[Any] = []
# Keep ordinary doubled letters and short sung-word elongations intact.
_MIN_REPEATED_CHARACTER_RUN_LENGTH = 8
_REPEATED_CHARACTER_RUN = re.compile(
    rf"(?P<run>(?P<char>[^\s])(?P=char){{{_MIN_REPEATED_CHARACTER_RUN_LENGTH - 1},}})"
    r"(?:\ufffd+)?"
)
# The negative lookbehinds avoid stripping these letter sequences from the
# legitimate words "Mississippi" and "cinnamon".
_KNOWN_TEXT_ARTIFACT = re.compile(r"(?<!miss)issippi|(?<!cin)namon", re.IGNORECASE)


def _prepare_windows_dll_search_path() -> None:
    """Expose FFmpeg's shared DLLs before pyannote imports TorchCodec."""
    if os.name != "nt" or _DLL_DIRECTORY_HANDLES:
        return
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe or not hasattr(os, "add_dll_directory"):
        return
    ffmpeg_dir = Path(ffmpeg_exe).resolve().parent
    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(ffmpeg_dir)))


def load_whisperx_asr_model(
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    initial_prompt: str | None,
):
    """Load a WhisperX ASR model configured for one song."""
    _prepare_windows_dll_search_path()
    import whisperx

    asr_options = {"initial_prompt": initial_prompt} if initial_prompt else None
    return whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
    )


def load_faster_whisper_model(model_name: str, device: str, compute_type: str):
    """Load standalone faster-whisper without WhisperX's VAD segmentation."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def load_audio(path: str):
    """Decode an audio file into WhisperX's 16 kHz waveform format."""
    _prepare_windows_dll_search_path()
    import whisperx

    return whisperx.load_audio(path)


def filter_repeated_character_runs(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove ASR character-run artifacts before WhisperX forced alignment."""
    filtered_segments: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        filtered_artifact_count = 0

        def replace_artifact(match: re.Match[str]) -> str:
            nonlocal filtered_artifact_count
            filtered_artifact_count += 1
            artifact = match.group(0)
            repeated_run = match.group("run")
            repeated_character = match.group("char")
            run_length = len(repeated_run)
            replacement_count = len(artifact) - run_length
            replacement_note = (
                f" followed by {replacement_count} Unicode replacement character(s)"
                if replacement_count
                else ""
            )
            logger.warning(
                "Filtered repeated-character artifact before forced alignment: "
                "segment=%d offset=%d character=%r count=%d%s",
                segment_index + 1,
                match.start(),
                repeated_character,
                run_length,
                replacement_note,
            )
            return " "

        filtered_text = _REPEATED_CHARACTER_RUN.sub(replace_artifact, text)
        if not filtered_artifact_count:
            filtered_segments.append(segment)
            continue

        if not filtered_text.strip():
            logger.warning(
                "Dropping WhisperX segment %d because no text remained after filtering",
                segment_index + 1,
            )
            continue

        filtered_segment = dict(segment)
        filtered_segment["text"] = filtered_text
        filtered_segments.append(filtered_segment)

    return filtered_segments


def filter_known_text_artifacts(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove known WhisperX hallucination fragments before forced alignment."""
    filtered_segments: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        filtered_artifact_count = 0

        def replace_artifact(match: re.Match[str]) -> str:
            nonlocal filtered_artifact_count
            filtered_artifact_count += 1
            artifact = match.group(0)
            logger.warning(
                "Filtered known transcription artifact before forced alignment: "
                "segment=%d offset=%d artifact=%r",
                segment_index + 1,
                match.start(),
                artifact,
            )

            left = text[match.start() - 1] if match.start() else ""
            right = text[match.end()] if match.end() < len(text) else ""
            return " " if left.isalnum() and right.isalnum() else ""

        filtered_text = _KNOWN_TEXT_ARTIFACT.sub(replace_artifact, text)
        if not filtered_artifact_count:
            filtered_segments.append(segment)
            continue

        if not filtered_text.strip():
            logger.warning(
                "Dropping WhisperX segment %d because no text remained after filtering",
                segment_index + 1,
            )
            continue

        filtered_segment = dict(segment)
        filtered_segment["text"] = filtered_text
        filtered_segments.append(filtered_segment)

    return filtered_segments


def transcribe_with_faster_whisper(
    model,
    audio_path: str,
    language_hint: str | None,
    initial_prompt: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Transcribe with the standalone faster-whisper decoding path."""
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        language=language_hint,
    )
    transcript = [
        {
            "text": str(segment.text),
            "start": float(segment.start),
            "end": float(segment.end),
            "avg_logprob": float(segment.avg_logprob),
        }
        for segment in segments
        if str(segment.text).strip()
    ]
    return transcript, str(info.language or language_hint or "en")


def transcribe_with_whisperx(
    model,
    audio,
    batch_size: int,
    language_hint: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Transcribe with WhisperX's batched, VAD-driven ASR wrapper."""
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language_hint,
    )
    return list(result.get("segments") or []), str(
        result.get("language") or language_hint or "en"
    )


def align_segments(
    segments: list[dict[str, Any]],
    language: str,
    audio,
    device: str,
    align_model_name: str | None,
    interpolate_method: str,
    align_model_cache: dict[str, tuple[Any, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str]:
    """Forced-align ASR segments with WhisperX and retain character timings."""
    _prepare_windows_dll_search_path()
    import whisperx

    segments = filter_repeated_character_runs(segments)
    segments = filter_known_text_artifacts(segments)
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
