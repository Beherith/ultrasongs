"""Word timing, pitch aggregation, interpolation, and lyric line alignment."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AlignedSyllable,
    AlignedWord,
    AlignmentDebug,
    AlignmentResult,
    AlignmentSummary,
    BacktrackDebugStep,
    CharacterAlignment,
    Pause,
    PitchFrame,
    WordSource,
    WordTimestamp,
)
from .normalization import normalize_character
from .smith_waterman import smith_waterman
from .syllables import split_word

DebugSink = Callable[[str, str], None]


@dataclass(slots=True)
class _LyricCharacter:
    original: str
    normalized: str
    word_index: int
    line_index: int


@dataclass(slots=True)
class _TranscriptionCharacter:
    original: str
    normalized: str
    word_index: int


@dataclass(slots=True)
class _MutableWord:
    word: str
    line_index: int
    start: float = 0.0
    end: float = 0.0
    midi: int = 60
    source: WordSource = "sw_aligned"
    transcription_indices: list[int] = field(default_factory=list)
    character_alignments: list[CharacterAlignment] = field(default_factory=list)
    pitch_frames: list[PitchFrame] = field(default_factory=list)


def align_lyrics(
    lyrics: str,
    transcription_words: Sequence[WordTimestamp | Mapping[str, Any]],
    language: str,
    pauses: Sequence[Pause | Mapping[str, Any]] = (),
    *,
    debug_sink: DebugSink | None = None,
    match_score: float = 4.0,
    gap_open_penalty: float = 4.0,
    gap_extend_penalty: float = 0.5,
) -> list[AlignedSyllable]:
    """Compatibility-friendly API returning just the aligned syllable list."""
    return list(
        align_lyrics_with_debug(
            lyrics,
            transcription_words,
            language,
            pauses,
            debug_sink=debug_sink,
            match_score=match_score,
            gap_open_penalty=gap_open_penalty,
            gap_extend_penalty=gap_extend_penalty,
        ).syllables
    )


def align_lyrics_with_debug(
    lyrics: str,
    transcription_words: Sequence[WordTimestamp | Mapping[str, Any]],
    language: str,
    pauses: Sequence[Pause | Mapping[str, Any]] = (),
    *,
    debug_sink: DebugSink | None = None,
    match_score: float = 4.0,
    gap_open_penalty: float = 4.0,
    gap_extend_penalty: float = 0.5,
) -> AlignmentResult:
    """Align supplied lyrics without filesystem writes or module-global state."""
    words = tuple(_coerce_word(word) for word in transcription_words)
    normalized_pauses = tuple(_coerce_pause(pause) for pause in pauses)
    messages: list[str] = []

    def emit(tag: str, message: str) -> None:
        messages.append(f"[align:{tag}] {message}")
        if debug_sink is not None:
            debug_sink(tag, message)

    raw_lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    lyric_words = [
        (word, line_index)
        for line_index, line in enumerate(raw_lines)
        for word in line.split()
        if word
    ]
    lyric_characters = _lyric_characters(lyric_words)
    transcription_characters = _transcription_characters(words)

    emit("init", f"lyrics={len(lyric_words)} words, transcription={len(words)} words")
    sw_result = smith_waterman(
        [char.normalized for char in lyric_characters],
        [char.normalized for char in transcription_characters],
        match_score=match_score,
        gap_open_penalty=gap_open_penalty,
        gap_extend_penalty=gap_extend_penalty,
    )
    emit(
        "sw",
        f"maxScore={sw_result.max_score:.2f} at ({sw_result.max_i}, {sw_result.max_j})",
    )

    mutable_words = [_MutableWord(word=word, line_index=line) for word, line in lyric_words]
    matched = [False] * len(mutable_words)
    debug_backtrack: list[BacktrackDebugStep] = []

    for step in sw_result.backtrack:
        lyric_character = (
            lyric_characters[step.i - 1] if 0 < step.i <= len(lyric_characters) else None
        )
        transcription_character = (
            transcription_characters[step.j - 1]
            if 0 < step.j <= len(transcription_characters)
            else None
        )
        transcription_word = (
            words[transcription_character.word_index].word if transcription_character else "-"
        )
        debug_backtrack.append(
            BacktrackDebugStep(
                lyric_index=step.i,
                transcription_index=step.j,
                matrix=step.matrix,
                score=step.score,
                lyric_character=lyric_character.original if lyric_character else "-",
                transcription_character=(
                    transcription_character.original if transcription_character else "-"
                ),
                transcription_word_index=(
                    transcription_character.word_index if transcription_character else -1
                ),
                transcription_word=transcription_word,
            )
        )
        if (
            step.matrix != "M"
            or lyric_character is None
            or transcription_character is None
            or lyric_character.normalized == " "
            or transcription_character.normalized == " "
        ):
            continue
        word_index = lyric_character.word_index
        matched[word_index] = True
        transcription_index = transcription_character.word_index
        if transcription_index not in mutable_words[word_index].transcription_indices:
            mutable_words[word_index].transcription_indices.append(transcription_index)
        mutable_words[word_index].character_alignments.append(
            CharacterAlignment(
                lyric_character=lyric_character.original,
                transcription_character=transcription_character.original,
                transcription_word_index=transcription_index,
                score=step.score,
            )
        )

    _assign_matched_word_data(mutable_words, matched, words)
    _interpolate_unmatched_words(mutable_words, matched)
    syllables = _build_syllables(mutable_words, raw_lines, language)

    aligned_words = tuple(
        AlignedWord(
            word=word.word,
            line_index=word.line_index,
            start=word.start,
            end=word.end,
            midi=word.midi,
            source=word.source,
            transcription_word_indices=tuple(sorted(word.transcription_indices)),
            character_alignments=tuple(word.character_alignments),
        )
        for word in mutable_words
    )
    aligned_count = sum(word.source == "sw_aligned" for word in mutable_words)
    line_breaks = sum(syllable.is_line_break for syllable in syllables)
    summary = AlignmentSummary(
        total_lyric_words=len(mutable_words),
        aligned_words=aligned_count,
        interpolated_words=len(mutable_words) - aligned_count,
        total_syllables=len(syllables) - line_breaks,
        line_breaks=line_breaks,
    )
    time_range = (words[0].start, words[-1].end) if words else (0.0, 0.0)
    debug = AlignmentDebug(
        language=language,
        lyric_character_count=len(lyric_characters),
        transcription_character_count=len(transcription_characters),
        transcription_word_count=len(words),
        transcription_time_range=time_range,
        smith_waterman_max_score=sw_result.max_score,
        smith_waterman_max_position=(sw_result.max_i, sw_result.max_j),
        backtrack=tuple(debug_backtrack),
        words=aligned_words,
        pauses=normalized_pauses,
        summary=summary,
    )
    emit(
        "summary",
        f"aligned={summary.aligned_words}, interpolated={summary.interpolated_words}, "
        f"syllables={summary.total_syllables}, lineBreaks={summary.line_breaks}",
    )
    return AlignmentResult(syllables=tuple(syllables), debug=debug, messages=tuple(messages))


def midi_for_range(
    word: WordTimestamp,
    start: float,
    end: float,
    thresholds: Sequence[float] = (0.5, 0.3, 0.1),
) -> tuple[int, int]:
    """Return legacy median MIDI and contributing frame count for a time range."""
    if not word.pitch_frames:
        return word.midi, 0
    for threshold in thresholds:
        values = [
            frame.midi
            for frame in word.pitch_frames
            if start <= frame.time <= end
            and frame.confidence > threshold
            and math.isfinite(frame.midi)
            and frame.midi > 0
        ]
        if values:
            return _legacy_median(values), len(values)
    return word.midi, 0


def _legacy_median(values: Sequence[float]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[middle])
    # MIDI is positive, so this exactly matches JavaScript Math.round here.
    return math.floor(((ordered[middle - 1] + ordered[middle]) / 2) + 0.5)


def _lyric_characters(words: Sequence[tuple[str, int]]) -> list[_LyricCharacter]:
    characters: list[_LyricCharacter] = []
    for word_index, (word, line_index) in enumerate(words):
        characters.extend(
            _LyricCharacter(character, normalize_character(character), word_index, line_index)
            for character in word
        )
        if word_index < len(words) - 1:
            characters.append(_LyricCharacter(" ", " ", word_index, line_index))
    return characters


def _transcription_characters(words: Sequence[WordTimestamp]) -> list[_TranscriptionCharacter]:
    characters: list[_TranscriptionCharacter] = []
    for word_index, word in enumerate(words):
        cleaned = normalize_character(word.word)
        characters.extend(
            _TranscriptionCharacter(character, normalize_character(character), word_index)
            for character in cleaned
        )
        if word_index < len(words) - 1:
            characters.append(_TranscriptionCharacter(" ", " ", word_index))
    return characters


def _assign_matched_word_data(
    results: list[_MutableWord], matched: Sequence[bool], words: Sequence[WordTimestamp]
) -> None:
    for index, result in enumerate(results):
        if not matched[index]:
            continue
        indices = sorted(result.transcription_indices)
        result.start = min(words[word_index].start for word_index in indices)
        result.end = max(words[word_index].end for word_index in indices)
        result.pitch_frames = [
            frame for word_index in indices for frame in words[word_index].pitch_frames
        ]
        if result.pitch_frames:
            aggregate = WordTimestamp(
                word="",
                start=result.start,
                end=result.end,
                midi=60,
                pitch_frames=tuple(result.pitch_frames),
            )
            result.midi, _ = midi_for_range(aggregate, result.start, result.end)
        else:
            result.midi = words[indices[0]].midi if indices else 60


def _interpolate_unmatched_words(results: list[_MutableWord], matched: Sequence[bool]) -> None:
    matched_indices = [index for index, is_matched in enumerate(matched) if is_matched]
    if not matched_indices:
        for result in results:
            result.source = "interpolated_before"
        return

    first = matched_indices[0]
    last = matched_indices[-1]
    for index in range(first):
        results[index].source = "interpolated_before"
    if first > 0:
        anchor_start = results[first].start
        slot = max(0.1, anchor_start / first)
        for index in range(first):
            results[index].start = max(0.0, anchor_start - ((first - index) * slot))
            results[index].end = max(0.0, anchor_start - ((first - index - 1) * slot))
            results[index].midi = results[first].midi

    for left, right in zip(matched_indices, matched_indices[1:], strict=False):
        if right - left <= 1:
            continue
        time_start = results[left].end
        duration = max(0.0, results[right].start - time_start)
        gaps = right - left
        for offset in range(1, gaps):
            index = left + offset
            fraction = offset / gaps
            results[index].source = "interpolated_between"
            results[index].start = time_start + (fraction * duration)
            results[index].end = time_start + (((offset + 1) / gaps) * duration)
            results[index].midi = _js_round(
                results[left].midi + (fraction * (results[right].midi - results[left].midi))
            )

    for index in range(last + 1, len(results)):
        results[index].source = "interpolated_after"
    if last < len(results) - 1:
        average_duration = sum(
            results[index].end - results[index].start for index in matched_indices
        )
        average_duration /= len(matched_indices)
        fallback = max(0.2, average_duration)
        for index in range(last + 1, len(results)):
            offset = (index - last) * fallback
            results[index].start = results[last].end + offset
            results[index].end = results[last].end + offset + fallback
            results[index].midi = results[last].midi


def _build_syllables(
    words: Sequence[_MutableWord], raw_lines: Sequence[str], language: str
) -> list[AlignedSyllable]:
    unbroken: list[AlignedSyllable] = []
    for word in words:
        parts = split_word(word.word, language) or [word.word]
        duration = max(0.01, (word.end - word.start) / len(parts))
        for index, syllable in enumerate(parts):
            start = word.start + (index * duration)
            end = word.start + ((index + 1) * duration)
            midi = word.midi
            if word.pitch_frames:
                aggregate = WordTimestamp(
                    word=word.word,
                    start=start,
                    end=end,
                    midi=word.midi,
                    pitch_frames=tuple(word.pitch_frames),
                )
                midi, _ = midi_for_range(aggregate, start, end)
            unbroken.append(AlignedSyllable(syllable=syllable, start=start, end=end, midi=midi))

    output: list[AlignedSyllable] = []
    position = 0
    for line_index, line in enumerate(raw_lines):
        count = sum(len(split_word(word, language)) for word in line.split() if word)
        output.extend(unbroken[position : position + count])
        position += count
        if line_index < len(raw_lines) - 1 and position < len(unbroken):
            next_syllable = unbroken[position]
            output.append(
                AlignedSyllable(
                    syllable="",
                    start=next_syllable.start,
                    end=next_syllable.start,
                    midi=0,
                    is_line_break=True,
                )
            )
    return output


def _coerce_word(value: WordTimestamp | Mapping[str, Any]) -> WordTimestamp:
    if isinstance(value, WordTimestamp):
        return value
    raw_frames = value.get("pitch_frames", value.get("pitchFrames", ()))
    frames = tuple(
        frame
        if isinstance(frame, PitchFrame)
        else PitchFrame(
            time=float(frame["time"]),
            midi=float(frame["midi"]),
            confidence=float(frame["confidence"]),
        )
        for frame in raw_frames
    )
    return WordTimestamp(
        word=str(value["word"]),
        start=float(value["start"]),
        end=float(value["end"]),
        midi=int(value.get("midi", 60)),
        pitch_frames=frames,
    )


def _coerce_pause(value: Pause | Mapping[str, Any]) -> Pause:
    if isinstance(value, Pause):
        return value
    return Pause(start=float(value["start"]), end=float(value["end"]))


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


__all__ = ["align_lyrics", "align_lyrics_with_debug", "midi_for_range"]
