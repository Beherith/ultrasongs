"""Helpers for the faster-whisper -> lyric chunks -> WhisperX pipeline."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from cli.align import normalize_char, smith_waterman
from cli.pipeline_types import Pause


@dataclass(frozen=True)
class ApproximateLyricWord:
    """A lyric word placed approximately on the faster-whisper timeline."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class LyricChunk:
    """Authoritative lyric text paired with one pause-delimited audio range."""

    text: str
    start: float
    end: float
    first_word: int
    last_word: int


def _character_sequence(words: list[str]) -> list[tuple[str, int]]:
    sequence: list[tuple[str, int]] = []
    for word_index, word in enumerate(words):
        sequence.extend((char, word_index) for char in normalize_char(word))
        if word_index < len(words) - 1:
            sequence.append((" ", word_index))
    return sequence


def align_lyrics_approximately(
    lyrics: str,
    transcript_words: list[dict[str, Any]],
) -> list[ApproximateLyricWord]:
    """Map authoritative lyric words onto approximate ASR word timestamps.

    This intentionally mirrors the character-level Smith-Waterman mapping used
    by the final lyric alignment. Its timestamps are only used to decide which
    side of a long pause each lyric word belongs on; WhisperX replaces them.
    """
    lyric_tokens = lyrics.split()
    if not lyric_tokens:
        return []

    usable_transcript = [
        word for word in transcript_words
        if str(word.get("word") or "").strip()
        and word.get("start") is not None
        and word.get("end") is not None
    ]
    if not usable_transcript:
        return [
            ApproximateLyricWord(word, index * 0.3, (index + 1) * 0.3)
            for index, word in enumerate(lyric_tokens)
        ]

    lyric_chars = _character_sequence(lyric_tokens)
    transcript_chars = _character_sequence([
        str(word["word"]).strip() for word in usable_transcript
    ])
    alignment = smith_waterman(
        [char for char, _ in lyric_chars],
        [char for char, _ in transcript_chars],
    )

    mapped: list[set[int]] = [set() for _ in lyric_tokens]
    for step in alignment["backtrack"]:
        if step["matrix"] != "M" or step["i"] <= 0 or step["j"] <= 0:
            continue
        lyric_char, lyric_index = lyric_chars[step["i"] - 1]
        transcript_char, transcript_index = transcript_chars[step["j"] - 1]
        if lyric_char != " " and transcript_char != " ":
            mapped[lyric_index].add(transcript_index)

    starts: list[float | None] = [None] * len(lyric_tokens)
    ends: list[float | None] = [None] * len(lyric_tokens)
    for index, transcript_indexes in enumerate(mapped):
        if not transcript_indexes:
            continue
        starts[index] = min(float(usable_transcript[i]["start"]) for i in transcript_indexes)
        ends[index] = max(float(usable_transcript[i]["end"]) for i in transcript_indexes)

    # A single ASR token can cover multiple lyric words. Split its approximate
    # interval by character count so a pause cannot assign the whole group to
    # an arbitrary side.
    index = 0
    while index < len(lyric_tokens):
        signature = tuple(sorted(mapped[index]))
        group_end = index + 1
        while (
            signature
            and group_end < len(lyric_tokens)
            and tuple(sorted(mapped[group_end])) == signature
        ):
            group_end += 1
        if group_end - index > 1:
            group_start_time = min(float(starts[i]) for i in range(index, group_end))
            group_end_time = max(float(ends[i]) for i in range(index, group_end))
            weights = [max(1, len(normalize_char(lyric_tokens[i]))) for i in range(index, group_end)]
            total_weight = sum(weights)
            cursor = group_start_time
            for word_index, weight in zip(range(index, group_end), weights):
                next_cursor = cursor + (group_end_time - group_start_time) * weight / total_weight
                starts[word_index], ends[word_index] = cursor, next_cursor
                cursor = next_cursor
        index = group_end

    matched = [index for index, start in enumerate(starts) if start is not None]
    timeline_start = float(usable_transcript[0]["start"])
    timeline_end = float(usable_transcript[-1]["end"])
    if not matched:
        slot = max(0.01, (timeline_end - timeline_start) / len(lyric_tokens))
        starts = [timeline_start + index * slot for index in range(len(lyric_tokens))]
        ends = [timeline_start + (index + 1) * slot for index in range(len(lyric_tokens))]
    else:
        first, last = matched[0], matched[-1]
        if first:
            anchor = float(starts[first])
            slot = max(0.01, (anchor - timeline_start) / first)
            for word_index in range(first):
                starts[word_index] = timeline_start + word_index * slot
                ends[word_index] = timeline_start + (word_index + 1) * slot

        for left, right in zip(matched, matched[1:]):
            missing = right - left - 1
            if missing <= 0:
                continue
            gap_start = float(ends[left])
            gap_end = float(starts[right])
            slot = max(0.01, (gap_end - gap_start) / missing)
            for offset in range(1, missing + 1):
                word_index = left + offset
                starts[word_index] = gap_start + (offset - 1) * slot
                ends[word_index] = min(gap_end, gap_start + offset * slot)

        if last < len(lyric_tokens) - 1:
            remaining = len(lyric_tokens) - last - 1
            anchor = float(ends[last])
            slot = max(0.01, (timeline_end - anchor) / remaining)
            for offset in range(1, remaining + 1):
                word_index = last + offset
                starts[word_index] = anchor + (offset - 1) * slot
                ends[word_index] = anchor + offset * slot

    output: list[ApproximateLyricWord] = []
    previous_midpoint = 0.0
    for word, start, end in zip(lyric_tokens, starts, ends):
        resolved_start = max(0.0, float(start))
        resolved_end = max(resolved_start, float(end))
        midpoint = max(previous_midpoint, (resolved_start + resolved_end) / 2)
        duration = resolved_end - resolved_start
        resolved_start = max(0.0, midpoint - duration / 2)
        resolved_end = midpoint + duration / 2
        previous_midpoint = midpoint
        output.append(ApproximateLyricWord(word, resolved_start, resolved_end))
    return output


def build_lyric_chunks(
    lyrics: str,
    transcript_words: list[dict[str, Any]],
    pauses: list[Pause],
    audio_duration: float,
    min_pause_seconds: float = 1.0,
) -> list[LyricChunk]:
    """Split aligned lyrics at vocal pauses strictly longer than the threshold."""
    approximate_words = align_lyrics_approximately(lyrics, transcript_words)
    if not approximate_words or audio_duration <= 0:
        return []

    cut_points = sorted({
        (max(0.0, pause.start) + min(audio_duration, pause.end)) / 2
        for pause in pauses
        if pause.end - pause.start > min_pause_seconds
        and pause.end > 0
        and pause.start < audio_duration
    })
    cut_points = [cut for cut in cut_points if 0 < cut < audio_duration]

    regions: list[list[tuple[int, ApproximateLyricWord]]] = [
        [] for _ in range(len(cut_points) + 1)
    ]
    previous_region = 0
    for word_index, word in enumerate(approximate_words):
        midpoint = (word.start + word.end) / 2
        region = max(previous_region, bisect_right(cut_points, midpoint))
        region = min(region, len(regions) - 1)
        regions[region].append((word_index, word))
        previous_region = region

    chunks: list[LyricChunk] = []
    boundaries = [0.0, *cut_points, audio_duration]
    for region_index, region_words in enumerate(regions):
        if not region_words:
            continue
        chunks.append(LyricChunk(
            text=" ".join(word.word for _, word in region_words),
            start=boundaries[region_index],
            end=boundaries[region_index + 1],
            first_word=region_words[0][0],
            last_word=region_words[-1][0],
        ))
    return chunks


def slice_audio(audio: Any, start: float, end: float, audio_duration: float) -> Any:
    """Return the samples for a global time range without assuming a sample rate."""
    sample_count = len(audio)
    if sample_count == 0 or audio_duration <= 0:
        return audio[0:0]
    samples_per_second = sample_count / audio_duration
    first = max(0, min(sample_count, round(start * samples_per_second)))
    last = max(first, min(sample_count, round(end * samples_per_second)))
    return audio[first:last]


def offset_aligned_words(
    words: list[dict[str, Any]],
    offset: float,
) -> list[dict[str, Any]]:
    """Translate chunk-relative WhisperX word and character times to song time."""
    output: list[dict[str, Any]] = []
    for source in words:
        word = dict(source)
        word["start"] = float(word["start"]) + offset
        word["end"] = float(word["end"]) + offset
        word["characters"] = [
            {
                **character,
                "start": (
                    float(character["start"]) + offset
                    if character.get("start") is not None else None
                ),
                "end": (
                    float(character["end"]) + offset
                    if character.get("end") is not None else None
                ),
            }
            for character in source.get("characters", [])
        ]
        output.append(word)
    return output
