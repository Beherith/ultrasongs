"""Convert automatic alignment output into an UltraStar song."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ultrasongs.domain.alignment import AlignedSyllable

from .beat_mapping import ms_to_beats
from .models import LineBreak, NoteType, UltrastarMetadata, UltrastarNote, UltrastarSong


@dataclass(frozen=True, slots=True)
class GenerationResult:
    song: UltrastarSong
    bpm: float
    gap_ms: float


def generate_song_from_alignment(
    aligned: Sequence[AlignedSyllable | Mapping[str, Any]],
    *,
    title: str,
    artist: str,
    mp3_filename: str,
    bpm: float,
    gap_ms: float | None = None,
    video_filename: str | None = None,
) -> GenerationResult:
    """Apply the legacy automatic beat mapping and overlap rules."""

    syllables = tuple(_coerce_syllable(item) for item in aligned)
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero")
    if gap_ms is None:
        first = next(
            (item for item in syllables if not item.is_line_break and item.start > 0),
            None,
        )
        gap_ms = max(0.0, first.start * 1000 - 500) if first is not None else 0.0

    events: list[UltrastarNote | LineBreak] = []
    previous_end = -1
    for syllable in syllables:
        if syllable.is_line_break:
            next_note_beat = int(ms_to_beats(syllable.start * 1000, bpm, gap_ms))
            last_note_index = next(
                (
                    index
                    for index in range(len(events) - 1, -1, -1)
                    if isinstance(events[index], UltrastarNote)
                ),
                None,
            )
            if last_note_index is not None:
                last = events[last_note_index]
                assert isinstance(last, UltrastarNote)
                maximum_duration = max(1, next_note_beat - 2 - int(last.start_beat))
                if last.duration_beats > maximum_duration:
                    capped = UltrastarNote(
                        note_type=last.note_type,
                        start_beat=last.start_beat,
                        duration_beats=maximum_duration,
                        pitch=last.pitch,
                        lyric=last.lyric,
                    )
                    events[last_note_index] = capped
                    previous_end = int(capped.end_beat)

            line_break_beat = max(
                previous_end + 1 if previous_end >= 0 else 0,
                next_note_beat - 4,
            )
            events.append(LineBreak(line_break_beat))
            previous_end = line_break_beat
            continue

        start_beat = int(ms_to_beats(syllable.start * 1000, bpm, gap_ms))
        end_beat = int(ms_to_beats(syllable.end * 1000, bpm, gap_ms))
        duration = max(1, end_beat - start_beat)
        adjusted_start = max(start_beat, previous_end + 1) if previous_end >= 0 else start_beat
        note = UltrastarNote(
            note_type=NoteType.NORMAL,
            start_beat=adjusted_start,
            duration_beats=duration,
            pitch=syllable.midi,
            lyric=syllable.syllable,
        )
        events.append(note)
        previous_end = int(note.end_beat)

    song = UltrastarSong(
        metadata=UltrastarMetadata(
            title=title,
            artist=artist,
            mp3=mp3_filename,
            bpm=bpm,
            gap_ms=gap_ms,
            video=video_filename,
        ),
        events=tuple(events),
    )
    return GenerationResult(song=song, bpm=bpm, gap_ms=gap_ms)


def _coerce_syllable(value: AlignedSyllable | Mapping[str, Any]) -> AlignedSyllable:
    if isinstance(value, AlignedSyllable):
        return value
    return AlignedSyllable(
        syllable=str(value.get("syllable", "")),
        start=float(value["start"]),
        end=float(value["end"]),
        midi=int(value.get("midi", 60)),
        is_line_break=bool(value.get("is_line_break", value.get("isLineBreak", False))),
    )
