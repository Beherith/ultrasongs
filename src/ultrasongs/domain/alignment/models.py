"""Typed inputs and outputs for deterministic lyric alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class PitchFrame:
    time: float
    midi: float
    confidence: float


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    midi: int
    pitch_frames: tuple[PitchFrame, ...] = ()

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("word end cannot be before its start")
        object.__setattr__(self, "pitch_frames", tuple(self.pitch_frames))


@dataclass(frozen=True, slots=True)
class Pause:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("pause end cannot be before its start")


@dataclass(frozen=True, slots=True)
class AlignedSyllable:
    syllable: str
    start: float
    end: float
    midi: int
    is_line_break: bool = False


WordSource = Literal[
    "sw_aligned",
    "interpolated_before",
    "interpolated_between",
    "interpolated_after",
]


@dataclass(frozen=True, slots=True)
class CharacterAlignment:
    lyric_character: str
    transcription_character: str
    transcription_word_index: int
    score: float


@dataclass(frozen=True, slots=True)
class AlignedWord:
    word: str
    line_index: int
    start: float
    end: float
    midi: int
    source: WordSource
    transcription_word_indices: tuple[int, ...] = ()
    character_alignments: tuple[CharacterAlignment, ...] = ()


@dataclass(frozen=True, slots=True)
class BacktrackDebugStep:
    lyric_index: int
    transcription_index: int
    matrix: Literal["M", "X", "Y"]
    score: float
    lyric_character: str
    transcription_character: str
    transcription_word_index: int
    transcription_word: str


@dataclass(frozen=True, slots=True)
class AlignmentSummary:
    total_lyric_words: int
    aligned_words: int
    interpolated_words: int
    total_syllables: int
    line_breaks: int


@dataclass(frozen=True, slots=True)
class AlignmentDebug:
    language: str
    lyric_character_count: int
    transcription_character_count: int
    transcription_word_count: int
    transcription_time_range: tuple[float, float]
    smith_waterman_max_score: float
    smith_waterman_max_position: tuple[int, int]
    backtrack: tuple[BacktrackDebugStep, ...]
    words: tuple[AlignedWord, ...]
    pauses: tuple[Pause, ...]
    summary: AlignmentSummary


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    syllables: tuple[AlignedSyllable, ...]
    debug: AlignmentDebug
    messages: tuple[str, ...] = field(default_factory=tuple)

