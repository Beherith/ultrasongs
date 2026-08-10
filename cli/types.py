"""Shared data types for the CLI pipeline."""

from dataclasses import dataclass, field


@dataclass
class PitchFrame:
    time: float
    midi: int
    confidence: float


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    midi: int
    pitch_frames: list[PitchFrame] = field(default_factory=list)


@dataclass
class Pause:
    start: float
    end: float


@dataclass
class AlignedSyllable:
    syllable: str
    start: float
    end: float
    midi: int
    is_line_break: bool = False


@dataclass
class TranscribeResult:
    words: list[WordTimestamp]
    language: str
    vocals_path: str
    accompaniment_path: str
    pauses: list[Pause]


@dataclass
class UltrastarNote:
    note_type: str       # ":", "*", "-"
    start_beat: int
    duration: int        # 0 for line breaks
    pitch: int           # 0 for line breaks
    syllable: str        # "" for line breaks


@dataclass
class UltrastarMeta:
    title: str
    artist: str
    mp3: str             # filename within output package
    bpm: float
    gap: int             # milliseconds
    video: str | None = None
