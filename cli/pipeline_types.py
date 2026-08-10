"""Shared data types for the CLI pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PitchFrame:
    time: float
    midi: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PitchFrame":
        return cls(**d)


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    midi: int
    pitch_frames: list[PitchFrame] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "midi": self.midi,
            "pitch_frames": [pf.to_dict() for pf in self.pitch_frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=d["word"],
            start=d["start"],
            end=d["end"],
            midi=d["midi"],
            pitch_frames=[PitchFrame.from_dict(pf) for pf in d.get("pitch_frames", [])],
        )


@dataclass
class Pause:
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pause":
        return cls(**d)


@dataclass
class AlignedSyllable:
    syllable: str
    start: float
    end: float
    midi: int
    is_line_break: bool = False
    pitch_end: float = 0.0


@dataclass
class TranscribeResult:
    words: list[WordTimestamp]
    language: str
    vocals_path: str
    accompaniment_path: str
    pauses: list[Pause]

    def to_dict(self) -> dict[str, Any]:
        return {
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "vocals_path": self.vocals_path,
            "accompaniment_path": self.accompaniment_path,
            "pauses": [p.to_dict() for p in self.pauses],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscribeResult":
        return cls(
            words=[WordTimestamp.from_dict(w) for w in d["words"]],
            language=d["language"],
            vocals_path=d["vocals_path"],
            accompaniment_path=d["accompaniment_path"],
            pauses=[Pause.from_dict(p) for p in d.get("pauses", [])],
        )


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
