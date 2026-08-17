"""Shared data types for the CLI pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PitchFrame:
    time: float
    midi: int
    confidence: float
    amplitude: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PitchFrame":
        return cls(**d)


@dataclass
class CharacterTimestamp:
    char: str
    start: float | None = None
    end: float | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CharacterTimestamp":
        return cls(
            char=d["char"],
            start=d.get("start"),
            end=d.get("end"),
            score=d.get("score"),
        )


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    midi: int
    pitch_frames: list[PitchFrame] = field(default_factory=list)
    characters: list[CharacterTimestamp] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "midi": self.midi,
            "pitch_frames": [pf.to_dict() for pf in self.pitch_frames],
            "characters": [char.to_dict() for char in self.characters],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=d["word"],
            start=d["start"],
            end=d["end"],
            midi=d["midi"],
            pitch_frames=[PitchFrame.from_dict(pf) for pf in d.get("pitch_frames", [])],
            characters=[CharacterTimestamp.from_dict(char) for char in d.get("characters", [])],
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
class BpmResult:
    bpm: float
    first_beat_ms: float
    stable: bool
    chunk_bpms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BpmResult":
        return cls(
            bpm=d["bpm"],
            first_beat_ms=d["first_beat_ms"],
            stable=d["stable"],
            chunk_bpms=list(d.get("chunk_bpms", [])),
        )


@dataclass
class TranscribeResult:
    words: list[WordTimestamp]
    language: str
    vocals_path: str
    accompaniment_path: str
    pauses: list[Pause]
    pitch_frames: list[PitchFrame] = field(default_factory=list)
    bpm: float | None = None
    bpm_result: BpmResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "vocals_path": self.vocals_path,
            "accompaniment_path": self.accompaniment_path,
            "pauses": [p.to_dict() for p in self.pauses],
            "pitch_frames": [pf.to_dict() for pf in self.pitch_frames],
            "bpm": self.bpm,
            "bpm_result": self.bpm_result.to_dict() if self.bpm_result else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscribeResult":
        words = [WordTimestamp.from_dict(w) for w in d["words"]]
        pitch_frames = [PitchFrame.from_dict(pf) for pf in d.get("pitch_frames", [])]
        if not pitch_frames:
            by_time = {
                pf.time: pf
                for word in words
                for pf in word.pitch_frames
            }
            pitch_frames = [by_time[t] for t in sorted(by_time)]

        bpm_result = (
            BpmResult.from_dict(d["bpm_result"])
            if d.get("bpm_result")
            else None
        )
        return cls(
            words=words,
            language=d["language"],
            vocals_path=d["vocals_path"],
            accompaniment_path=d["accompaniment_path"],
            pauses=[Pause.from_dict(p) for p in d.get("pauses", [])],
            pitch_frames=pitch_frames,
            bpm=d.get("bpm"),
            bpm_result=bpm_result,
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
