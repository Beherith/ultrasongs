"""Typed domain objects for UltraStar song files."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias


class NoteType(StrEnum):
    """Supported UltraStar note markers."""

    NORMAL = ":"
    GOLDEN = "*"
    FREESTYLE = "F"


@dataclass(frozen=True, slots=True)
class UltrastarNote:
    """One singable note, expressed in UltraStar beat units (sixteenth notes)."""

    note_type: NoteType
    start_beat: float
    duration_beats: float
    pitch: int
    lyric: str

    def __post_init__(self) -> None:
        if self.duration_beats < 0:
            raise ValueError("note duration cannot be negative")

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats

    @property
    def chorus(self) -> bool:
        """Compatibility alias for the old scorer's golden-note flag."""

        return self.note_type is NoteType.GOLDEN


@dataclass(frozen=True, slots=True)
class LineBreak:
    """An UltraStar lyric-line separator."""

    start_beat: float
    end_beat: float | None = None


UltrastarEvent: TypeAlias = UltrastarNote | LineBreak


@dataclass(frozen=True, slots=True)
class UltrastarMetadata:
    """Known UltraStar headers plus lossless storage for extension headers."""

    title: str = ""
    artist: str = ""
    mp3: str = ""
    bpm: float = 120.0
    gap_ms: float = 0.0
    video: str | None = None
    extras: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError("BPM must be greater than zero")
        normalized = {str(key).upper(): str(value) for key, value in self.extras.items()}
        object.__setattr__(self, "extras", MappingProxyType(normalized))

    @property
    def gap(self) -> float:
        """Compatibility alias for code that uses the UltraStar header name."""

        return self.gap_ms

    def get(self, key: str, default: str | None = None) -> str | None:
        """Read a header using its conventional case-insensitive name."""

        known: dict[str, str | None] = {
            "TITLE": self.title,
            "ARTIST": self.artist,
            "MP3": self.mp3,
            "BPM": format_number(self.bpm),
            "GAP": format_number(self.gap_ms),
            "VIDEO": self.video,
        }
        key = key.upper()
        value = known.get(key, self.extras.get(key))
        return default if value is None else value

    def as_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "TITLE": self.title,
            "ARTIST": self.artist,
            "MP3": self.mp3,
        }
        if self.video is not None:
            headers["VIDEO"] = self.video
        headers["BPM"] = format_number(self.bpm)
        headers["GAP"] = format_number(self.gap_ms)
        headers.update(self.extras)
        return headers


@dataclass(frozen=True, slots=True)
class UltrastarSong:
    metadata: UltrastarMetadata
    events: tuple[UltrastarEvent, ...] = ()

    @property
    def notes(self) -> tuple[UltrastarNote, ...]:
        return tuple(event for event in self.events if isinstance(event, UltrastarNote))

    @property
    def line_breaks(self) -> tuple[LineBreak, ...]:
        return tuple(event for event in self.events if isinstance(event, LineBreak))

    def verses(self) -> Iterator[tuple[UltrastarNote, ...]]:
        verse: list[UltrastarNote] = []
        for event in self.events:
            if isinstance(event, LineBreak):
                if verse:
                    yield tuple(verse)
                    verse.clear()
            else:
                verse.append(event)
        if verse:
            yield tuple(verse)


def format_number(value: float) -> str:
    """Format numeric song values without unnecessary decimal noise."""

    return str(int(value)) if float(value).is_integer() else format(value, ".12g")
