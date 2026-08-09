"""Deterministic lyrics-to-transcription alignment."""

from .aligner import align_lyrics, align_lyrics_with_debug, midi_for_range
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
    WordTimestamp,
)
from .normalization import normalize_character, normalize_word, phonetic_score
from .smith_waterman import BacktrackStep, SmithWatermanResult, smith_waterman
from .syllables import split_word, syllabify_line

__all__ = [
    "AlignedSyllable",
    "AlignedWord",
    "AlignmentDebug",
    "AlignmentResult",
    "AlignmentSummary",
    "BacktrackDebugStep",
    "BacktrackStep",
    "CharacterAlignment",
    "Pause",
    "PitchFrame",
    "SmithWatermanResult",
    "WordTimestamp",
    "align_lyrics",
    "align_lyrics_with_debug",
    "midi_for_range",
    "normalize_character",
    "normalize_word",
    "phonetic_score",
    "smith_waterman",
    "split_word",
    "syllabify_line",
]

