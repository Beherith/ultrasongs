"""UltraStar song-similarity metrics."""

from .similarity import (
    MatchedNotes,
    SimilarityResult,
    align_notes,
    compare_song_files,
    compare_songs,
    octave_corrected_distance,
)

__all__ = [
    "MatchedNotes",
    "SimilarityResult",
    "align_notes",
    "compare_song_files",
    "compare_songs",
    "octave_corrected_distance",
]
