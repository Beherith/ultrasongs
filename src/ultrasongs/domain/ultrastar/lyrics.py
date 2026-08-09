"""Best-effort lyric reconstruction for reference-song validation."""

from __future__ import annotations

from .models import UltrastarNote, UltrastarSong

_CONTINUATION_MARKERS = ("-", "~")


def reconstruct_lyrics(song: UltrastarSong) -> str:
    """Reconstruct reviewable lyric lines from UltraStar note events.

    UltraStar files can preserve word boundaries by retaining leading spaces in
    lyric tokens. Some generators, including the legacy Ultrasongs exporter,
    omit that distinction. When a verse contains explicit leading spaces we
    honor them and concatenate unspaced syllables; otherwise we conservatively
    separate tokens with spaces. The validation UI must always let the user
    review and correct this best-effort text before starting an expensive run.
    """

    return "\n".join(_reconstruct_verse(verse) for verse in song.verses())


def _reconstruct_verse(notes: tuple[UltrastarNote, ...]) -> str:
    if not notes:
        return ""
    has_explicit_boundaries = any(
        note.lyric[:1].isspace() for note in notes[1:] if note.lyric
    )
    if not has_explicit_boundaries:
        return " ".join(_clean_token(note.lyric) for note in notes).strip()

    output = ""
    previous_continues = False
    for index, note in enumerate(notes):
        raw = note.lyric
        explicit_new_word = index > 0 and bool(raw[:1].isspace())
        token = _clean_token(raw)
        if not token:
            continue
        if output and explicit_new_word and not previous_continues:
            output += " "
        output += token
        previous_continues = raw.rstrip().endswith(_CONTINUATION_MARKERS)
    return output.strip()


def _clean_token(value: str) -> str:
    token = value.strip()
    while token.endswith(_CONTINUATION_MARKERS):
        token = token[:-1]
    while token.startswith(_CONTINUATION_MARKERS):
        token = token[1:]
    return token
