"""Character normalization and phonetic similarity shared by alignment."""

from __future__ import annotations

import unicodedata

PHONETIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("a", "e", "i"),
    ("o", "u"),
    ("s", "z", "c"),
    ("t", "d"),
    ("p", "b"),
    ("k", "g"),
    ("f", "v", "w"),
    ("m", "n"),
    ("l", "r"),
    ("y", "i"),
    ("h", "j"),
    ("b", "v"),
)

PHONETIC_CROSS_PAIRS: tuple[tuple[str, str], ...] = (
    ("a", "o"),
    ("a", "u"),
    ("e", "i"),
    ("e", "o"),
    ("i", "y"),
    ("s", "sh"),
    ("z", "zh"),
    ("f", "ph"),
    ("c", "k"),
    ("q", "k"),
    ("w", "u"),
    ("r", "l"),
    ("b", "p"),
    ("d", "t"),
    ("g", "k"),
)


def normalize_character(character: str) -> str:
    """Match JavaScript's lower-case NFD normalization and accent removal."""
    decomposed = unicodedata.normalize("NFD", character.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_word(word: str, *, alphanumeric_only: bool = False) -> str:
    normalized = normalize_character(word)
    if alphanumeric_only:
        # TypeScript's parity target is deliberately ASCII-only here.
        return "".join(char for char in normalized if char.isascii() and char.isalnum())
    return normalized


def phonetic_score(left: str, right: str) -> float:
    """Return the current TypeScript implementation's character similarity."""
    if left == right:
        return 1.0
    for group in PHONETIC_GROUPS:
        if left in group and right in group:
            return 0.6 - (0.1 * abs(group.index(left) - group.index(right)))
    for first, second in PHONETIC_CROSS_PAIRS:
        if (left == first and right == second) or (left == second and right == first):
            return 0.5
    return -0.3

