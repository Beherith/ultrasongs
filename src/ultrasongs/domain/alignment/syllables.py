"""Language-aware word splitting with an optional TeX-pattern backend."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

LANGUAGE_ALIASES: dict[str, str] = {
    "en-us": "en_US",
    "en-gb": "en_GB",
    "zh-tw": "en_US",
    "pt-br": "pt_BR",
    "pt-pt": "pt_PT",
    "de-at": "de_AT",
    "de-ch": "de_CH",
    "fr-ca": "fr",
    "fr-be": "fr",
    "nl-be": "nl_NL",
    "sr-latn": "en_US",
    "no": "nb_NO",
}

LANGUAGE_CODES: dict[str, str] = {
    "it": "it_IT",
    "en": "en_US",
    "es": "es",
    "fr": "fr",
    "de": "de_DE",
    "pt": "pt_PT",
    "ca": "ca",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "ru": "ru_RU",
    "sv": "sv_SE",
    "nb": "nb_NO",
    "da": "da_DK",
    "fi": "fi_FI",
    "ro": "ro_RO",
    "cs": "cs_CZ",
    "sk": "sk_SK",
    "hr": "hr_HR",
    "tr": "tr_TR",
    "eu": "eu",
    "gl": "gl",
    "af": "af_ZA",
}

VOWELS = frozenset("aeiouy")


def split_word(word: str, language: str) -> list[str]:
    """Split a word using pyphen when present, with a stable local fallback."""
    clean = word.strip()
    if not clean or len(clean) <= 2:
        return [clean] if clean else []

    hyphenator = _resolve_hyphenator(language)
    if hyphenator is not None:
        try:
            parts = [
                part
                for part in hyphenator.inserted(clean, hyphen="\u00ad").split("\u00ad")
                if part
            ]
            if parts:
                return parts
        except (KeyError, TypeError, ValueError):
            pass
    return _fallback_split(clean)


def syllabify_line(line: str, language: str) -> list[tuple[str, list[str]]]:
    return [(word, split_word(word, language)) for word in line.strip().split() if word]


@lru_cache(maxsize=64)
def _resolve_hyphenator(language: str) -> Any | None:
    try:
        import pyphen
    except ImportError:
        return None

    normalized = language.strip().lower()
    code = LANGUAGE_ALIASES.get(normalized)
    if code is None:
        base = normalized.split("-", 1)[0]
        code = LANGUAGE_CODES.get(normalized) or LANGUAGE_CODES.get(base)
    if code is None or code not in pyphen.LANGUAGES:
        return None
    return pyphen.Pyphen(lang=code)


def _fallback_split(word: str) -> list[str]:
    """Conservatively split between vowel nuclei without changing characters."""
    letter_positions = [index for index, char in enumerate(word) if char.isalpha()]
    if len(letter_positions) <= 3:
        return [word]

    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", word.lower())
        if not unicodedata.combining(char)
    )
    nuclei = [match.span() for match in re.finditer(r"[aeiouy]+", normalized)]
    if len(nuclei) <= 1:
        return [word]

    boundaries: list[int] = []
    for previous, following in zip(nuclei, nuclei[1:], strict=False):
        consonants = following[0] - previous[1]
        # Keep one consonant as the onset of the following syllable. With a
        # single consonant this produces e.g. mi-ning; with a cluster it keeps
        # the final consonant with the next nucleus.
        boundary = previous[1] if consonants <= 1 else following[0] - 1
        if 0 < boundary < len(word):
            boundaries.append(boundary)

    if not boundaries:
        return [word]
    parts: list[str] = []
    start = 0
    for boundary in boundaries:
        if boundary > start:
            parts.append(word[start:boundary])
            start = boundary
    parts.append(word[start:])
    return [part for part in parts if part] or [word]


__all__ = ["split_word", "syllabify_line"]
