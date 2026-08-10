"""Syllabification using pyphen (replaces hyphen npm package)."""

import pyphen

from cli.logging_setup import get_logger

logger = get_logger("cli.syllabify")

# Whisper ISO 639-1 codes -> pyphen locale codes
_LANG_ALIASES: dict[str, str] = {
    "en": "en_US",
    "en-us": "en_US",
    "en-gb": "en_GB",
    "zh": "en_US",    # fallback — pyphen has no Chinese
    "zh-tw": "en_US",
    "pt": "pt_BR",
    "pt-br": "pt_BR",
    "pt-pt": "pt_BR",
    "de": "de_DE",
    "de-at": "de_DE",
    "de-ch": "de_DE",
    "fr": "fr_FR",
    "fr-ca": "fr_FR",
    "fr-be": "fr_FR",
    "es": "es_ES",
    "it": "it_IT",
    "nl": "nl_NL",
    "nl-be": "nl_NL",
    "pl": "pl_PL",
    "ru": "ru_RU",
    "sv": "sv_SE",
    "nb": "nb_NO",
    "no": "nb_NO",
    "da": "da_DK",
    "fi": "fi_FI",
    "ro": "ro_RO",
    "cs": "cs_CZ",
    "sk": "sk_SK",
    "hr": "hr_HR",
    "tr": "tr_TR",
    "ca": "ca_ES",
    "gl": "gl_ES",
    "sr-latn": "en_US",  # fallback
}

# Cached hyphenator instances
_hyphenators: dict[str, pyphen.Pyphen] = {}


def _get_hyphenator(lang: str) -> pyphen.Pyphen | None:
    """Get or create a pyphen hyphenator for the given language."""
    locale = _LANG_ALIASES.get(lang, lang.split("-")[0])
    if locale not in _hyphenators:
        try:
            _hyphenators[locale] = pyphen.Pyphen(lang=locale)
        except (ValueError, KeyError):
            # Language not available in pyphen
            _hyphenators[locale] = None  # type: ignore[assignment]
    return _hyphenators[locale]


def split_word(word: str, lang: str) -> list[str]:
    """Split a single word into syllables using pyphen.

    Args:
        word: The word to split.
        lang: ISO 639-1 language code.

    Returns:
        List of syllable strings. For words <= 2 chars or unsupported
        languages, returns the whole word as a single syllable.
    """
    clean = word.strip()
    if not clean or len(clean) <= 2:
        return [clean] if clean else []

    h = _get_hyphenator(lang)
    if h is None:
        return [clean]

    try:
        parts = h.divided(clean)
        parts = [p for p in parts if p]
        return parts if parts else [clean]
    except Exception:
        return [clean]


def syllabify_line(line: str, lang: str) -> list[list[str]]:
    """Split all words in a line into syllable arrays.

    Args:
        line: The text line to syllabify.
        lang: ISO 639-1 language code.

    Returns:
        List of syllable lists, one per word.
    """
    return [
        split_word(word, lang)
        for word in line.strip().split()
        if word
    ]
