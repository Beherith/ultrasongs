"""Tests for lyric alignment module."""

import pytest
from cli.align import (
    align_lyrics,
    normalize_char,
    phonetic_score,
    smith_waterman,
)
from cli.config import Config
from cli.types import AlignedSyllable, Pause, WordTimestamp


class TestNormalizeChar:
    def test_basic(self):
        assert normalize_char("A") == "a"
        assert normalize_char("Z") == "z"

    def test_diacritics(self):
        assert normalize_char("é") == "e"
        assert normalize_char("ü") == "u"
        assert normalize_char("ñ") == "n"


class TestPhoneticScore:
    def test_exact_match(self):
        assert phonetic_score("a", "a") == 1.0

    def test_phonetic_group(self):
        # a, e, i are in the same group
        score = phonetic_score("a", "e")
        assert 0.0 < score < 1.0

    def test_mismatch(self):
        assert phonetic_score("a", "z") < 0

    def test_cross_pair(self):
        score = phonetic_score("a", "o")
        assert score == 0.5


class TestSmithWaterman:
    def test_identical(self):
        result = smith_waterman(list("hello"), list("hello"))
        assert result["maxScore"] > 0
        assert len(result["backtrack"]) > 0

    def test_partial_match(self):
        result = smith_waterman(list("hell"), list("hello"))
        assert result["maxScore"] > 0

    def test_empty(self):
        result = smith_waterman([], list("hello"))
        assert result["maxScore"] == 0


class TestAlignLyrics:
    def _make_words(self, words: list[tuple[str, float, float]]) -> list[WordTimestamp]:
        return [
            WordTimestamp(word=w, start=s, end=e, midi=60)
            for w, s, e in words
        ]

    def test_basic_alignment(self):
        lyrics = "hello world"
        words = self._make_words([
            ("hello", 0.5, 1.0),
            ("world", 1.2, 1.7),
        ])
        result = align_lyrics(lyrics, words, "en", config=Config())
        assert len(result) > 0
        # Should have syllables for both words
        singing = [s for s in result if not s.is_line_break]
        assert len(singing) >= 2

    def test_line_breaks(self):
        lyrics = "first line\nsecond line"
        words = self._make_words([
            ("first", 0.0, 0.5),
            ("line", 0.6, 1.0),
            ("second", 2.0, 2.5),
            ("line", 2.6, 3.0),
        ])
        result = align_lyrics(lyrics, words, "en", config=Config())
        linebreaks = [s for s in result if s.is_line_break]
        assert len(linebreaks) == 1  # one break between two lines

    def test_partial_match(self):
        lyrics = "hello world today"
        words = self._make_words([
            ("hello", 0.5, 1.0),
            ("world", 1.2, 1.7),
        ])
        result = align_lyrics(lyrics, words, "en", config=Config())
        # "today" should be interpolated
        assert len(result) > 0

    def test_empty_lyrics(self):
        words = self._make_words([("hello", 0.5, 1.0)])
        result = align_lyrics("", words, "en", config=Config())
        assert result == []

    def test_timestamps_ordered(self):
        lyrics = "one two three"
        words = self._make_words([
            ("one", 0.0, 0.3),
            ("two", 0.4, 0.7),
            ("three", 0.8, 1.1),
        ])
        result = align_lyrics(lyrics, words, "en", config=Config())
        singing = [s for s in result if not s.is_line_break]
        for i in range(1, len(singing)):
            assert singing[i].start >= singing[i - 1].start
