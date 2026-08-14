"""Tests for syllabification module."""

import pytest
from cli.syllabify import split_word, syllabify_line


class TestSplitWord:
    def test_empty_word(self):
        assert split_word("", "en") == []

    def test_short_word(self):
        assert split_word("I", "en") == ["I"]
        assert split_word("am", "en") == ["am"]

    def test_simple_word(self):
        syllables = split_word("hello", "en")
        assert len(syllables) >= 1
        assert "".join(syllables) == "hello"

    def test_longer_word(self):
        assert split_word("beautiful", "en") == ["beau", "ti", "ful"]

    def test_preserves_punctuation(self):
        assert split_word("rejoice!", "en") == ["re", "joice!"]

    def test_german(self):
        syllables = split_word("schneemann", "de")
        assert len(syllables) >= 1

    def test_spanish(self):
        syllables = split_word("hola", "es")
        assert len(syllables) >= 1

    def test_unsupported_language(self):
        # Should return whole word
        assert split_word("hello", "xx") == ["hello"]


class TestSyllabifyLine:
    def test_single_word(self):
        result = syllabify_line("hello", "en")
        assert len(result) == 1

    def test_multiple_words(self):
        result = syllabify_line("hello world", "en")
        assert len(result) == 2

    def test_empty_line(self):
        result = syllabify_line("", "en")
        assert result == []
