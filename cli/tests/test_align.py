"""Tests for lyric alignment module."""

import pytest
from cli.align import (
    align_lyrics,
    normalize_char,
    phonetic_score,
    smith_waterman,
)
from cli.config import Config
from cli.pipeline_types import AlignedSyllable, CharacterTimestamp, Pause, PitchFrame, WordTimestamp


class TestNormalizeChar:
    def test_basic(self):
        assert normalize_char("A") == "a"
        assert normalize_char("Z") == "z"

    def test_diacritics(self):
        assert normalize_char("é") == "e"
        assert normalize_char("ü") == "u"
        assert normalize_char("ñ") == "n"

    def test_non_latin_characters_are_preserved(self):
        assert normalize_char("Привет") == "привет"


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
        assert "".join(s.syllable for s in singing) == lyrics

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

    def test_interpolation_does_not_stretch_words_across_long_gap(self):
        words = self._make_words([
            ("alpha", 0.0, 0.3),
            ("delta", 10.0, 10.3),
        ])

        result = align_lyrics("alpha beta gamma delta", words, "en", config=Config())
        singing = [s for s in result if not s.is_line_break]

        assert singing[-2].end <= 1.9 + 1e-9
        assert singing[-1].start == 10.0

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

    def test_logs_visual_backtrace_instead_of_word_summary(self, caplog):
        lyrics = "hello world"
        words = self._make_words([
            ("hello", 0.5, 1.0),
            ("world", 1.2, 1.7),
        ])

        with caplog.at_level("INFO", logger="cli.align"):
            align_lyrics(lyrics, words, "en", config=Config())

        assert "Smith-Waterman alignment:\n" in caplog.text
        assert "q: hello world" in caplog.text
        assert "s: hello world" in caplog.text
        assert "Word alignment summary:" not in caplog.text

    def test_amplitude_trims_quiet_word_edges(self):
        words = self._make_words([("sing", 0.0, 1.0)])
        frames = [
            PitchFrame(
                time=i / 100,
                midi=60,
                confidence=0.9 if 20 <= i <= 50 else 0.1,
                amplitude=0.8 if 20 <= i <= 50 else 0.01,
            )
            for i in range(101)
        ]

        result = align_lyrics("sing", words, "en", config=Config(), pitch_frames=frames)
        singing = [s for s in result if not s.is_line_break]

        assert len(singing) == 1
        assert singing[0].start == pytest.approx(0.2)
        assert singing[0].end == pytest.approx(0.51)

    def test_sustained_pitch_change_creates_continuation_note(self):
        words = self._make_words([("sing", 0.0, 1.0)])
        frames = [
            PitchFrame(
                time=i / 100,
                midi=60 if i <= 40 else 64,
                confidence=0.9 if 10 <= i <= 80 else 0.1,
                amplitude=0.8 if 10 <= i <= 80 else 0.01,
            )
            for i in range(101)
        ]

        result = align_lyrics("sing", words, "en", config=Config(), pitch_frames=frames)
        singing = [s for s in result if not s.is_line_break]

        assert [s.midi for s in singing] == [60, 64]
        assert [s.syllable for s in singing] == ["sing", ""]

    def test_words_sharing_whisper_token_divide_its_time(self):
        words = self._make_words([("raiseyour", 1.0, 2.0)])

        result = align_lyrics("raise your", words, "en", config=Config())
        singing = [s for s in result if not s.is_line_break]

        assert singing[0].start == 1.0
        assert singing[0].end == singing[1].start
        assert singing[1].end == 2.0

    def test_whisperx_characters_anchor_syllable_intervals(self):
        characters = [
            CharacterTimestamp("h", 1.00, 1.10, 0.9),
            CharacterTimestamp("e", 1.10, 1.20, 0.9),
            CharacterTimestamp("l", 1.20, 1.30, 0.9),
            CharacterTimestamp("l", 1.60, 1.70, 0.9),
            CharacterTimestamp("o", 1.70, 1.80, 0.9),
        ]
        words = [WordTimestamp("hello", 1.0, 1.8, 60, characters=characters)]

        result = align_lyrics("hello", words, "en", config=Config())
        singing = [s for s in result if not s.is_line_break]

        assert [s.syllable for s in singing] == ["hel", "lo"]
        assert singing[0].start == 1.0
        assert singing[0].end == 1.3
        assert singing[1].start == 1.6
        assert singing[1].end == 1.8

    def test_pause_separates_overlapping_word_timestamps(self):
        words = self._make_words([
            ("one", 0.0, 0.8),
            ("two", 0.7, 1.5),
        ])

        result = align_lyrics(
            "one two",
            words,
            "en",
            pauses=[Pause(0.6, 0.9)],
            config=Config(),
        )
        singing = [s for s in result if not s.is_line_break]

        assert singing[0].end == 0.6
        assert singing[1].start == 0.9
