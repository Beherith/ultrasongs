"""Tests for medley (chorus) detection."""

from pathlib import Path

from cli.medley import detect_medley, normalize_lyric_line

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestNormalizeLyricLine:
    def test_lowercase(self):
        assert normalize_lyric_line("Hello World") == "hello world"

    def test_punctuation_stripped(self):
        assert normalize_lyric_line("Sing, sing, sing with me!") == "sing sing sing with me"

    def test_whitespace_collapsed(self):
        assert normalize_lyric_line("  a \t b  c ") == "a b c"

    def test_trailing_ellipsis_equals_plain(self):
        assert normalize_lyric_line("digging a hole...") == normalize_lyric_line("digging a hole")


class TestDetectMedley:
    def test_no_repeat(self):
        assert detect_medley(["a b", "c d", "e f", "g h"]) is None

    def test_too_short(self):
        assert detect_medley(["a b", "a b"]) is None

    def test_single_line_repeats_ignored(self):
        assert detect_medley(["hook", "one", "two", "hook", "three", "four"]) is None

    def test_two_line_chorus(self):
        lines = [
            "verse one a", "verse one b",
            "la la chorus", "ho ho chorus",
            "verse two a", "verse two b",
            "la la chorus", "ho ho chorus",
            "verse three a", "verse three b",
        ]
        assert detect_medley(lines) == (2, 3)

    def test_first_occurrence_wins(self):
        lines = [
            "chorus a", "chorus b",
            "verse a", "verse b",
            "chorus a", "chorus b",
        ]
        assert detect_medley(lines) == (0, 1)

    def test_case_and_punctuation_insensitive(self):
        lines = [
            "verse a", "verse b",
            "La La, chorus!", "Ho Ho; chorus?",
            "verse c", "verse d",
            "la la  chorus", "ho ho chorus",
        ]
        assert detect_medley(lines) == (2, 3)

    def test_most_frequent_block_wins_over_longer_rare_block(self):
        # 2-line hook repeated 3 times beats a 4-line block repeated twice.
        lines = [
            "hook one", "hook two",
            "verse a",
            "chorus a", "chorus b", "chorus c", "chorus d",
            "bridge a",
            "hook one", "hook two",
            "verse b",
            "chorus a", "chorus b", "chorus c", "chorus d",
            "bridge b",
            "hook one", "hook two",
        ]
        assert detect_medley(lines) == (0, 1)

    def test_longer_block_wins_on_frequency_tie(self):
        lines = [
            "verse a", "verse b",
            "chorus a", "chorus b",
            "verse c", "verse d",
            "chorus a", "chorus b",
            "chorus a", "chorus b",
        ]
        assert detect_medley(lines) == (2, 3)

    def test_test_song(self):
        lines = [
            line.strip()
            for line in (REPO_ROOT / "test_song_lyrics_only.txt").read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        start, end = detect_medley(lines)
        assert start == 13
        assert end == 16
        assert lines[start] == "I am a dwarf and I'm digging a hole"
        assert lines[end] == "Diggy, diggy hole, digging a hole"
