"""Tests for multi-run Whisper transcription consolidation."""

from cli.consensus import consolidate_whisper_runs, word_similarity


def _words(pairs: list[tuple[str, float, float]]) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in pairs]


class TestWordSimilarity:
    def test_exact_match(self):
        assert word_similarity("hello", "hello") == 1.0

    def test_case_and_diacritic_insensitive(self):
        assert word_similarity("Hello", "hello") == 1.0
        assert word_similarity("über", "uber") == 1.0

    def test_empty_is_mismatch(self):
        assert word_similarity("", "hi") < 0
        assert word_similarity("hi", "") < 0

    def test_dissimilar_words_are_negative(self):
        assert word_similarity("who", "zzq") < 0

    def test_near_homophones_score_high(self):
        assert word_similarity("hello", "hallo") > 0.5


class TestConsolidate:
    def test_empty_input(self):
        assert consolidate_whisper_runs([]) == []
        assert consolidate_whisper_runs([[], []]) == []

    def test_single_run_returned_unchanged(self):
        words = _words([("hello", 0.1, 0.5), ("world", 0.6, 1.0)])
        result = consolidate_whisper_runs([words])
        assert [w["word"] for w in result] == ["hello", "world"]
        assert result[0]["start"] == 0.1
        assert result[0]["end"] == 0.5

    def test_majority_vote(self):
        runs = [
            _words([("hello", 0.10, 0.50), ("world", 0.60, 1.00)]),
            _words([("hallo", 0.12, 0.51), ("world", 0.61, 1.00)]),
            _words([("hello", 0.09, 0.49), ("world", 0.60, 1.01)]),
        ]
        result = consolidate_whisper_runs(runs)
        assert [w["word"] for w in result] == ["hello", "world"]

    def test_timing_is_averaged_across_runs(self):
        runs = [
            _words([("hello", 0.0, 1.0)]),
            _words([("hello", 0.2, 1.2)]),
        ]
        result = consolidate_whisper_runs(runs)
        assert result[0]["start"] == 0.1
        assert result[0]["end"] == 1.1

    def test_uses_longest_run_as_reference(self):
        runs = [
            _words([("one", 0.0, 0.4), ("three", 0.8, 1.2)]),
            _words([("one", 0.0, 0.4), ("two", 0.45, 0.75), ("three", 0.8, 1.2)]),
            _words([("one", 0.0, 0.4), ("two", 0.46, 0.74), ("three", 0.8, 1.2)]),
        ]
        result = consolidate_whisper_runs(runs)
        assert [w["word"] for w in result] == ["one", "two", "three"]

    def test_reference_word_wins_ties(self):
        runs = [
            _words([("sunny", 0.0, 0.5)]),
            _words([("money", 0.0, 0.5)]),
        ]
        result = consolidate_whisper_runs(runs)
        assert result[0]["word"] == "sunny"

    def test_preserves_reference_boundaries(self):
        runs = [
            _words([("hello", 0.1, 0.5), ("world", 0.6, 1.0)]),
            _words([("hellowo", 0.1, 1.0)]),
        ]
        result = consolidate_whisper_runs(runs)
        assert len(result) == 2
