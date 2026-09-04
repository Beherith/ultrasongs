"""Tests for multi-run WhisperX forced-alignment consolidation."""

from cli.consensus import (
    consolidate_transcription_runs,
    consolidate_timing_runs,
    consolidate_whisperx_runs,
    word_similarity,
)


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
    def test_generic_entry_point(self):
        result = consolidate_transcription_runs([_words([("hello", 0.1, 0.5)])])
        assert result[0]["word"] == "hello"

    def test_empty_input(self):
        assert consolidate_whisperx_runs([]) == []
        assert consolidate_whisperx_runs([[], []]) == []

    def test_single_run_returned_unchanged(self):
        words = _words([("hello", 0.1, 0.5), ("world", 0.6, 1.0)])
        result = consolidate_whisperx_runs([words])
        assert [w["word"] for w in result] == ["hello", "world"]
        assert result[0]["start"] == 0.1
        assert result[0]["end"] == 0.5

    def test_majority_vote(self):
        runs = [
            _words([("hello", 0.10, 0.50), ("world", 0.60, 1.00)]),
            _words([("hallo", 0.12, 0.51), ("world", 0.61, 1.00)]),
            _words([("hello", 0.09, 0.49), ("world", 0.60, 1.01)]),
        ]
        result = consolidate_whisperx_runs(runs)
        assert [w["word"] for w in result] == ["hello", "world"]

    def test_timing_comes_from_one_coherent_candidate(self):
        runs = [
            _words([("hello", 0.0, 1.0)]),
            _words([("hello", 0.2, 1.2)]),
        ]
        result = consolidate_whisperx_runs(runs)
        assert (result[0]["start"], result[0]["end"]) in {(0.0, 1.0), (0.2, 1.2)}

    def test_prefers_candidate_with_stronger_character_alignment(self):
        runs = [
            [{
                "word": "hello", "start": 0.0, "end": 1.0,
                "characters": [{"char": "h", "start": 0.0, "end": 0.1, "score": 0.2}],
            }],
            [{
                "word": "hello", "start": 0.1, "end": 0.9,
                "characters": [{"char": "h", "start": 0.1, "end": 0.2, "score": 0.95}],
            }],
        ]
        result = consolidate_whisperx_runs(runs)
        assert result[0]["start"] == 0.1
        assert result[0]["characters"][0]["score"] == 0.95

    def test_uses_longest_run_as_reference(self):
        runs = [
            _words([("one", 0.0, 0.4), ("three", 0.8, 1.2)]),
            _words([("one", 0.0, 0.4), ("two", 0.45, 0.75), ("three", 0.8, 1.2)]),
            _words([("one", 0.0, 0.4), ("two", 0.46, 0.74), ("three", 0.8, 1.2)]),
        ]
        result = consolidate_whisperx_runs(runs)
        assert [w["word"] for w in result] == ["one", "two", "three"]

    def test_reference_word_wins_ties(self):
        runs = [
            _words([("sunny", 0.0, 0.5)]),
            _words([("money", 0.0, 0.5)]),
        ]
        result = consolidate_whisperx_runs(runs)
        assert result[0]["word"] == "sunny"

    def test_preserves_reference_boundaries(self):
        runs = [
            _words([("hello", 0.1, 0.5), ("world", 0.6, 1.0)]),
            _words([("hellowo", 0.1, 1.0)]),
        ]
        result = consolidate_whisperx_runs(runs)
        assert len(result) == 2


class TestTimingConsolidation:
    def test_selects_median_timing_candidate_from_odd_runs(self):
        runs = [
            [{
                "word": "hello", "start": 0.0, "end": 0.4,
                "characters": [{"char": "h", "start": 0.0, "end": 0.1, "score": 0.9}],
            }],
            [{
                "word": "hello", "start": 0.2, "end": 0.6,
                "characters": [{"char": "h", "start": 0.2, "end": 0.3, "score": 0.8}],
            }],
            [{
                "word": "hello", "start": 4.0, "end": 4.5,
                "characters": [{"char": "h", "start": 4.0, "end": 4.1, "score": 1.0}],
            }],
        ]

        result = consolidate_timing_runs(runs)

        assert result[0]["start"] == 0.2
        assert result[0]["characters"][0]["start"] == 0.2

    def test_timing_consolidation_handles_empty_and_single_runs(self):
        assert consolidate_timing_runs([]) == []
        word = {"word": "one", "start": 0.1, "end": 0.4}
        assert consolidate_timing_runs([[word]]) == [word]
