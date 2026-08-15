"""Consolidate multiple Whisper transcription runs into one consensus.

Whisper (and the Demucs vocal separation it transcribes) is not fully
deterministic, so independent runs of the same audio can recognize slightly
different words. Running the transcription several times and voting on the
result — with Smith-Waterman used to align the word sequences before voting —
produces a more robust consensus than any single run.

This module is intentionally free of heavy (torch/whisper) imports so it can
be unit-tested without GPU dependencies.
"""

from typing import Any

from cli.align import normalize_char, phonetic_score, smith_waterman
from cli.logging_setup import get_logger

logger = get_logger("cli.consensus")

_WORD_MISMATCH_SCORE = -0.3


def word_similarity(a: str, b: str) -> float:
    """Score how similar two transcribed words are, roughly in [-0.3, 1.0].

    Equal (after normalization) words score 1.0. Otherwise the score is the
    mean of a symmetric character-level "best phonetic match" between the two
    words, so near-homophones score high and unrelated words score negative.
    """
    na = normalize_char(a)
    nb = normalize_char(b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return _WORD_MISMATCH_SCORE

    pool_a = set(na)
    pool_b = set(nb)
    score_a = sum(max(phonetic_score(c, p) for p in pool_b) for c in na) / len(na)
    score_b = sum(max(phonetic_score(c, p) for p in pool_a) for c in nb) / len(nb)
    return (score_a + score_b) / 2


def _align_to_reference(
    reference_norm: list[str],
    run_words: list[str],
) -> dict[int, int]:
    """Map each reference index to the run index aligned to it (or -1 for a gap)."""
    alignment: dict[int, int] = {}
    sw = smith_waterman(
        reference_norm,
        [normalize_char(w) for w in run_words],
        score_fn=word_similarity,
    )
    for step in sw["backtrack"]:
        i, j, m = step["i"], step["j"], step["matrix"]
        if m == "M" and i > 0 and 0 <= j - 1 < len(run_words):
            alignment[i - 1] = j - 1
        elif m == "X" and i > 0:
            alignment[i - 1] = -1
    return alignment


def consolidate_whisper_runs(
    runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Consolidate several Whisper word lists into one consensus list.

    Each ``runs`` entry is a full transcription as a list of word dicts with at
    least ``word``, ``start`` and ``end`` keys. The longest run is used as the
    structural reference: its word boundaries define the consensus, and for
    every reference word the best (majority-vote) reading across all runs is
    chosen, with timing averaged over the aligned words.

    A single run is returned unchanged. Returns an empty list when there is no
    usable transcription.
    """
    non_empty = [i for i in range(len(runs)) if len(runs[i])]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return [dict(w) for w in runs[non_empty[0]]]

    reference_idx = max(non_empty, key=lambda i: len(runs[i]))
    reference = runs[reference_idx]
    reference_words = [w["word"] for w in reference]
    reference_norm = [normalize_char(w) for w in reference_words]

    logger.info(
        "Consolidating transcription runs: "
        + " | ".join(f"run {i + 1} = {len(runs[i])} words" for i in range(len(runs)))
    )
    logger.info(
        f"Reference: run {reference_idx + 1} "
        f"({len(reference_words)} words, longest run)"
    )

    # columns[i] holds the word dicts aligned to reference word i; the
    # reference word itself is always index 0 and therefore wins any tie.
    # origins[i][k] records which run column[i][k] came from.
    columns: list[list[dict[str, Any]]] = [[dict(w)] for w in reference]
    origins: list[list[int]] = [[reference_idx] for _ in reference]

    for run_idx, run in enumerate(runs):
        if run_idx == reference_idx or not run:
            continue
        run_word_list = [w["word"] for w in run]
        alignment = _align_to_reference(reference_norm, run_word_list)
        mapped_positions = 0
        used_j: set[int] = set()
        for ref_i in range(len(reference_words)):
            run_j = alignment.get(ref_i, -1)
            if run_j >= 0:
                columns[ref_i].append(dict(run[run_j]))
                origins[ref_i].append(run_idx)
                mapped_positions += 1
                used_j.add(run_j)
        gaps = len(reference_words) - mapped_positions
        unmatched = len(run) - len(used_j)
        logger.info(
            f"Aligned run {run_idx + 1} ({len(run)} words) to reference: "
            f"{mapped_positions}/{len(reference_words)} positions matched, "
            f"{gaps} reference gaps, {unmatched} extra words"
        )

    consensus: list[dict[str, Any]] = []
    disagreements = 0
    for ref_i, column in enumerate(columns):
        words = [w["word"] for w in column]
        starts = [float(w["start"]) for w in column]
        ends = [float(w["end"]) for w in column]

        counts: dict[str, int] = {}
        first_index: dict[str, int] = {}
        for pos, w in enumerate(words):
            norm = normalize_char(w)
            counts[norm] = counts.get(norm, 0) + 1
            first_index.setdefault(norm, pos)
        winner_norm = min(counts, key=lambda n: (-counts[n], first_index[n]))
        winner_word = words[first_index[winner_norm]]

        consensus.append({
            "word": winner_word,
            "start": sum(starts) / len(starts),
            "end": sum(ends) / len(ends),
        })

        if len(counts) > 1:
            disagreements += 1
            readouts = "  ".join(
                f"run{origins[ref_i][pos] + 1}='{words[pos]}'"
                f"({starts[pos]:.2f}-{ends[pos]:.2f}s)"
                for pos in range(len(words))
            )
            votes = ", ".join(
                f"{c}x'{n}'"
                for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            logger.info(
                f"Disagree @ reference pos {ref_i} (ref='{reference_words[ref_i]}'): "
                f"{readouts}  ->  '{winner_word}'  [{votes}]"
            )

    logger.info(
        f"Consolidated {len(runs)} whisper runs "
        f"(reference: run {reference_idx + 1}, {len(reference_words)} words; "
        f"{disagreements} words disagreed)"
    )

    return consensus
