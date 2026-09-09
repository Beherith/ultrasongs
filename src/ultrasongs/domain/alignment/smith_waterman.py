"""Smith-Waterman alignment with the legacy affine-gap semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .normalization import phonetic_score

MATCH_SCORE = 4.0
GAP_OPEN = 4.0
GAP_EXTEND = 0.5

MatrixName = Literal["M", "X", "Y"]


@dataclass(frozen=True, slots=True)
class BacktrackStep:
    i: int
    j: int
    matrix: MatrixName
    score: float


@dataclass(frozen=True, slots=True)
class SmithWatermanResult:
    max_score: float
    max_i: int
    max_j: int
    backtrack: tuple[BacktrackStep, ...]


def smith_waterman(
    lyric_characters: Sequence[str],
    transcription_characters: Sequence[str],
    *,
    match_score: float = MATCH_SCORE,
    gap_open_penalty: float = GAP_OPEN,
    gap_extend_penalty: float = GAP_EXTEND,
) -> SmithWatermanResult:
    """Align normalized characters, preserving the current TypeScript tie rules.

    The trace matrices are populated for semantic parity, although the legacy
    backtracker chooses the largest matrix at each preceding cell instead of
    following the stored state transition.
    """
    lyric_length = len(lyric_characters)
    transcription_length = len(transcription_characters)
    shape = (lyric_length + 1, transcription_length + 1)

    match = [[0.0] * shape[1] for _ in range(shape[0])]
    lyric_gap = [[0.0] * shape[1] for _ in range(shape[0])]
    transcription_gap = [[0.0] * shape[1] for _ in range(shape[0])]

    max_score = 0.0
    max_i = 0
    max_j = 0

    for i in range(1, lyric_length + 1):
        for j in range(1, transcription_length + 1):
            score = (
                phonetic_score(lyric_characters[i - 1], transcription_characters[j - 1])
                * match_score
            )
            match[i][j] = max(
                0.0,
                score + match[i - 1][j - 1],
                score
                - gap_open_penalty
                + max(lyric_gap[i - 1][j - 1], transcription_gap[i - 1][j - 1]),
            )
            lyric_gap[i][j] = max(
                -gap_open_penalty + match[i - 1][j],
                -gap_extend_penalty + lyric_gap[i - 1][j],
            )
            transcription_gap[i][j] = max(
                -gap_open_penalty + match[i][j - 1],
                -gap_extend_penalty + transcription_gap[i][j - 1],
            )

            best = max(match[i][j], lyric_gap[i][j], transcription_gap[i][j])
            if best > max_score:
                max_score = best
                max_i = i
                max_j = j

    i, j = max_i, max_j
    matrix, current_score = _best_matrix(match, lyric_gap, transcription_gap, i, j)
    reverse_steps: list[BacktrackStep] = []

    while current_score > 0 and (i > 0 or j > 0):
        reverse_steps.append(BacktrackStep(i=i, j=j, matrix=matrix, score=current_score))
        if matrix == "M":
            i -= 1
            j -= 1
        elif matrix == "X":
            i -= 1
        else:
            j -= 1

        i = max(i, 0)
        j = max(j, 0)
        matrix, current_score = _best_matrix(match, lyric_gap, transcription_gap, i, j)

    reverse_steps.reverse()
    return SmithWatermanResult(
        max_score=max_score,
        max_i=max_i,
        max_j=max_j,
        backtrack=tuple(reverse_steps),
    )


def _best_matrix(
    match: list[list[float]],
    lyric_gap: list[list[float]],
    transcription_gap: list[list[float]],
    i: int,
    j: int,
) -> tuple[MatrixName, float]:
    if match[i][j] >= lyric_gap[i][j] and match[i][j] >= transcription_gap[i][j]:
        return "M", match[i][j]
    if lyric_gap[i][j] >= transcription_gap[i][j]:
        return "X", lyric_gap[i][j]
    return "Y", transcription_gap[i][j]
