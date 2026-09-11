"""Medley (chorus) detection in plain lyrics.

A medley is the largest, most frequently repeated section of the lyrics: a
block of at least two consecutive lines that occurs at least twice. The
medley section is the first occurrence of that block.
"""

import re

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_lyric_line(line: str) -> str:
    """Normalize a lyric line for repeat comparison.

    Lowercases, strips punctuation, and collapses whitespace so lines that
    differ only in casing, punctuation, or spacing compare equal.
    """
    text = _PUNCT_RE.sub(" ", line.lower())
    return _WS_RE.sub(" ", text).strip()


def detect_medley(lines: list[str]) -> tuple[int, int] | None:
    """Find the medley (chorus) section in a list of lyric lines.

    Every block of at least two consecutive lines is compared against all
    other positions (after normalization). The medley block is the one that
    occurs most often; ties go to the longer block, then the earliest one.

    Args:
        lines: Non-empty lyric lines.

    Returns:
        (start_line, end_line) 0-based, inclusive, of the first occurrence of
        the medley block, or None when no block repeats.
    """
    normalized = [normalize_lyric_line(line) for line in lines]
    n = len(normalized)
    if n < 4:
        return None

    line_ids: dict[str, int] = {}
    ids: list[int] = []
    positions: list[list[int]] = []
    for index, text in enumerate(normalized):
        line_id = line_ids.get(text)
        if line_id is None:
            line_id = len(line_ids)
            line_ids[text] = line_id
            positions.append([])
        ids.append(line_id)
        positions[line_id].append(index)

    def count_occurrences(first_id: int, block: list[int]) -> tuple[int, int]:
        count = 0
        first = -1
        block_len = len(block)
        for start in positions[first_id]:
            if start + block_len > n or ids[start:start + block_len] != block:
                continue
            count += 1
            if first < 0:
                first = start
        return count, first

    best: tuple[int, int, int, int, int] | None = None  # (count, length, -first, start, end)
    for i in range(n):
        first_id = ids[i]
        if len(positions[first_id]) < 2:
            continue
        block: list[int] = [first_id]
        for length in range(2, n - i + 1):
            block.append(ids[i + length - 1])
            count, first = count_occurrences(first_id, block)
            if count < 2:
                break
            candidate = (count, length, -first, first, first + length - 1)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        return None
    return best[3], best[4]
