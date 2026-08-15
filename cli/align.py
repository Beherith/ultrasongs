"""Lyric alignment: Smith-Waterman with phonetic scoring.

Ported from app/lib/align.ts (704 lines).
"""

import bisect
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cli.config import Config
from cli.logging_setup import get_logger
from cli.syllabify import split_word
from cli.pipeline_types import AlignedSyllable, Pause, PitchFrame, WordTimestamp

logger = get_logger("cli.align")

# ── Constants ────────────────────────────────────────────────────────────────

MATCH_SCORE = 4
GAP_OPEN = 4
GAP_EXTEND = 0.5


# ── Character normalization ──────────────────────────────────────────────────

def normalize_char(c: str) -> str:
    """Lowercase, NFD decompose, strip diacritics."""
    return unicodedata.normalize("NFD", c.lower()).encode("ascii", "ignore").decode("ascii")


# ── Phonetic character matching ──────────────────────────────────────────────

_PHONETIC_GROUPS = [
    ["a", "e", "i"],
    ["o", "u"],
    ["s", "z", "c"],
    ["t", "d"],
    ["p", "b"],
    ["k", "g"],
    ["f", "v", "w"],
    ["m", "n"],
    ["l", "r"],
    ["y", "i"],
    ["h", "j"],
    ["b", "v"],
]

_CROSS_PAIRS = [
    ("a", "o"), ("a", "u"), ("e", "i"), ("e", "o"), ("i", "y"),
    ("s", "sh"), ("z", "zh"), ("f", "ph"), ("c", "k"), ("q", "k"),
    ("w", "u"), ("r", "l"), ("b", "p"), ("d", "t"), ("g", "k"),
]


def phonetic_score(x: str, y: str) -> float:
    """Score how phonetically similar two characters are."""
    if x == y:
        return 1.0

    for g in _PHONETIC_GROUPS:
        ix = g.index(x) if x in g else -1
        iy = g.index(y) if y in g else -1
        if ix >= 0 and iy >= 0:
            return 0.6 - 0.1 * abs(ix - iy)

    for a, b in _CROSS_PAIRS:
        if (x == a and y == b) or (x == b and y == a):
            return 0.5

    return -0.3


# ── Smith-Waterman with affine gap penalties ─────────────────────────────────

def smith_waterman(lyric_chars: list[str], whisper_chars: list[str]) -> dict[str, Any]:
    """Local sequence alignment with affine gaps.

    Returns dict with maxScore, maxI, maxJ, and backtrack path.
    """
    L = len(lyric_chars)
    W = len(whisper_chars)

    # Three matrices: M (match), X (gap in whisper), Y (gap in lyric)
    M = [[0.0] * (W + 1) for _ in range(L + 1)]
    X = [[0.0] * (W + 1) for _ in range(L + 1)]
    Y = [[0.0] * (W + 1) for _ in range(L + 1)]

    # Trace matrices for backtracking
    traceM = [[0] * (W + 1) for _ in range(L + 1)]
    traceX = [[0] * (W + 1) for _ in range(L + 1)]
    traceY = [[0] * (W + 1) for _ in range(L + 1)]

    max_score = 0.0
    max_i = 0
    max_j = 0

    for i in range(1, L + 1):
        for j in range(1, W + 1):
            s = phonetic_score(lyric_chars[i - 1], whisper_chars[j - 1]) * MATCH_SCORE

            M[i][j] = max(0, s + M[i - 1][j - 1], s - GAP_OPEN + max(X[i - 1][j - 1], Y[i - 1][j - 1]))
            if M[i][j] == 0:
                traceM[i][j] = 0
            elif M[i][j] == s + M[i - 1][j - 1]:
                traceM[i][j] = 1
            else:
                traceM[i][j] = 2

            X[i][j] = max(-GAP_OPEN + M[i - 1][j], -GAP_EXTEND + X[i - 1][j])
            traceX[i][j] = 3 if X[i][j] == -GAP_OPEN + M[i - 1][j] else 4

            Y[i][j] = max(-GAP_OPEN + M[i][j - 1], -GAP_EXTEND + Y[i][j - 1])
            traceY[i][j] = 5 if Y[i][j] == -GAP_OPEN + M[i][j - 1] else 6

            best = max(M[i][j], X[i][j], Y[i][j])
            if best > max_score:
                max_score = best
                max_i = i
                max_j = j

    # Backtrack
    backtrack: list[dict[str, Any]] = []
    ci, cj = max_i, max_j

    # Determine starting matrix
    if M[ci][cj] >= X[ci][cj] and M[ci][cj] >= Y[ci][cj]:
        cmat = "M"
        cscore = M[ci][cj]
    elif X[ci][cj] >= Y[ci][cj]:
        cmat = "X"
        cscore = X[ci][cj]
    else:
        cmat = "Y"
        cscore = Y[ci][cj]

    while cscore > 0 and (ci > 0 or cj > 0):
        backtrack.append({
            "i": ci, "j": cj,
            "matrix": cmat,
            "score": cscore,
        })

        if cmat == "M":
            ci -= 1
            cj -= 1
        elif cmat == "X":
            ci -= 1
        else:  # Y
            cj -= 1

        ci = max(0, ci)
        cj = max(0, cj)

        # Determine next matrix
        if M[ci][cj] >= X[ci][cj] and M[ci][cj] >= Y[ci][cj]:
            cmat = "M"
            cscore = M[ci][cj]
        elif X[ci][cj] >= Y[ci][cj]:
            cmat = "X"
            cscore = X[ci][cj]
        else:
            cmat = "Y"
            cscore = Y[ci][cj]

        if cscore <= 0:
            break

    backtrack.reverse()
    return {
        "maxScore": max_score,
        "maxI": max_i,
        "maxJ": max_j,
        "backtrack": backtrack,
    }


# ── Backtrace visualization ─────────────────────────────────────────────────

_LINE_WIDTH = 120  # characters per alignment line (excluding prefix)


def _format_backtrace(
    sw_result: dict[str, Any],
    lyric_chars: list[dict[str, Any]],
    whisper_chars: list[dict[str, Any]],
) -> str:
    """Render the SW backtrack as a 120-char-wide visual alignment.

    Format per block:
        <pos>
        q: <lyric chars, - for gaps>
        -- <alignment symbols: |=match, .=phonetic, space=mismatch>
        s: <whisper chars, - for gaps>
        <pos>
    """
    columns: list[tuple[str, str, str]] = []

    for step in sw_result["backtrack"]:
        if step["matrix"] == "M" and step["i"] > 0 and step["j"] > 0:
            lc = lyric_chars[step["i"] - 1]
            wc = whisper_chars[step["j"] - 1]
            ls = lc["orig"]
            ws = wc["orig"]
            if ls == ws:
                sym = "|"
            else:
                sym = "." if phonetic_score(lc["norm"], wc["norm"]) >= 0 else " "
            columns.append((ls, ws, sym))
        elif step["matrix"] == "X" and step["i"] > 0:
            lc = lyric_chars[step["i"] - 1]
            columns.append((lc["orig"], "-", " "))
        elif step["matrix"] == "Y" and step["j"] > 0:
            wc = whisper_chars[step["j"] - 1]
            columns.append(("-", wc["orig"], " "))

    if not columns:
        return "(empty backtrack)"

    # Chunk into 120-char-wide blocks
    blocks: list[tuple[str, str, str, int]] = []
    q_buf, s_buf, sym_buf = "", "", ""
    start_pos = 0

    for ql, sl, sym in columns:
        q_buf += ql
        s_buf += sl
        sym_buf += sym
        if len(q_buf) >= _LINE_WIDTH:
            blocks.append((q_buf, s_buf, sym_buf, start_pos))
            start_pos += len(q_buf)
            q_buf, s_buf, sym_buf = "", "", ""

    if q_buf:
        blocks.append((q_buf, s_buf, sym_buf, start_pos))

    lines = []
    for q_line, s_line, sym_line, pos in blocks:
        lines.append(str(pos))
        lines.append(f"q: {q_line}")
        lines.append(f"-- {sym_line}")
        lines.append(f"s: {s_line}")
    lines.append(str(start_pos + len(q_buf) if blocks else 0))

    return "\n".join(lines)


_UNALIGNED_END_MAX_CHARS = 200


def _clip(text: str) -> str:
    if len(text) > _UNALIGNED_END_MAX_CHARS:
        return text[:_UNALIGNED_END_MAX_CHARS] + "..."
    return text


def _unaligned_ends(
    sw_result: dict[str, Any],
    lyric_chars: list[dict[str, Any]],
    whisper_chars: list[dict[str, Any]],
) -> str:
    """Return the lyric and whisper text outside the SW alignment span."""
    lyric_used: set[int] = set()
    whisper_used: set[int] = set()
    for step in sw_result["backtrack"]:
        i, j, m = step["i"], step["j"], step["matrix"]
        if m in ("M", "X") and i > 0:
            lyric_used.add(i - 1)
        if m in ("M", "Y") and j > 0:
            whisper_used.add(j - 1)

    def ends(chars: list[dict[str, Any]], used: set[int]) -> tuple[str, str]:
        if not used:
            prefix, suffix = chars, []
        else:
            first = min(used)
            last = max(used)
            prefix = chars[:first] if first > 0 else []
            suffix = chars[last + 1:] if last + 1 < len(chars) else []
        return (
            _clip("".join(c["orig"] for c in prefix).strip()),
            _clip("".join(c["orig"] for c in suffix).strip()),
        )

    l_pre, l_post = ends(lyric_chars, lyric_used)
    w_pre, w_post = ends(whisper_chars, whisper_used)

    if not any((l_pre, l_post, w_pre, w_post)):
        return ""

    lines = ["Unaligned ends:"]
    if l_pre:
        lines.append(f'  lyric before: "{l_pre}"')
    if l_post:
        lines.append(f'  lyric after:  "{l_post}"')
    if w_pre:
        lines.append(f'  whisper before: "{w_pre}"')
    if w_post:
        lines.append(f'  whisper after:  "{w_post}"')
    return "\n".join(lines)


def _word_similarity(lyric_norm: str, whisper_norm: str) -> float:
    """Average best phonetic score per lyric char against the whisper word."""
    if not lyric_norm or not whisper_norm:
        return 0.0
    pool = set(whisper_norm)
    chars = [c for c in lyric_norm if c != " "]
    if not chars:
        return 0.0
    return sum(max(phonetic_score(c, p) for p in pool) for c in chars) / len(chars)


def _log_unmatched_word(wi: int, wr: dict[str, Any], whisper_words: list[WordTimestamp]) -> None:
    """Log debug info explaining why a lyric word was not matched by SW."""

    word = wr["word"]
    norm = normalize_char(word)
    scored = [
        (_word_similarity(norm, normalize_char(ww.word)), idx, ww.word, ww.start, ww.end)
        for idx, ww in enumerate(whisper_words)
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = ", ".join(
        f"'{w}' (idx {i}, {start:.2f}-{end:.2f}s, score {s:.2f})"
        for s, i, w, start, end in scored[:3]
    )
    logger.info(
        f"Lyric word {wi} '{word}' (line {wr['lineIdx']}) not matched by SW: "
        f"normalized='{norm}', length={len(word)}; "
        f"best whisper candidates: {top or '(no whisper words)'}"
    )


# ── MIDI extraction ──────────────────────────────────────────────────────────

def _median(values: list[int]) -> int | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)


def midi_for_range(word: WordTimestamp, start: float, end: float) -> tuple[int, int]:
    """Get median MIDI note from pitch frames within a time range.

    Returns (midi, frame_count).
    """
    frames = word.pitch_frames
    if not frames:
        return word.midi, 0

    for threshold in (0.5, 0.3, 0.1):
        values = [
            f.midi for f in frames
            if f.time >= start and f.time <= end and f.confidence > threshold and 0 < f.midi
        ]
        val = _median(values)
        if val is not None:
            return val, len(values)

    return word.midi, 0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _activity_threshold(frames: list[PitchFrame], config: Config) -> float:
    """Estimate a song-relative energy threshold from quiet and voiced frames."""
    amplitudes = [f.amplitude for f in frames]
    quiet = [f.amplitude for f in frames if f.confidence < config.activity_quiet_confidence]
    voiced = [f.amplitude for f in frames if f.confidence >= config.activity_voiced_confidence]
    noise = _percentile(quiet, config.activity_noise_percentile) \
        if quiet else _percentile(amplitudes, config.activity_noise_fallback_percentile)
    signal = _percentile(voiced, config.activity_signal_percentile) \
        if voiced else _percentile(amplitudes, config.activity_signal_fallback_percentile)
    return noise + max(0.0, signal - noise) * config.activity_threshold_ratio


def _note_segments(
    frames: list[PitchFrame],
    start: float,
    end: float,
    fallback_midi: int,
    amplitude_threshold: float,
    config: Config,
) -> list[tuple[float, float, int]]:
    """Trim a syllable to vocal activity and split sustained pitch changes."""
    dropout_gap = config.note_dropout_gap_ms / 1000
    min_duration = config.note_min_duration_ms / 1000
    window = [f for f in frames if start <= f.time <= end and f.midi > 0]
    active = [
        f for f in window
        if f.confidence >= config.note_min_confidence and f.amplitude >= amplitude_threshold
    ]
    if not active:
        active = [f for f in window if f.confidence >= config.note_fallback_confidence]
    if not active:
        return [(start, end, fallback_midi)]

    # Keep the strongest contiguous vocal island. Short confidence/energy
    # dropouts up to note_dropout_gap_ms are treated as part of the same sung sound.
    islands: list[list[PitchFrame]] = []
    for frame in active:
        if not islands or frame.time - islands[-1][-1].time > dropout_gap:
            islands.append([frame])
        else:
            islands[-1].append(frame)
    active = max(
        islands,
        key=lambda island: sum(f.amplitude * f.confidence for f in island),
    )

    # A median window removes vibrato/jitter before finding pitch changes.
    half = config.note_smooth_window // 2
    smoothed: list[tuple[PitchFrame, int]] = []
    for i, frame in enumerate(active):
        nearby = active[max(0, i - half):i + half + 1]
        midi = _median([f.midi for f in nearby]) or fallback_midi
        smoothed.append((frame, midi))

    runs: list[list[tuple[PitchFrame, int]]] = []
    for item in smoothed:
        if not runs:
            runs.append([item])
            continue
        run_midi = _median([midi for _, midi in runs[-1]]) or fallback_midi
        gap = item[0].time - runs[-1][-1][0].time
        if gap <= dropout_gap and abs(item[1] - run_midi) <= config.note_pitch_tolerance:
            runs[-1].append(item)
        else:
            runs.append([item])

    # Very short pitch changes are detection noise, not singable notes.
    merged: list[list[tuple[PitchFrame, int]]] = []
    for run in runs:
        duration = run[-1][0].time - run[0][0].time
        if duration < min_duration and merged:
            merged[-1].extend(run)
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0][-1][0].time - merged[0][0][0].time < min_duration:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)

    frame_step = config.note_frame_step_ms / 1000
    if len(active) > 1:
        frame_step = max(0.001, active[1].time - active[0].time)

    result: list[tuple[float, float, int]] = []
    for run in merged:
        weights = [max(0.001, frame.amplitude * frame.confidence) for frame, _ in run]
        midi = round(sum(value * weight for (_, value), weight in zip(run, weights)) / sum(weights))
        run_start = max(start, run[0][0].time)
        run_end = min(end, run[-1][0].time + frame_step)
        if run_end > run_start:
            result.append((run_start, run_end, midi))

    return result or [(start, end, fallback_midi)]


# ── Main alignment ───────────────────────────────────────────────────────────

def align_lyrics(
    lyrics: str,
    whisper_words: list[WordTimestamp],
    language: str,
    pauses: list[Pause] | None = None,
    config: Config | None = None,
    pitch_frames: list[PitchFrame] | None = None,
) -> list[AlignedSyllable]:
    """Align lyric text to Whisper word timestamps using Smith-Waterman.

    Steps:
        1. Character normalization
        2. Phonetic scoring via Smith-Waterman
        3. Word-level match extraction
        4. Timestamp computation for matched words
        5. Linear interpolation for unmatched words
        6. Syllabification
        7. Line break insertion

    Args:
        lyrics: Raw lyric text with line breaks.
        whisper_words: Word timestamps from transcription.
        language: ISO 639-1 language code.
        pauses: Detected silence regions.
        config: Pipeline configuration.
        pitch_frames: Complete CREPE pitch, confidence, and amplitude timeline.

    Returns:
        List of aligned syllables with timestamps and MIDI notes.
    """
    if pauses is None:
        pauses = []
    if pitch_frames is None:
        pitch_frames = []
    if config is None:
        config = Config()

    raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]

    logger.info(f"Alignment started")
    logger.info(f"Language: {language}")
    logger.info(f"Lyric lines: {len(raw_lines)}")
    logger.info(f"Whisper words: {len(whisper_words)}")

    # ── Build lyric character sequence ────────────────────────────────────

    lyric_words: list[dict[str, Any]] = []
    for li, line in enumerate(raw_lines):
        for w in line.split():
            lyric_words.append({"word": w, "lineIdx": li})

    lyric_chars: list[dict[str, Any]] = []
    for wi, lw in enumerate(lyric_words):
        for ch in lw["word"]:
            lyric_chars.append({
                "orig": ch,
                "norm": normalize_char(ch),
                "wordIdx": wi,
                "lineIdx": lw["lineIdx"],
            })
        if wi < len(lyric_words) - 1:
            lyric_chars.append({
                "orig": " ",
                "norm": " ",
                "wordIdx": wi,
                "lineIdx": lw["lineIdx"],
            })

    # ── Build whisper character sequence ──────────────────────────────────

    whisper_chars: list[dict[str, Any]] = []
    for wi, ww in enumerate(whisper_words):
        cleaned = normalize_char(ww.word)
        for ch in cleaned:
            whisper_chars.append({
                "orig": ch,
                "norm": normalize_char(ch),
                "wordIdx": wi,
            })
        if wi < len(whisper_words) - 1:
            whisper_chars.append({
                "orig": " ",
                "norm": " ",
                "wordIdx": wi,
            })

    logger.info(f"Lyric chars: {len(lyric_chars)} ({len(lyric_words)} words)")
    logger.info(f"Whisper chars: {len(whisper_chars)} ({len(whisper_words)} words)")

    # ── Smith-Waterman alignment ──────────────────────────────────────────

    logger.info(f"Running Smith-Waterman: {len(lyric_chars)} x {len(whisper_chars)}")
    sw_result = smith_waterman(
        [c["norm"] for c in lyric_chars],
        [c["norm"] for c in whisper_chars],
    )
    logger.info(
        f"SW complete: maxScore={sw_result['maxScore']:.2f} "
        f"at ({sw_result['maxI']}, {sw_result['maxJ']}), "
        f"backtrack={len(sw_result['backtrack'])} steps"
    )

    sw_backtrace_text = _format_backtrace(sw_result, lyric_chars, whisper_chars)
    unaligned_ends = _unaligned_ends(sw_result, lyric_chars, whisper_chars)
    if unaligned_ends:
        sw_backtrace_text += "\n\n" + unaligned_ends

    # ── Extract word-level matches from backtrack ─────────────────────────

    lyric_word_matched = [False] * len(lyric_words)
    lyric_word_whisper_idxs: list[list[int]] = [[] for _ in range(len(lyric_words))]
    lyric_word_char_alignments: list[list[dict[str, Any]]] = [[] for _ in range(len(lyric_words))]

    for step in sw_result["backtrack"]:
        if step["matrix"] == "M" and step["i"] > 0 and step["j"] > 0:
            lc = lyric_chars[step["i"] - 1]
            wc = whisper_chars[step["j"] - 1]
            if lc and wc and lc["norm"] != " " and wc["norm"] != " ":
                lyric_word_matched[lc["wordIdx"]] = True
                if wc["wordIdx"] not in lyric_word_whisper_idxs[lc["wordIdx"]]:
                    lyric_word_whisper_idxs[lc["wordIdx"]].append(wc["wordIdx"])
                lyric_word_char_alignments[lc["wordIdx"]].append({
                    "lyricChar": lc["orig"],
                    "whisperChar": wc["orig"],
                    "whisperWordIdx": wc["wordIdx"],
                    "score": step["score"],
                })

    # ── Compute timestamps for matched lyric words ────────────────────────

    word_results: list[dict[str, Any]] = []
    for wi, lw in enumerate(lyric_words):
        word_results.append({
            "word": lw["word"],
            "lineIdx": lw["lineIdx"],
            "start": 0.0,
            "end": 0.0,
            "midi": 60,
            "source": "sw_aligned",
            "whisperIdxs": lyric_word_whisper_idxs[wi],
            "charAlignments": lyric_word_char_alignments[wi],
            "pitchFrames": [],
        })

    for wi in range(len(word_results)):
        if not lyric_word_matched[wi]:
            _log_unmatched_word(wi, word_results[wi], whisper_words)
            continue
        idxs = sorted(word_results[wi]["whisperIdxs"])
        starts = [whisper_words[idx].start for idx in idxs]
        ends = [whisper_words[idx].end for idx in idxs]
        word_results[wi]["start"] = min(starts)
        word_results[wi]["end"] = max(ends)

        all_frames: list[Any] = []
        for idx in idxs:
            all_frames.extend(whisper_words[idx].pitch_frames)

        if all_frames:
            word_results[wi]["pitchFrames"] = all_frames
            mr, _ = midi_for_range(
                WordTimestamp(word="", start=word_results[wi]["start"], end=word_results[wi]["end"], midi=60, pitch_frames=all_frames),
                word_results[wi]["start"],
                word_results[wi]["end"],
            )
            word_results[wi]["midi"] = mr
        else:
            word_results[wi]["midi"] = whisper_words[idxs[0]].midi if idxs else 60

    # Whisper can merge several lyric words into one token. Divide that token's
    # interval instead of assigning the full interval to every lyric word.
    wi = 0
    while wi < len(word_results):
        idxs = tuple(sorted(word_results[wi]["whisperIdxs"]))
        end_wi = wi + 1
        while (
            idxs
            and end_wi < len(word_results)
            and tuple(sorted(word_results[end_wi]["whisperIdxs"])) == idxs
        ):
            end_wi += 1
        if end_wi - wi > 1:
            group = word_results[wi:end_wi]
            group_start = min(wr["start"] for wr in group)
            group_end = max(wr["end"] for wr in group)
            weights = [max(1, len(normalize_char(wr["word"]))) for wr in group]
            total_weight = sum(weights)
            cursor = group_start
            for wr, weight in zip(group, weights):
                next_cursor = cursor + (group_end - group_start) * weight / total_weight
                wr["start"] = cursor
                wr["end"] = next_cursor
                cursor = next_cursor
        wi = end_wi

    # ── Interpolate unmatched words ───────────────────────────────────────

    matched_indices = [wi for wi in range(len(word_results)) if lyric_word_matched[wi]]

    if not matched_indices:
        logger.warning("No matched words; distributing lyrics over the transcript")
        timeline_start = whisper_words[0].start if whisper_words else 0.0
        timeline_end = whisper_words[-1].end if whisper_words else timeline_start + 0.3 * len(word_results)
        slot = max(0.01, (timeline_end - timeline_start) / max(1, len(word_results)))
        for wi, wr in enumerate(word_results):
            wr["source"] = "interpolated_before"
            wr["start"] = timeline_start + wi * slot
            wr["end"] = timeline_start + (wi + 1) * slot
            if whisper_words:
                wr["midi"] = whisper_words[min(wi, len(whisper_words) - 1)].midi
    else:
        first = matched_indices[0]
        last = matched_indices[-1]

        # Before first matched word
        for wi in range(first):
            word_results[wi]["source"] = "interpolated_before"
        if first > 0:
            anchor_start = word_results[first]["start"]
            slot = max(0.1, anchor_start / first)
            for wi in range(first):
                word_results[wi]["start"] = max(0, anchor_start - (first - wi) * slot)
                word_results[wi]["end"] = max(0, anchor_start - (first - wi - 1) * slot)
                word_results[wi]["midi"] = word_results[first]["midi"]

        # Between matched words
        for ai in range(len(matched_indices) - 1):
            a = matched_indices[ai]
            b = matched_indices[ai + 1]
            if b - a <= 1:
                continue
            missing = b - a - 1

            t_start = word_results[a]["end"]
            t_end = word_results[b]["start"]
            a_whisper = word_results[a]["whisperIdxs"]
            b_whisper = word_results[b]["whisperIdxs"]
            if a_whisper and b_whisper:
                unused = list(range(max(a_whisper) + 1, min(b_whisper)))
                if unused:
                    t_start = whisper_words[unused[0]].start
                    t_end = whisper_words[unused[-1]].end
            duration = min(max(0, t_end - t_start), missing * 0.8)
            midi_a = word_results[a]["midi"]
            midi_b = word_results[b]["midi"]

            for k in range(1, missing + 1):
                wi = a + k
                word_results[wi]["source"] = "interpolated_between"
                word_results[wi]["start"] = t_start + (k - 1) * duration / missing
                word_results[wi]["end"] = t_start + k * duration / missing
                word_results[wi]["midi"] = round(midi_a + k / (missing + 1) * (midi_b - midi_a))

        # After last matched word
        for wi in range(last + 1, len(word_results)):
            word_results[wi]["source"] = "interpolated_after"
        if last < len(word_results) - 1:
            avg_dur = sum(
                word_results[idx]["end"] - word_results[idx]["start"]
                for idx in matched_indices
            ) / len(matched_indices)
            fallback = max(0.2, avg_dur)
            for wi in range(last + 1, len(word_results)):
                offset = (wi - last - 1) * fallback
                word_results[wi]["start"] = word_results[last]["end"] + offset
                word_results[wi]["end"] = word_results[last]["end"] + offset + fallback
                word_results[wi]["midi"] = word_results[last]["midi"]

    # Silence between adjacent words is a hard timing boundary.
    for current, following in zip(word_results, word_results[1:]):
        between = next((
            pause for pause in pauses
            if current["start"] <= pause.start and pause.end <= following["end"]
        ), None)
        if between:
            current["end"] = min(current["end"], between.start)
            following["start"] = max(following["start"], between.end)

        if current["end"] > following["start"]:
            boundary = (current["end"] + following["start"]) / 2
            current["end"] = max(current["start"], boundary)
            following["start"] = min(following["end"], boundary)

    # Show the same visual character alignment written to align_backtrace.txt.
    logger.info("Smith-Waterman alignment:\n%s", sw_backtrace_text)

    # ── Syllabification + output ──────────────────────────────────────────

    final_output: list[AlignedSyllable] = []
    amplitude_threshold = _activity_threshold(pitch_frames, config) if pitch_frames else 0.0
    previous_line: int | None = None
    # pitch_frames is already sorted by time; precompute keys so per-word
    # frame lookup is a slice of a bisect range instead of a linear filter.
    pitch_times = [f.time for f in pitch_frames] if pitch_frames else []

    for wr in word_results:
        syllables = split_word(wr["word"], language)
        print(f"Word '{wr['word']}' split into syllables: {syllables}")
        syl_duration = max(0.01, (wr["end"] - wr["start"]) / len(syllables))
        if pitch_frames:
            lo = bisect.bisect_left(pitch_times, wr["start"])
            hi = bisect.bisect_right(pitch_times, wr["end"])
            word_frames = pitch_frames[lo:hi]
        else:
            word_frames = wr["pitchFrames"]
        word_output: list[AlignedSyllable] = []

        for si, syl in enumerate(syllables):
            syl_start = wr["start"] + si * syl_duration
            syl_end = wr["start"] + (si + 1) * syl_duration
            lyric_syllable = syl
            if si == 0 and previous_line == wr["lineIdx"]:
                lyric_syllable = " " + lyric_syllable
            midi = wr["midi"]
            if word_frames:
                midi, _ = midi_for_range(
                    WordTimestamp(
                        word=wr["word"],
                        start=syl_start,
                        end=syl_end,
                        midi=wr["midi"],
                        pitch_frames=word_frames,
                    ),
                    syl_start,
                    syl_end,
                )

            segments = _note_segments(
                word_frames,
                syl_start,
                syl_end,
                midi,
                amplitude_threshold,
                config,
            )
            for segment_index, (note_start, note_end, note_midi) in enumerate(segments):
                word_output.append(AlignedSyllable(
                    syllable=lyric_syllable if segment_index == 0 else "",
                    start=note_start,
                    end=note_end,
                    midi=note_midi,
                    pitch_end=note_end,
                ))

        if previous_line is not None and wr["lineIdx"] != previous_line and word_output:
            first = word_output[0]
            final_output.append(AlignedSyllable(
                syllable="",
                start=first.start,
                end=first.start,
                midi=0,
                is_line_break=True,
            ))
        final_output.extend(word_output)
        previous_line = wr["lineIdx"]

    # ── Insert line breaks ────────────────────────────────────────────────

    # ── Summary ───────────────────────────────────────────────────────────

    aligned_count = sum(1 for wr in word_results if wr["source"] == "sw_aligned")
    interp_count = sum(1 for wr in word_results if wr["source"] != "sw_aligned")
    total_syllables = sum(1 for s in final_output if not s.is_line_break and s.syllable)
    total_notes = sum(1 for s in final_output if not s.is_line_break)
    line_breaks = sum(1 for s in final_output if s.is_line_break)

    logger.info(f"Alignment complete")
    logger.info(f"  Lyric words:    {len(lyric_words)}")
    logger.info(f"  Aligned:        {aligned_count} ({100 * aligned_count / max(len(lyric_words), 1):.1f}%)")
    logger.info(f"  Interpolated:   {interp_count}")
    logger.info(f"  Syllables:      {total_syllables}")
    logger.info(f"  Notes:          {total_notes}")
    logger.info(f"  Line breaks:    {line_breaks}")

    # Optional debug output
    if config and config.debug_alignment:
        config.temp_path.mkdir(parents=True, exist_ok=True)
        debug_data = {
            "language": language,
            "lyricCharCount": len(lyric_chars),
            "whisperCharCount": len(whisper_chars),
            "whisperWordCount": len(whisper_words),
            "swMaxScore": sw_result["maxScore"],
            "swMaxPos": [sw_result["maxI"], sw_result["maxJ"]],
            "swBacktrackLength": len(sw_result["backtrack"]),
            "summary": {
                "totalLyricWords": len(lyric_words),
                "alignedWords": aligned_count,
                "interpolatedWords": interp_count,
                "totalSyllables": total_syllables,
                "totalNotes": total_notes,
                "lineBreaks": line_breaks,
            },
            "final_output": [asdict(s) for s in final_output],
        }
        debug_path = config.temp_path / "align_debug.json"
        debug_path.write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
        logger.info(f"Debug data written to {debug_path}")

        backtrace_path = config.temp_path / "align_backtrace.txt"
        backtrace_path.write_text(sw_backtrace_text, encoding="utf-8")
        logger.info(f"Backtrace written to {backtrace_path}")

        # Whisper words + pitch frames JSON
        whisper_pitch_data = {
            "words": [
                {
                    "word": ww.word,
                    "start": ww.start,
                    "end": ww.end,
                    "midi": ww.midi,
                    "pitchFrames": [
                        {"time": pf.time, "midi": pf.midi, "confidence": pf.confidence, "amplitude": pf.amplitude  }
                        for pf in ww.pitch_frames
                    ],
                }
                for ww in whisper_words
            ],
            "done": True,
        }
        whisper_json_path = config.temp_path / "whisper_pitch.json"
        whisper_json_path.write_text(json.dumps(whisper_pitch_data, indent=2), encoding="utf-8")
        logger.info(f"Whisper pitch data written to {whisper_json_path}")

        # Pitch visualization HTML
        from cli.html_preview import build_html
        html_title = f"{lyric_words[0]['word'] if lyric_words else 'Pitch'} — Whisper"
        html_content = build_html(whisper_pitch_data, html_title)
        html_path = config.temp_path / "whisper_pitch.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Pitch visualization written to {html_path}")

    return final_output
