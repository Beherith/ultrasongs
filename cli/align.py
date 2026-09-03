"""Lyric alignment: Smith-Waterman with phonetic scoring.

Ported from app/lib/align.ts (704 lines).
"""

import bisect
import itertools
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Callable

from cli.config import Config
from cli.logging_setup import get_logger
from cli.syllabify import split_word
from cli.pipeline_types import AlignedSyllable, Pause, PitchFrame, WordTimestamp
from matplotlib import pyplot as plt

logger = get_logger("cli.align")

# ── Constants ────────────────────────────────────────────────────────────────

MATCH_SCORE = 4
GAP_OPEN = 4
GAP_EXTEND = 0.5

# Monotonic counter so every _note_segments() invocation writes a unique plot file.
_NOTE_PLOT_COUNTER = itertools.count()

# Spectrogram analysis window in samples (bigger window -> finer frequency
# resolution at the cost of time resolution). 2048 samples ~= 46 ms @ 44.1 kHz,
# giving ~22 Hz frequency bins. Tunable if a shorter/longer slice is preferred.
_FFT_WINDOW = 2048
_FFT_HOP = 512
# How much extra audio (seconds) to show on each side of the word range so the
# spectrogram carries context beyond the word timestamps.
_FFT_CONTEXT_SEC = 0.3


# ── Character normalization ──────────────────────────────────────────────────

def normalize_char(c: str) -> str:
    """Lowercase and strip diacritics without discarding non-Latin scripts."""
    decomposed = unicodedata.normalize("NFD", c.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


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

def smith_waterman(
    lyric_chars: list[str],
    whisper_chars: list[str],
    score_fn: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    """Local sequence alignment with affine gaps.

    Args:
        lyric_chars: First sequence (rows).
        whisper_chars: Second sequence (columns).
        score_fn: Similarity function for two tokens; defaults to
            ``phonetic_score`` (character-level). Provide a custom scorer to
            align other token types (e.g. whole words).

    Returns dict with maxScore, maxI, maxJ, and backtrack path.
    """
    scorer = score_fn if score_fn is not None else phonetic_score

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
            s = scorer(lyric_chars[i - 1], whisper_chars[j - 1]) * MATCH_SCORE

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


def _syllable_intervals(
    syllables: list[str],
    char_alignments: list[dict[str, Any]],
    word_start: float,
    word_end: float,
) -> list[tuple[float, float]]:
    """Derive syllable intervals from matched WhisperX character anchors."""
    if not syllables:
        return []

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for syllable in syllables:
        ranges.append((cursor, cursor + len(syllable)))
        cursor += len(syllable)

    intervals: list[tuple[float, float] | None] = []
    for char_start, char_end in ranges:
        timed = [
            alignment
            for alignment in char_alignments
            if char_start <= alignment.get("lyricCharIdx", -1) < char_end
            and alignment.get("start") is not None
            and alignment.get("end") is not None
        ]
        if timed:
            intervals.append((
                max(word_start, min(float(item["start"]) for item in timed)),
                min(word_end, max(float(item["end"]) for item in timed)),
            ))
        else:
            intervals.append(None)

    if all(interval is None for interval in intervals):
        duration = max(0.01, (word_end - word_start) / len(syllables))
        return [
            (word_start + index * duration, word_start + (index + 1) * duration)
            for index in range(len(syllables))
        ]

    index = 0
    while index < len(intervals):
        if intervals[index] is not None:
            index += 1
            continue
        missing_start = index
        while index < len(intervals) and intervals[index] is None:
            index += 1
        missing_end = index
        previous = intervals[missing_start - 1] if missing_start > 0 else None
        following = intervals[missing_end] if missing_end < len(intervals) else None
        left = previous[1] if previous is not None else word_start
        right = following[0] if following is not None else word_end
        slot = max(0.0, right - left) / (missing_end - missing_start)
        for position in range(missing_start, missing_end):
            offset = position - missing_start
            intervals[position] = (left + offset * slot, left + (offset + 1) * slot)

    resolved = [interval for interval in intervals if interval is not None]
    for index in range(len(resolved) - 1):
        current_start, current_end = resolved[index]
        next_start, next_end = resolved[index + 1]
        if current_end > next_start:
            boundary = (current_end + next_start) / 2
            resolved[index] = (current_start, max(current_start, boundary))
            resolved[index + 1] = (min(next_end, boundary), next_end)
    return resolved


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


def _audio_spectrogram(
    audio: Any,
    sample_rate: int,
    t0: float,
    t1: float,
) -> tuple[Any, Any] | None:
    """Compute a dB-scaled FFT spectrogram (50-5000 Hz) over the time window.

    Returns ``(spec_db, times)`` where ``spec_db`` has the 50-5000 Hz bands
    along the first axis and ``times`` are the absolute frame times in seconds.
    Returns ``None`` when no usable audio is available.
    """
    if audio is None or sample_rate <= 0 or t1 <= t0:
        return None
    import numpy as np

    i0 = max(0, int(t0 * sample_rate))
    i1 = min(len(audio), int(t1 * sample_rate))
    available = i1 - i0
    n_fft = _FFT_WINDOW
    while n_fft > available and n_fft > 1024:
        n_fft //= 2
    hop = min(_FFT_HOP, n_fft)
    if available < n_fft:
        return None

    seg = np.asarray(audio[i0:i1], dtype=np.float64)
    n_frames = 1 + max(0, len(seg) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = seg[idx] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = t0 + (np.arange(n_frames) * hop) / sample_rate

    band = (freqs >= 50.0) & (freqs <= 5000.0)
    spec_db = 20.0 * np.log10(spec[:, band] + 1e-7)
    return spec_db.T, times


def _note_segments(
    frames: list[PitchFrame],
    start: float,
    end: float,
    fallback_midi: int,
    amplitude_threshold: float,
    config: Config,
    note_log: IO[str],
    syllable: str | None = None,
    word: str | None = None,
    audio: Any | None = None,
    sample_rate: int = 44100,
    word_start: float | None = None,
    word_end: float | None = None,
) -> list[tuple[float, float, int]]:
    """Trim a syllable to vocal activity and split sustained pitch changes.

    On every invocation a detailed diagnostic figure is rendered and saved to
    ``<temp>/note_segments_plots/``. The figure overlays every input frame,
    each threshold, the active/selected frames, islands, the smoothed median
    pitch, pre-merge pitch runs, the pitch-change split points, the merged
    notes, the final returned notes, and a full decision/reasoning log.
    """
    messages: list[str] = []

    def log(msg: str) -> None:
        messages.append(msg)
        note_log.write(msg + "\n")

    # Plot state accumulators. Populated as the algorithm progresses so any
    # early-return still renders a complete picture of everything decided so far.
    window: list[PitchFrame] = []
    active: list[PitchFrame] = []
    islands: list[list[PitchFrame]] = []
    selected: list[PitchFrame] = []
    smoothed: list[tuple[PitchFrame, int]] = []
    runs: list[list[tuple[PitchFrame, int]]] = []
    merged: list[list[tuple[PitchFrame, int]]] = []
    result: list[tuple[float, float, int]] = []
    split_points: list[dict[str, Any]] = []
    decision = "in progress"

    def _draw() -> None:
        if not config.note_segment_plots:
            return
        try:
            plot_dir = config.temp_path / "note_segments_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_index = next(_NOTE_PLOT_COUNTER)

            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(
                4, 1,
                height_ratios=[3.0, 1.6, 1.6, 2.0],
                hspace=0.4,
                left=0.06, right=0.98, top=0.92, bottom=0.05,
            )
            ax_midi = fig.add_subplot(gs[0])
            ax_conf = fig.add_subplot(gs[1], sharex=ax_midi)
            ax_amp = fig.add_subplot(gs[2], sharex=ax_midi)
            ax_txt = fig.add_subplot(gs[3])

            fig.suptitle(
                f"_note_segments #{plot_index}   syllable={syllable!r}  word={word!r}   "
                f"syll={start:.3f}-{end:.3f}s  word={word_start if word_start is not None else start:.3f}-"
                f"{word_end if word_end is not None else end:.3f}s   "
                f"fallback_midi={fallback_midi}   "
                f"decision={decision}",
                fontsize=11,
            )

            if window:
                wt = [f.time for f in window]
                x0, x1 = min(wt), max(wt)
            else:
                x0, x1 = start, end

            # The spectrogram (and shared x-range) covers the FULL word plus a
            # context margin on each side, not just the syllable under analysis.
            ws = word_start if word_start is not None else start
            we = word_end if word_end is not None else end
            if ws >= we:
                ws, we = start, end
            plot_t0 = ws - _FFT_CONTEXT_SEC
            plot_t1 = we + _FFT_CONTEXT_SEC
            ax_midi.set_xlim(plot_t0, plot_t1)

            # Context guides: faint red = syllable being trimmed; dotted = word bounds.
            for _axc in (ax_midi, ax_conf, ax_amp):
                _axc.axvspan(start, end, color="red", alpha=0.06, zorder=0)
                _axc.axvline(ws, color="0.45", ls=":", lw=0.8, zorder=0)
                _axc.axvline(we, color="0.45", ls=":", lw=0.8, zorder=0)

            # ── MIDI panel ──
            if window:
                sc = ax_midi.scatter(
                    [f.time for f in window],
                    [f.midi for f in window],
                    c=[f.confidence for f in window],
                    cmap="viridis", s=18, alpha=0.85, edgecolors="k", linewidths=0.3,
                    label="window frames (color=confidence)", zorder=2,
                )
                fig.colorbar(sc, ax=ax_midi, pad=0.01, label="crepe confidence")
                sel_times = {f.time for f in active}
                ax_midi.scatter(
                    [f.time for f in active],
                    [f.midi for f in active],
                    c="red", marker="x", s=45, zorder=4,
                    label="selected (active) frames",
                )
            else:
                sel_times = set()
                ax_midi.text(x0, 4, "no pitch frames in window",
                             ha="left", va="center", fontsize=9, color="red")

            if smoothed:
                ax_midi.step(
                    [f.time for f, _ in smoothed],
                    [m for _, m in smoothed],
                    where="mid", color="blue", lw=1.6, zorder=3,
                    label=f"smoothed median midi (win={config.note_smooth_window})",
                )

            if islands:
                island_colors = ["tab:blue", "tab:orange", "tab:gray", "tab:purple", "tab:brown"]
                for i, isl in enumerate(islands):
                    mid_t = (isl[0].time + isl[-1].time) / 2
                    energy = sum(f.amplitude * f.confidence for f in isl)
                    is_sel = set(map(id, isl)) == set(map(id, active))
                    ax_midi.axvspan(
                        isl[0].time, isl[-1].time,
                        color=island_colors[i % len(island_colors)], alpha=0.12, zorder=0,
                    )
                    ax_midi.annotate(
                        f"I{i} n={len(isl)} e={energy:.3f}" + (" (SEL)" if is_sel else ""),
                        xy=(mid_t, ax_midi.get_ylim()[1]),
                        ha="center", va="bottom", fontsize=6.5,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc=island_colors[i % len(island_colors)], ec="k"),
                    )

            for sp in split_points:
                ax_midi.axvline(sp["time"], color="crimson", ls="--", lw=0.9,
                                alpha=0.85, zorder=1)
                ax_midi.annotate(
                    f"SPLIT\n{sp['text']}",
                    xy=(sp["time"], ax_midi.get_ylim()[1]),
                    xytext=(sp["time"], ax_midi.get_ylim()[1] + 0.2),
                    ha="center", va="bottom", fontsize=6, color="crimson",
                )

            if result:
                note_colors = ["#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#3498db"]
                for i, (ns, ne, nm) in enumerate(result):
                    col = note_colors[i % len(note_colors)]
                    ax_midi.add_patch(plt.Rectangle(
                        (ns, nm - 0.5), ne - ns, 1.0,
                        facecolor=col, alpha=0.35, edgecolor=col, lw=1.8, zorder=1,
                    ))
                    ax_midi.text((ns + ne) / 2, nm, f"N{i + 1}", ha="center", va="center",
                                 fontsize=8, fontweight="bold", zorder=5,
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=col))

            ax_midi.axhline(fallback_midi, color="green", ls=":", lw=1.2,
                            label=f"fallback_midi={fallback_midi}")
            all_midi = [f.midi for f in window] + [m for _, m in smoothed] + [r[2] for r in result] + [fallback_midi]
            ax_midi.set_ylim(min(all_midi) - 1, max(all_midi) + 1.5)
            ax_midi.set_ylabel("midi note")
            ax_midi.set_title("MIDI pitch: raw frames, smoothed median, islands, splits, final notes",
                              fontsize=9)
            ax_midi.grid(True, alpha=0.25)
            ax_midi.legend(loc="upper right", fontsize=7, framealpha=0.9)

            # ── Confidence panel ──
            if window:
                ax_conf.scatter(
                    [f.time for f in window],
                    [f.confidence for f in window],
                    c=["red" if f.time in sel_times else "gray" for f in window],
                    s=12, alpha=0.7, zorder=2,
                    label="frame confidence (red=selected)",
                )
                ax_conf.axhline(config.note_min_confidence, color="orange", ls="-", lw=1.3,
                                label=f"note_min_confidence={config.note_min_confidence}")
                ax_conf.axhline(config.note_fallback_confidence, color="purple", ls="--", lw=1.3,
                                label=f"note_fallback_confidence={config.note_fallback_confidence}")
                ax_conf.set_ylim(0, 1.05)
                ax_conf.legend(loc="upper right", fontsize=7, framealpha=0.9)
            else:
                ax_conf.text(x0, 0.5, "no frames", ha="left", fontsize=8, color="red")
            ax_conf.set_ylabel("confidence")
            ax_conf.grid(True, alpha=0.25)

            # ── Amplitude panel (FFT spectrogram background) ──
            spec = _audio_spectrogram(audio, sample_rate, plot_t0, plot_t1)
            if spec is not None:
                spec_db, spec_times = spec
                ax_amp.imshow(
                    spec_db,
                    extent=[spec_times[0], spec_times[-1], 50.0, 5000.0],
                    aspect="auto", origin="lower", cmap="magma",
                    interpolation="nearest", alpha=0.9, zorder=0,
                )
                ax_amp.set_yscale("log", base=2)
                ax_amp.set_ylim(50, 5000)
                ax_amp.set_yticks([64, 128, 256, 512, 1024, 2048, 4096])
                ax_amp.set_yticklabels(["64", "128", "256", "512", "1k", "2k", "4k"])
                ax_amp.set_ylabel("frequency (Hz, log2)")
                ax_amp.set_title(
                    f"FFT spectrogram 50-5000 Hz (win={_FFT_WINDOW} samples, log2 freq) + amplitude",
                    fontsize=9,
                )
                amps = [f.amplitude for f in window]
                amps_max = max(amps) if amps else 0.0
                ax_amp2 = ax_amp.twinx()
                ax_amp2.plot([f.time for f in window], amps, color="cyan", lw=1.3,
                             zorder=3, label="amplitude envelope")
                ax_amp2.axhline(amplitude_threshold, color="lime", ls="--", lw=1.4,
                                zorder=3, label=f"amplitude_threshold={amplitude_threshold:.4f}")
                ax_amp2.set_ylabel("amplitude", color="cyan")
                ax_amp2.set_ylim(0, max(1e-3, amps_max * 1.25))
                ax_amp2.legend(loc="upper right", fontsize=6, framealpha=0.9)
            else:
                if window:
                    ax_amp.step([f.time for f in window], [f.amplitude for f in window],
                                where="mid", color="teal", lw=1.0, zorder=2,
                                label="frame amplitude")
                    ax_amp.fill_between([f.time for f in window], 0,
                                        [f.amplitude for f in window],
                                        color="teal", alpha=0.15, zorder=1)
                ax_amp.axhline(amplitude_threshold, color="green", ls="--", lw=1.3,
                               label=f"amplitude_threshold={amplitude_threshold:.4f}")
                ax_amp.set_ylabel("amplitude")
                ax_amp.legend(loc="upper right", fontsize=7, framealpha=0.9)
            ax_amp.set_xlabel("time (s)")
            ax_amp.grid(True, alpha=0.25)

            # ── Decision log panel ──
            ax_txt.axis("off")
            ax_txt.set_title(f"Decision log & reasoning  ({len(messages)} messages)",
                             fontsize=9, loc="left")
            log_text = "\n".join(messages) if messages else "(no decisions recorded)"
            ax_txt.text(0.01, 0.99, log_text, fontsize=7.5, family="monospace",
                        va="top", ha="left")

            safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in f"{word or 'w'}/{syllable or 's'}")
            fname = f"{plot_index:04d}_{safe}_{start:.3f}_{end:.3f}.png"
            fig.savefig(plot_dir / fname, dpi=140)
            logger.info(f"[note_segments] plot saved to {fname}")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - plotting must never break the pipeline
            logger.error("[note_segments] plot render failed: %s", exc)

    log(f"Processing syllable [{syllable}] of word [{word}]")
    dropout_gap = config.note_dropout_gap_ms / 1000
    min_duration = config.note_min_duration_ms / 1000
    window = [f for f in frames if start <= f.time <= end and f.midi > 0]
    active = [
        f for f in window
        if f.confidence >= config.note_min_confidence and f.amplitude >= amplitude_threshold
    ]
    max_conf = max((f.confidence for f in window), default=0.0)
    max_amp = max((f.amplitude for f in window), default=0.0)
    log(
        f"[note_segments] syllable {start:.3f}-{end:.3f}s fallback_midi={fallback_midi}: "
        f"window={len(window)}/{len(frames)} frames, "
        f"window max_conf={max_conf:.3f} max_amp={max_amp:.4f}, "
        f"active={len(active)} frames "
        f"(min_conf={config.note_min_confidence}, amp_thresh={amplitude_threshold:.4f})"
    )
    if not active:
        active = [f for f in window if f.confidence >= config.note_fallback_confidence]
        log(
            f"[note_segments] reason=below primary threshold "
            f"(max_conf={max_conf:.3f}, max_amp={max_amp:.4f} vs "
            f"min_conf={config.note_min_confidence}, amp_thresh={amplitude_threshold:.4f}); "
            f"decision=use fallback confidence {config.note_fallback_confidence}: "
            f"{len(active)} frames qualify"
        )
    if not active:
        decision = "no active frames -> untrimmed fallback note"
        log(
            f"[note_segments] reason=no active frames in window; "
            f"decision=return untrimmed fallback note "
            f"({start:.3f}-{end:.3f}s, midi={fallback_midi})"
        )
        _draw()
        return [(start, end, fallback_midi)]

    # Keep the strongest contiguous vocal island. Short confidence/energy
    # dropouts up to note_dropout_gap_ms are treated as part of the same sung sound.
    islands = []
    for frame in active:
        if not islands or frame.time - islands[-1][-1].time > dropout_gap:
            islands.append([frame])
        else:
            islands[-1].append(frame)
    if len(islands) > 1:
        island_stats = [
            f"island{i}: {len(isl)} frames {isl[0].time:.3f}-{isl[-1].time:.3f}s "
            f"energy={sum(f.amplitude * f.confidence for f in isl):.4f}"
            for i, isl in enumerate(islands)
        ]
        log(
            f"[note_segments] reason={len(islands)} vocal islands found "
            f"(dropout_gap={dropout_gap * 1000:.0f}ms); {', '.join(island_stats)}"
        )
    selected = max(
        islands,
        key=lambda island: sum(f.amplitude * f.confidence for f in island),
    )
    decision = f"keep strongest of {len(islands)} island(s)"
    log(
        f"[note_segments] decision=keep strongest island: "
        f"{len(selected)} frames {selected[0].time:.3f}-{selected[-1].time:.3f}s "
        f"energy={sum(f.amplitude * f.confidence for f in selected):.4f}"
    )
    active = selected

    # A median window removes vibrato/jitter before finding pitch changes.
    half = config.note_smooth_window // 2
    smoothed = []
    for i, frame in enumerate(active):
        nearby = active[max(0, i - half):i + half + 1]
        midi = _median([f.midi for f in nearby]) or fallback_midi
        smoothed.append((frame, midi))

    runs = []
    for item in smoothed:
        if not runs:
            runs.append([item])
            continue
        run_midi = _median([midi for _, midi in runs[-1]]) or fallback_midi
        gap = item[0].time - runs[-1][-1][0].time
        if gap <= dropout_gap and abs(item[1] - run_midi) <= config.note_pitch_tolerance:
            runs[-1].append(item)
        else:
            if runs[-1][-1][0].time > runs[-1][0][0].time:
                log(
                    f"[note_segments] pitch-change split after "
                    f"{runs[-1][0][0].time:.3f}-{runs[-1][-1][0].time:.3f}s "
                    f"(run_midi={run_midi}): next midi={item[1]} at {item[0].time:.3f}s, "
                    f"gap={gap * 1000:.0f}ms (dropout_allowed={dropout_gap * 1000:.0f}ms), "
                    f"drift={abs(item[1] - run_midi)} (tolerance={config.note_pitch_tolerance})"
                )
                split_points.append({
                    "time": item[0].time,
                    "text": f"drift={abs(item[1] - run_midi)} tol={config.note_pitch_tolerance} "
                            f"gap={gap * 1000:.0f}ms",
                })
            runs.append([item])
    log(
        f"[note_segments] pitch runs before merge: "
        + ", ".join(
            f"[{run[0][0].time:.3f}-{run[-1][0].time:.3f}s "
            f"midi={_median([m for _, m in run]) or fallback_midi}]"
            for run in runs
        )
    )

    # Very short pitch changes are detection noise, not singable notes.
    merged: list[list[tuple[PitchFrame, int]]] = []
    for run in runs:
        duration = run[-1][0].time - run[0][0].time
        if duration < min_duration and merged:
            log(
                f"[note_segments] reason=run {run[0][0].time:.3f}-{run[-1][0].time:.3f}s "
                f"is {duration * 1000:.0f}ms < min_duration {min_duration * 1000:.0f}ms; "
                f"decision=merge into previous note"
            )
            merged[-1].extend(run)
        else:
            merged.append(run)
    if len(merged) > 1 and merged[0][-1][0].time - merged[0][0][0].time < min_duration:
        first_dur = merged[0][-1][0].time - merged[0][0][0].time
        log(
            f"[note_segments] reason=first merged note is only {first_dur * 1000:.0f}ms "
            f"< min_duration {min_duration * 1000:.0f}ms; "
            f"decision=merge it into the second note"
        )
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    decision = f"{len(runs)} runs merged into {len(merged)} notes"
    log(f"[note_segments] merged note count: {len(runs)} runs -> {len(merged)} notes")

    frame_step = config.note_frame_step_ms / 1000
    if len(active) > 1:
        frame_step = max(0.001, active[1].time - active[0].time)

    result = []
    for idx, run in enumerate(merged):
        weights = [max(0.001, frame.amplitude * frame.confidence) for frame, _ in run]
        midi = round(sum(value * weight for (_, value), weight in zip(run, weights)) / sum(weights))
        run_start = max(start, run[0][0].time)
        run_end = min(end, run[-1][0].time + frame_step)
        if run_end > run_start:
            log(
                f"[note_segments] note {idx + 1}/{len(merged)}: "
                f"{run_start:.3f}-{run_end:.3f}s midi={midi} "
                f"({len(run)} frames, raw span {run[0][0].time:.3f}-{run[-1][0].time:.3f}s, "
                f"clamped to syllable {start:.3f}-{end:.3f}s, frame_step={frame_step * 1000:.1f}ms)"
            )
            result.append((run_start, run_end, midi))
        else:
            log(
                f"[note_segments] dropping merged run {idx}: "
                f"run_start={run_start:.3f} >= run_end={run_end:.3f} "
                f"(raw span {run[0][0].time:.3f}-{run[-1][0].time:.3f}s)"
            )

    if not result:
        decision = "all runs clamped empty -> fallback note"
        log(
            f"[note_segments] all merged runs clamped empty; "
            f"decision=return fallback note ({start:.3f}-{end:.3f}s, midi={fallback_midi})"
        )
        _draw()
        return [(start, end, fallback_midi)]
    decision = f"return {len(result)} note(s)"
    log(
        f"[note_segments] decision=return {len(result)} note(s) for "
        f"syllable {start:.3f}-{end:.3f}s"
    )
    _draw()
    return result


# ── Main alignment ───────────────────────────────────────────────────────────

def align_lyrics(
    lyrics: str,
    whisper_words: list[WordTimestamp],
    language: str,
    pauses: list[Pause] | None = None,
    config: Config | None = None,
    pitch_frames: list[PitchFrame] | None = None,
    audio_path: Path | None = None,
) -> list[AlignedSyllable]:
    """Align lyric text to WhisperX word/character timestamps using Smith-Waterman.

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
        whisper_words: Forced-aligned word and character timestamps.
        language: ISO 639-1 language code.
        pauses: Detected silence regions.
        config: Pipeline configuration.
        pitch_frames: Complete CREPE pitch, confidence, and amplitude timeline.
        audio_path: Optional vocal-stem path (the htdemucs-separated vocals).
            When ``config.note_segment_plots`` is enabled, this signal is
            decoded once so each note-segment plot can render an FFT
            spectrogram (of the vocals only) as its amplitude-panel background.

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
    logger.info(f"WhisperX words: {len(whisper_words)}")

    # ── Build lyric character sequence ────────────────────────────────────

    lyric_words: list[dict[str, Any]] = []
    for li, line in enumerate(raw_lines):
        for w in line.split():
            lyric_words.append({"word": w, "lineIdx": li})

    lyric_chars: list[dict[str, Any]] = []
    for wi, lw in enumerate(lyric_words):
        for char_idx, ch in enumerate(lw["word"]):
            lyric_chars.append({
                "orig": ch,
                "norm": normalize_char(ch),
                "wordIdx": wi,
                "charIdx": char_idx,
                "lineIdx": lw["lineIdx"],
            })
        if wi < len(lyric_words) - 1:
            lyric_chars.append({
                "orig": " ",
                "norm": " ",
                "wordIdx": wi,
                "lineIdx": lw["lineIdx"],
            })

    # ── Build WhisperX character sequence ─────────────────────────────────

    whisper_chars: list[dict[str, Any]] = []
    for wi, ww in enumerate(whisper_words):
        timed_chars_added = 0
        for aligned_char in ww.characters:
            normalized = normalize_char(aligned_char.char)
            for ch in normalized:
                whisper_chars.append({
                    "orig": aligned_char.char,
                    "norm": ch,
                    "wordIdx": wi,
                    "start": aligned_char.start,
                    "end": aligned_char.end,
                    "score": aligned_char.score,
                })
                timed_chars_added += 1
        if timed_chars_added == 0:
            for ch in normalize_char(ww.word):
                whisper_chars.append({
                    "orig": ch,
                    "norm": ch,
                    "wordIdx": wi,
                    "start": None,
                    "end": None,
                    "score": None,
                })
        if wi < len(whisper_words) - 1:
            whisper_chars.append({
                "orig": " ",
                "norm": " ",
                "wordIdx": wi,
                "start": None,
                "end": None,
                "score": None,
            })

    logger.info(f"Lyric chars: {len(lyric_chars)} ({len(lyric_words)} words)")
    logger.info(f"WhisperX chars: {len(whisper_chars)} ({len(whisper_words)} words)")

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
                    "lyricCharIdx": lc["charIdx"],
                    "whisperChar": wc["orig"],
                    "whisperWordIdx": wc["wordIdx"],
                    "score": step["score"],
                    "alignmentScore": wc.get("score"),
                    "start": wc.get("start"),
                    "end": wc.get("end"),
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
            "hasCharTiming": False,
            "pitchFrames": [],
        })

    for wi in range(len(word_results)):
        if not lyric_word_matched[wi]:
            _log_unmatched_word(wi, word_results[wi], whisper_words)
            continue
        idxs = sorted(word_results[wi]["whisperIdxs"])
        timed_chars = [
            char
            for char in word_results[wi]["charAlignments"]
            if char.get("start") is not None and char.get("end") is not None
        ]
        if timed_chars:
            word_results[wi]["start"] = min(float(char["start"]) for char in timed_chars)
            word_results[wi]["end"] = max(float(char["end"]) for char in timed_chars)
            word_results[wi]["hasCharTiming"] = True
        else:
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

    # If forced character timestamps are unavailable, divide a WhisperX token
    # shared by several lyric words instead of assigning its full interval to all.
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
        if end_wi - wi > 1 and not any(wr["hasCharTiming"] for wr in word_results[wi:end_wi]):
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

    # Decode the raw audio once (only when plots are enabled) so the
    # note-segment figures can render an FFT spectrogram background.
    audio: Any | None = None
    if config.note_segment_plots and audio_path is not None:
        try:
            import numpy as np
            from cli.ffmpeg_pcm import extract_pcm

            pcm = extract_pcm(audio_path, config.sample_rate)
            audio = np.frombuffer(pcm, dtype=np.float32)
        except Exception as exc:
            logger.warning("Note-segment spectrogram unavailable (%s)", exc)
            audio = None

    # Route [note_segments] diagnostics to a text file instead of the console.
    config.temp_path.mkdir(parents=True, exist_ok=True)
    note_log_path = config.temp_path / "note_segments.txt"
    note_log = note_log_path.open("w", encoding="utf-8")
    try:
        for wr in word_results:
            syllables = split_word(wr["word"], language)
            # print(f"Word '{wr['word']}' split into syllables: {syllables}")
            syllable_intervals = _syllable_intervals(
                syllables,
                wr["charAlignments"],
                wr["start"],
                wr["end"],
            )
            if pitch_frames:
                lo = bisect.bisect_left(pitch_times, wr["start"])
                hi = bisect.bisect_right(pitch_times, wr["end"])
                word_frames = pitch_frames[lo:hi]
            else:
                word_frames = wr["pitchFrames"]
            word_output: list[AlignedSyllable] = []

            for si, syl in enumerate(syllables):
                syl_start, syl_end = syllable_intervals[si]
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
                    note_log,
                    syllable=lyric_syllable,
                    word=wr["word"],
                    audio=audio,
                    sample_rate=config.sample_rate,
                    word_start=wr["start"],
                    word_end=wr["end"],
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
    finally:
        note_log.close()
    logger.info(f"Note segment diagnostics written to {note_log_path}")

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
            "whisperxCharCount": len(whisper_chars),
            "whisperxWordCount": len(whisper_words),
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

        # WhisperX words, character alignments, and pitch frames JSON
        whisperx_pitch_data = {
            "words": [
                {
                    "word": ww.word,
                    "start": ww.start,
                    "end": ww.end,
                    "midi": ww.midi,
                    "characters": [char.to_dict() for char in ww.characters],
                    "pitchFrames": [
                        {"time": pf.time, "midi": pf.midi, "confidence": pf.confidence, "amplitude": pf.amplitude  }
                        for pf in ww.pitch_frames
                    ],
                }
                for ww in whisper_words
            ],
            "done": True,
        }
        whisperx_json_path = config.temp_path / "whisperx_pitch.json"
        whisperx_json_path.write_text(json.dumps(whisperx_pitch_data, indent=2), encoding="utf-8")
        logger.info(f"WhisperX pitch data written to {whisperx_json_path}")

        # Pitch visualization HTML
        from cli.html_preview import build_html
        html_title = f"{lyric_words[0]['word'] if lyric_words else 'Pitch'} — WhisperX"
        html_content = build_html(whisperx_pitch_data, html_title)
        html_path = config.temp_path / "whisperx_pitch.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Pitch visualization written to {html_path}")

    return final_output
