"""Conversions between UltraStar beat units and absolute time."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def beat_duration_ms(bpm: float) -> float:
    """Return the duration of one UltraStar unit (one sixteenth note)."""

    if bpm <= 0:
        raise ValueError("BPM must be greater than zero")
    return 60_000.0 / bpm / 4.0


def beats_to_ms(beats: float, bpm: float) -> float:
    return beats * beat_duration_ms(bpm)


def beat_to_ms(beat: float, bpm: float, gap_ms: float = 0.0) -> float:
    return gap_ms + beats_to_ms(beat, bpm)


def ms_to_beats(
    milliseconds: float,
    bpm: float,
    gap_ms: float = 0.0,
    *,
    round_result: bool = True,
) -> float | int:
    """Convert absolute milliseconds into UltraStar beat units.

    Rounded conversion uses half-away-from-zero semantics, matching JavaScript's
    ``Math.round`` for the non-negative timestamps produced by the pipeline.
    """

    beats = (milliseconds - gap_ms) / beat_duration_ms(bpm)
    if not round_result:
        return beats
    return int(Decimal(str(beats)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
