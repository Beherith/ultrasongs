"""BPM and first-beat detection using librosa."""

from pathlib import Path

import librosa
import numpy as np

from cli.config import Config
from cli.logging_setup import get_logger
from cli.pipeline_types import BpmResult

logger = get_logger("cli.bpm_detect")

# Fallback BPM when detection fails
_FALLBACK_BPM = 120.0
# Analysis sample rate for tempo detection
_SAMPLE_RATE = 22050
# Length of the chunks used to verify the tempo is constant across the song
CHUNK_DURATION_S = 30.0
# Chunks shorter than this are too short for a reliable tempo estimate
_MIN_CHUNK_S = 10.0
# Tolerance (BPM) for a chunk to count as matching the song's tempo
_STABILITY_ABS_TOLERANCE = 2.0
_STABILITY_REL_TOLERANCE = 0.05
# A break in the onsets longer than this many beats counts as a pause whose
# resumption is checked for a beat-grid offset
_MIN_PAUSE_BEATS = 2.0
# Maximum misalignment (fraction of a beat) when the beat resumes after a
# pause; larger offsets mean the beat continued at a different phase
_PAUSE_PHASE_TOLERANCE = 0.2
# Onset intervals outside this range (relative to the expected beat length)
# are not used to estimate the true beat length (rests, double beats, ...)
_BEAT_INTERVAL_MIN = 0.6
_BEAT_INTERVAL_MAX = 1.1


def detect_bpm(mp3_path: Path, config: Config) -> BpmResult:
    """Detect BPM and the time of the first beat from an audio file.

    The tempo is estimated with librosa.feature.tempo() over the full audio and
    reinforced by per-chunk estimates (the song is split into CHUNK_DURATION_S
    second chunks, the same check cli/debug_bpm.py performs). The final BPM is
    the median of all estimates, which makes it robust to outliers from
    intros, outros, or tempo changes.

    ``stable`` is False when the per-chunk BPM estimates disagree with each
    other beyond the tolerance (the song's tempo is not constant), or when the
    beat stops and then resumes with the same BPM but a different offset (a
    beat-grid phase discontinuity), which the per-chunk BPM check cannot see.

    The first beat time (milliseconds from the start of the audio) is found
    with librosa.beat.beat_track() using the estimated BPM as the tempo prior.
    It is used as Ultrastar's #GAP so notes land on the song's actual beat grid.

    Args:
        mp3_path: Path to the MP3 file.
        config: Pipeline configuration.

    Returns:
        BpmResult with bpm, first_beat_ms, stable flag, and per-chunk BPMs.
        Falls back to 120 BPM and first beat at 0 ms on any error.
    """
    try:
        logger.info(f"Detecting BPM from {mp3_path}")
        audio, sr = librosa.load(str(mp3_path), sr=_SAMPLE_RATE, mono=True)
        overall = _median_tempo(audio, sr)
        chunk_bpms = _chunk_bpms(audio, sr)
        bpm = float(np.median([overall] + chunk_bpms))
        first_beat_ms = _detect_first_beat_ms(audio, sr, bpm)
        bpm_stable = _chunks_stable(chunk_bpms)
        onset_times = _onset_times(audio, sr)
        phase_stable = _phases_stable(onset_times, bpm)
        stable = bpm_stable and phase_stable
        logger.debug(
            f"Tempo estimates: overall={overall:.2f}, "
            f"chunks={[round(c, 2) for c in chunk_bpms]}"
        )
        logger.info(
            f"BPM detected: {bpm:.1f} (first beat at {first_beat_ms:.0f} ms, "
            f"stable={stable})"
        )
        if not bpm_stable:
            logger.warning(
                "Tempo is not constant across the song; the fixed Ultrastar "
                f"beat grid may drift. Chunk BPMs: "
                f"{[round(c, 1) for c in chunk_bpms]}"
            )
        if not phase_stable:
            logger.warning(
                "The beat stops and resumes with a different offset at the "
                "same BPM (beat grid phase discontinuity); the fixed Ultrastar "
                "beat grid may drift."
            )
        return BpmResult(
            bpm=bpm,
            first_beat_ms=first_beat_ms,
            stable=stable,
            chunk_bpms=chunk_bpms,
        )
    except Exception as exc:
        logger.warning(f"BPM detection failed ({exc}), using fallback {_FALLBACK_BPM}")
        return BpmResult(bpm=_FALLBACK_BPM, first_beat_ms=0.0, stable=True)


def _median_tempo(audio: np.ndarray, sr: int) -> float:
    """Median of librosa's tempo estimates for an audio buffer."""
    estimates = np.atleast_1d(librosa.feature.tempo(y=audio, sr=sr))
    return float(np.median(estimates))


def _chunk_bpms(audio: np.ndarray, sr: int) -> list[float]:
    """Per-chunk median BPM estimates for 30 s chunks of the audio.

    Chunks shorter than _MIN_CHUNK_S (e.g. the tail of the song) are skipped
    because the tempo estimate would be unreliable.
    """
    chunk_len = int(CHUNK_DURATION_S * sr)
    min_len = int(_MIN_CHUNK_S * sr)
    estimates: list[float] = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < min_len:
            continue
        estimates.append(_median_tempo(chunk, sr))
    return estimates


def _chunks_stable(chunk_bpms: list[float]) -> bool:
    """True when all chunk BPM estimates agree within the tolerance."""
    if not chunk_bpms:
        return True
    ref = float(np.median(chunk_bpms))
    tolerance = max(_STABILITY_ABS_TOLERANCE, _STABILITY_REL_TOLERANCE * ref)
    return all(abs(c - ref) <= tolerance for c in chunk_bpms)


def _onset_times(audio: np.ndarray, sr: int) -> np.ndarray:
    """Times (seconds) of the detected onsets in the audio."""
    hop = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def _beat_interval(onset_times: np.ndarray, bpm: float) -> float:
    """Best estimate of the true beat length, in seconds.

    The median of the onset-to-onset intervals that look like a single beat.
    Estimating from the onsets' own spacing (rather than the tempo estimate)
    keeps the phase check unbiased even when librosa's BPM is slightly off.
    """
    expected = 60.0 / bpm
    diffs = np.diff(onset_times)
    intervals = diffs[
        (diffs >= _BEAT_INTERVAL_MIN * expected)
        & (diffs <= _BEAT_INTERVAL_MAX * expected)
    ]
    if len(intervals) == 0:
        return expected
    return float(np.median(intervals))


def _phases_stable(onset_times: np.ndarray, bpm: float) -> bool:
    """True when the beat resumes on the song's beat grid after every pause.

    A pause is a break in the onsets longer than _MIN_PAUSE_BEATS beats (the
    beat stops). On a continuous grid the time from the last onset before the
    pause to the first onset after it is an integer number of beats. If the
    beat resumes with the same BPM but a different offset, that interval
    leaves a sub-beat remainder, which is what this check catches even when
    every 30 s chunk reports the same BPM.
    """
    if len(onset_times) < 2:
        return True
    beat_len = _beat_interval(onset_times, bpm)
    if beat_len <= 0:
        return True
    tolerance = _PAUSE_PHASE_TOLERANCE * beat_len
    min_pause = _MIN_PAUSE_BEATS * beat_len
    for before, after in zip(onset_times[:-1], onset_times[1:]):
        interval = after - before
        if interval <= min_pause:
            continue
        beats = int(np.rint(interval / beat_len))
        residual = abs(interval - beats * beat_len)
        if residual > tolerance:
            logger.debug(
                f"Beat resumed off the grid after a pause: {interval:.2f} s "
                f"from {before:.2f} s to {after:.2f} s is "
                f"{residual / beat_len:.2f} beats off "
                f"(tolerance {tolerance / beat_len:.2f} beats)"
            )
            return False
    return True


def _detect_first_beat_ms(audio: np.ndarray, sr: int, bpm: float) -> float:
    """Time in milliseconds of the first beat detected in the audio.

    Runs librosa's beat tracker with the estimated BPM as tempo prior; the
    tracker's grid phase is chosen from the onsets, so the first returned beat
    is the song's actual downbeat snapped to the BPM grid.

    The tracker can skip the very first onset (e.g. an attack exactly at t=0),
    so the locked grid is walked back to t=0 and the earliest grid position
    with a strong onset is used.
    """
    try:
        hop = 512
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
        _, beats = librosa.beat.beat_track(
            y=audio, sr=sr, onset_envelope=onset_env,
            start_bpm=bpm, units="time",
        )
        if len(beats) == 0:
            logger.warning("No beats detected; assuming first beat at 0 ms")
            return 0.0
        beat_s = float(beats[1] - beats[0]) if len(beats) > 1 else 60.0 / bpm
        phase = float(beats[0]) % beat_s
        # The onset envelope measures frame-to-frame energy *increase*, so it
        # cannot represent an attack at exactly t=0. Check the raw energy at
        # the start of the audio for a missed first beat.
        if phase < beat_s / 4:
            rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr, hop_length=hop
            )
            if (
                len(onset_frames) > 0
                and rms[0] >= 0.5 * float(np.median(rms[onset_frames]))
            ):
                return phase * 1000.0
        max_env = float(np.max(onset_env))
        threshold = 0.5 * max_env if max_env > 0 else 0.0
        for t in np.arange(phase, float(beats[0]) + beat_s / 2, beat_s):
            frame = int(round(t * sr / hop))
            lo = max(0, frame - 2)
            hi = min(len(onset_env), frame + 3)
            if hi > lo and float(np.max(onset_env[lo:hi])) >= threshold:
                return float(t) * 1000.0
        return float(beats[0]) * 1000.0
    except Exception as exc:
        logger.warning(f"First-beat detection failed ({exc}), assuming first beat at 0 ms")
        return 0.0
