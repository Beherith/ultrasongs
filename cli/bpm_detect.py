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


def detect_bpm(mp3_path: Path, config: Config) -> BpmResult:
    """Detect BPM and the time of the first beat from an audio file.

    The tempo is estimated with librosa.feature.tempo() over the full audio and
    reinforced by per-chunk estimates (the song is split into CHUNK_DURATION_S
    second chunks, the same check cli/debug_bpm.py performs). The final BPM is
    the median of all estimates, which makes it robust to outliers from
    intros, outros, or tempo changes.

    ``stable`` is False when the per-chunk BPM estimates disagree with each
    other beyond the tolerance, i.e. the song's tempo is not constant.

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
        stable = _chunks_stable(chunk_bpms)
        logger.debug(
            f"Tempo estimates: overall={overall:.2f}, "
            f"chunks={[round(c, 2) for c in chunk_bpms]}"
        )
        logger.info(
            f"BPM detected: {bpm:.1f} (first beat at {first_beat_ms:.0f} ms, "
            f"stable={stable})"
        )
        if not stable:
            logger.warning(
                "Tempo is not constant across the song; the fixed Ultrastar "
                f"beat grid may drift. Chunk BPMs: "
                f"{[round(c, 1) for c in chunk_bpms]}"
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
