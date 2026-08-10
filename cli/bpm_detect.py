"""BPM detection using librosa."""

from pathlib import Path

import librosa
import numpy as np

from cli.config import Config
from cli.logging_setup import get_logger

logger = get_logger("cli.bpm_detect")

# Fallback BPM when detection fails
_FALLBACK_BPM = 120.0


def detect_bpm(mp3_path: Path, config: Config) -> float:
    """Detect BPM from an MP3 file using librosa.

    Uses librosa.feature.tempo() which returns multiple BPM estimates.
    Takes the median as the final estimate.

    Args:
        mp3_path: Path to the MP3 file.
        config: Pipeline configuration.

    Returns:
        Estimated BPM as a float. Falls back to 120.0 on any error.
    """
    try:
        logger.info(f"Detecting BPM from {mp3_path}")
        audio, sr = librosa.load(str(mp3_path), sr=22050, mono=True)
        tempo_estimates = librosa.feature.tempo(y=audio, sr=sr)
        bpm = float(np.median(tempo_estimates))
        logger.info(f"BPM detected: {bpm:.1f}")
        return bpm
    except Exception as exc:
        logger.warning(f"BPM detection failed ({exc}), using fallback {_FALLBACK_BPM}")
        return _FALLBACK_BPM
