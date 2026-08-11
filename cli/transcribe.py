"""Transcription pipeline: Demucs separation, torchcrepe pitch, Whisper transcription.

Refactored from python/transcribe_service.py into importable CLI modules.
Heavy dependencies are lazy-loaded to allow import without GPU packages.
"""

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cli.config import Config
from cli.logging_setup import get_logger
from cli.pipeline_types import Pause, PitchFrame, TranscribeResult, WordTimestamp

logger = get_logger("cli.transcribe")

# ── Constants ────────────────────────────────────────────────────────────────

CREPE_SR = 16000
DEMUCS_SR = 44100
DEMUCS_VOCALS_IDX = 3  # sources: ['drums', 'bass', 'other', 'vocals']


def _get_device(device_override: str = "auto") -> str:
    import torch
    if device_override == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_override != "auto":
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _get_device()


# ── Helpers ──────────────────────────────────────────────────────────────────

def hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    import numpy as np
    if hz <= 0:
        return 60
    return round(12 * np.log2(hz / 440) + 69)


def to_mp3_bytes(audio, sr: int, bitrate: int = 128) -> bytes:
    """Encode float32 audio to MP3 bytes using lameenc."""
    import lameenc
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(5)
    data = enc.encode(pcm.tobytes())
    data += enc.flush()
    return data


# ── Demucs separation ────────────────────────────────────────────────────────

def separate_all(audio, sr: int, demucs_model: str = "htdemucs"):
    """Demucs separation -> (vocals_mono, accompaniment_mono, out_sr)."""
    import numpy as np
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    logger.info(f"Loading Demucs model: {demucs_model}")
    print(f"[demucs] Running on {DEVICE}")
    model = get_model(demucs_model)
    model.eval()
    if DEVICE == "cuda":
        model = model.cuda()

    try:
        mono = torch.from_numpy(audio.astype(np.float32))
        if sr != DEMUCS_SR:
            mono = torchaudio.functional.resample(mono.unsqueeze(0), sr, DEMUCS_SR).squeeze(0)
        wav = mono.unsqueeze(0).expand(2, -1).unsqueeze(0)
        if DEVICE == "cuda":
            wav = wav.cuda()
        with torch.no_grad():
            sources = apply_model(model, wav, device=DEVICE)
        n_sources = sources.shape[1]
        vocals = sources[0, DEMUCS_VOCALS_IDX].mean(0).cpu().numpy()
        acc_stems = [sources[0, i] for i in range(n_sources) if i != DEMUCS_VOCALS_IDX]
        accompaniment = torch.stack(acc_stems).sum(0).mean(0).cpu().numpy()
        del sources, wav, mono
    finally:
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"VRAM after Demucs unload: {torch.cuda.memory_allocated()//1024**2} MB")

    return vocals, accompaniment, DEMUCS_SR


# ── Pitch analysis ──────────────────────────────────────────────────────────

def _compute_band_energy(audio, sr: int, hop_length: int, fmin: float = 60.0, fmax: float = 1000.0, n_frames: int | None = None):
    """Compute per-frame RMS energy in the given frequency band via FFT.

    Uses a Hann-windowed frame of 1024 samples for good frequency resolution
    (~15.6 Hz at 16 kHz sample rate).  Frames are aligned to crepe's hop grid.

    If ``n_frames`` is given, the result is truncated or zero-padded to match
    the caller's expected frame count (e.g. crepe's padded frame count).
    """
    import numpy as np
    frame_length = 1024
    n = (len(audio) - frame_length) // hop_length + 1
    if n <= 0:
        result = np.zeros(n_frames or 0, dtype=np.float32)
        return result

    window = np.hanning(frame_length)
    idx = np.arange(n)[:, None] * hop_length + np.arange(frame_length)
    frames = audio[idx] * window

    fft_results = np.fft.rfft(frames, axis=1)
    fft_mags = np.abs(fft_results)

    freq_res = sr / frame_length
    bin_min = max(0, int(fmin / freq_res))
    bin_max = int(fmax / freq_res) + 1
    bin_max = min(bin_max, fft_mags.shape[1])

    band_power = np.sum(fft_mags[:, bin_min:bin_max] ** 2, axis=1)
    n_bins = max(bin_max - bin_min, 1)
    energies = np.sqrt(band_power / n_bins).astype(np.float32)

    # Truncate or zero-pad to match crepe's frame count
    if n_frames is not None:
        if len(energies) < n_frames:
            energies = np.pad(energies, (0, n_frames - len(energies)))
        elif len(energies) > n_frames:
            energies = energies[:n_frames]

    return energies


def analyze_pitch(vocals, sr: int, fmin: float = 65.41, fmax: float = 1046.5, hop_ms: int = 10):
    """Run torchcrepe on the entire vocals track."""
    import numpy as np
    import torch
    import torchcrepe
    import torchaudio

    hop_length = int(CREPE_SR * hop_ms / 1000)
    audio_t = torch.from_numpy(vocals.astype(np.float32))
    if sr != CREPE_SR:
        audio_t = torchaudio.functional.resample(
            audio_t.unsqueeze(0), sr, CREPE_SR
        ).squeeze(0)

    logger.info(f"Running torchcrepe: {len(audio_t)} samples, hop={hop_ms}ms")
    print(f"[crepe] Running on {DEVICE}")
    pitch, periodicity = torchcrepe.predict(
        audio_t.unsqueeze(0),
        sample_rate=CREPE_SR,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=2048,
        device=DEVICE,
        pad=True,
    )

    n = pitch.shape[1]
    times = np.arange(n, dtype=np.float32) * (hop_length / CREPE_SR)
    pct = 100 * (periodicity[0].cpu().numpy() > 0.5).sum() / max(n, 1)
    logger.info(f"Pitch analysis: {n} frames, {pct:.0f}% confident")

    # Compute per-frame band-limited energy (60-1000 Hz) as amplitude proxy
    energies = _compute_band_energy(audio_t.numpy(), CREPE_SR, hop_length, n_frames=n)

    return times, pitch[0].cpu().numpy(), periodicity[0].cpu().numpy(), energies


def get_midi_for_word(times, freqs, confs, start_sec: float, end_sec: float) -> int:
    """Get median MIDI note for a word's time range."""
    import numpy as np
    for threshold in (0.5, 0.3, 0.1):
        mask = (times >= start_sec) & (times <= end_sec) & (confs > threshold)
        sel = freqs[mask]
        sel = sel[sel > 0]
        if len(sel) > 0:
            return hz_to_midi(float(np.median(sel)))
    return 60


def get_pitch_frames_for_word(times, freqs, confs, energies, start_sec: float, end_sec: float) -> list[PitchFrame]:
    """Get all pitch frames for a word's time range."""
    mask = (times >= start_sec) & (times <= end_sec) & (confs > 0.1) & (freqs > 0)
    return [
        PitchFrame(
            time=float(t),
            midi=hz_to_midi(float(f)),
            confidence=float(c),
            amplitude=float(e),
        )
        for t, f, c, e in zip(times[mask], freqs[mask], confs[mask], energies[mask])
    ]


def add_pitch_to_words(raw_words: list[dict[str, Any]], times, freqs, confs, energies) -> list[WordTimestamp]:
    """Attach pitch data to raw Whisper word results."""
    words = []
    for w in raw_words:
        start = float(w["start"])
        end = float(w["end"])
        words.append(WordTimestamp(
            word=str(w["word"]).strip(),
            start=start,
            end=end,
            midi=get_midi_for_word(times, freqs, confs, start, end),
            pitch_frames=get_pitch_frames_for_word(times, freqs, confs, energies, start, end),
        ))
    return words


# ── Pause detection ──────────────────────────────────────────────────────────

def detect_pauses(
    vocals,
    sr: int,
    frame_ms: int = 25,
    hop_ms: int = 10,
    min_silence_ms: int = 400,
    threshold_ratio: float = 0.05,
) -> list[Pause]:
    """Detect silence regions in Demucs vocals via RMS energy."""
    import numpy as np
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(1, (len(vocals) - frame_len) // hop_len + 1)

    idx = np.arange(n_frames)[:, None] * hop_len + np.arange(frame_len)
    idx = np.clip(idx, 0, len(vocals) - 1)
    rms = np.sqrt((vocals[idx] ** 2).mean(axis=1))

    p95 = float(np.percentile(rms, 95))
    if p95 < 1e-6:
        return []

    is_silent = rms < threshold_ratio * p95
    min_frames = max(1, round(min_silence_ms / hop_ms))
    times = np.arange(n_frames, dtype=np.float32) * (hop_ms / 1000)

    pauses: list[Pause] = []
    in_silence = False
    sil_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            sil_start = i
        elif not silent and in_silence:
            in_silence = False
            if i - sil_start >= min_frames:
                pauses.append(Pause(start=float(times[sil_start]), end=float(times[i])))

    if in_silence and n_frames - sil_start >= min_frames:
        pauses.append(Pause(start=float(times[sil_start]), end=float(times[n_frames - 1])))

    return pauses


# ── Whisper transcription ───────────────────────────────────────────────────

def _transcribe_with_whisperx(vocals_wav_path: str, prompt: str = "", whisperx_model: str = "small") -> tuple[list[dict[str, Any]], str]:
    """Transcribe using WhisperX via subprocess (same as web service)."""
    here = Path(__file__).parent
    worker = here.parent / "python" / "whisperx_worker.py"
    default_python = here.parent / ".venv" / "Scripts" / "python.exe"
    python_exe = Path(os.environ.get("WHISPERX_PYTHON", str(default_python)))

    if not python_exe.exists():
        raise RuntimeError(f"WhisperX Python not found: {python_exe}")
    if not worker.exists():
        raise RuntimeError(f"WhisperX worker not found: {worker}")

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    cmd = [
        str(python_exe), str(worker),
        "--audio", vocals_wav_path,
        "--model", whisperx_model,
    ]
    language = os.environ.get("WHISPERX_LANGUAGE", "")
    if language:
        cmd.extend(["--language", language])
    if prompt:
        cmd.extend(["--prompt", prompt])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.stderr:
        logger.warning(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"WhisperX failed (exit {proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    return payload.get("words", []), payload.get("language") or language or "en"


# ── Main transcription function ─────────────────────────────────────────────

def transcribe(mp3_path: Path, lyrics_prompt: str | None, config: Config) -> TranscribeResult:
    """Run the full transcription pipeline."""
    import torch
    import soundfile as sf

    global DEVICE
    DEVICE = _get_device(config.device)

    logger.info(f"Starting transcription: {mp3_path}")
    logger.info(f"Device: {DEVICE}, Whisper model: {config.whisper_model}")

    base = mp3_path.with_suffix("")
    vocals_mp3 = base.with_name(base.name + "_vocals.mp3")
    acc_mp3 = base.with_name(base.name + "_accompaniment.mp3")
    vocals_wav = base.with_name(base.name + "_vocals.wav")

    # Step 1: Load audio
    logger.info("Step 1/6: Loading audio…")
    audio, sr = sf.read(str(mp3_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Step 2: Demucs separation
    logger.info("Step 2/6: Separating vocals with Demucs…")
    vocals, accompaniment, out_sr = separate_all(audio, sr, config.demucs_model)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Step 3: Save stems
    logger.info("Step 3/6: Saving vocal stems…")
    with open(vocals_mp3, "wb") as f:
        f.write(to_mp3_bytes(vocals, out_sr))
    with open(acc_mp3, "wb") as f:
        f.write(to_mp3_bytes(accompaniment, out_sr))
    sf.write(str(vocals_wav), vocals, out_sr)

    # Step 4: Pitch analysis
    logger.info("Step 4/6: Analyzing pitch with torchcrepe…")
    times, freqs, confs, energies = analyze_pitch(
        vocals, out_sr,
        fmin=config.pitch_min_hz,
        fmax=config.pitch_max_hz,
        hop_ms=config.crepe_hop_ms,
    )

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Step 5: Pause detection
    pauses = detect_pauses(
        vocals, out_sr,
        min_silence_ms=config.pause_min_silence_ms,
        threshold_ratio=config.pause_threshold_pct / 100.0,
    )
    logger.info(f"Detected {len(pauses)} pause regions")

    # Step 6: Whisper transcription
    logger.info("Step 5/6: Transcribing with Whisper…")
    from faster_whisper import WhisperModel
    compute_type = "int8" if "large" in config.whisper_model and DEVICE == "cuda" else "auto"
    print(f"[whisper] Running on {DEVICE} (compute_type={compute_type})")
    whisper = WhisperModel(config.whisper_model, device=DEVICE, compute_type=compute_type)

    segs, info = whisper.transcribe(
        str(vocals_wav),
        word_timestamps=True,
        initial_prompt=lyrics_prompt,
        language=config.whisper_language if config.whisper_language else None,
    )

    raw_words = []
    for seg in segs:
        for w in seg.words:
            raw_words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
            })

    if not raw_words:
        raise RuntimeError("Whisper returned empty transcription")

    words = add_pitch_to_words(raw_words, times, freqs, confs, energies)

    # Cleanup temporary WAV
    if vocals_wav.exists():
        vocals_wav.unlink()

    logger.info(f"Transcription complete: {len(words)} words, language={info.language}")

    return TranscribeResult(
        words=words,
        language=info.language,
        vocals_path=str(vocals_mp3),
        accompaniment_path=str(acc_mp3),
        pauses=pauses,
    )
