"""Metadata table driving the auto-generated settings form.

Covers every key in cli/config.py. Kinds: select / number / float /
checkbox / text. Enums mirror the validation sets in cli/config.py.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SettingMeta:
    key: str
    kind: str                      # select | number | float | checkbox | text
    group: str
    help: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)   # for select
    min_value: float | None = None                            # for number/float
    max_value: float | None = None
    step: float = 1


GROUP_MODELS = "Models & backends"
GROUP_ASR = "ASR & timing"
GROUP_PITCH = "Pitch & activity"
GROUP_NOTES = "Note segmentation"
GROUP_PAUSES = "Pauses"
GROUP_OUTPUT = "Output & debug"

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large")
BACKENDS = ("faster-whisper", "whisperx")
FW_COMPUTE_TYPES = ("auto", "default", "float16", "float32", "int8", "int8_float16")
WX_COMPUTE_TYPES = ("default", "float16", "float32", "int8")
INTERPOLATE_METHODS = ("nearest", "linear", "ignore")
DEVICES = ("auto", "cuda", "cpu")

# Whisper model languages (ISO-639-1 codes from the Whisper tokenizer).
# "om" and "or" have no wav2vec2 alignment model; still listed since
# faster-whisper transcription works with them.
WHISPER_LANGUAGES = (
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
    "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "om", "or", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su",
    "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur",
    "uz", "vi", "yi", "yo", "yue", "zh",
)

# ("Auto-detect" is stored as ""; see app.py which maps it)
LANGUAGE_CHOICES = ("",) + WHISPER_LANGUAGES

SETTINGS: list[SettingMeta] = [
    # ── Models & backends ────────────────────────────────────────────────
    SettingMeta("device", "select", GROUP_MODELS,
                "GPU/CPU selection. auto uses CUDA if available.", DEVICES),
    SettingMeta("transcription_backend", "select", GROUP_MODELS,
                "faster-whisper: hybrid multi-pass consensus path (default). whisperx: legacy VAD ASR path.",
                BACKENDS),
    SettingMeta("whisper_model", "select", GROUP_MODELS,
                "Whisper ASR model size. medium balances accuracy and speed; large is slower and uses more VRAM.",
                WHISPER_MODELS),
    SettingMeta("whisper_language", "select", GROUP_MODELS,
                "Language hint (ISO-639-1). Empty lets Whisper auto-detect.",
                LANGUAGE_CHOICES),
    SettingMeta("faster_whisper_compute_type", "select", GROUP_MODELS,
                "CTranslate2 compute type for standalone faster-whisper.", FW_COMPUTE_TYPES),
    SettingMeta("whisperx_batch_size", "number", GROUP_MODELS,
                "Audio chunks per WhisperX inference batch (legacy backend). Min 1.",
                min_value=1),
    SettingMeta("whisperx_compute_type", "select", GROUP_MODELS,
                "CTranslate2 compute type used by WhisperX.", WX_COMPUTE_TYPES),
    SettingMeta("whisperx_align_model", "text", GROUP_MODELS,
                "Optional wav2vec2 alignment model override. Empty = language default."),
    SettingMeta("whisperx_interpolate_method", "select", GROUP_MODELS,
                "How WhisperX fills timestamps for unalignable characters.",
                INTERPOLATE_METHODS),
    SettingMeta("whisperx_chunk_pause_ms", "number", GROUP_ASR,
                "Split lyrics/audio at pauses strictly longer than this (ms).",
                min_value=0),
    SettingMeta("whisperx_align_runs", "number", GROUP_ASR,
                "WhisperX forced-alignment passes per chunk. Must be a positive odd number.",
                min_value=1, step=2),
    SettingMeta("transcribe_runs", "number", GROUP_MODELS,
                "Demucs + ASR passes consolidated by majority vote. Higher = slower but more robust.",
                min_value=1),
    SettingMeta("demucs_model", "text", GROUP_MODELS,
                "Demucs source separation model (htdemucs recommended)."),
    SettingMeta("sample_rate", "number", GROUP_MODELS,
                "Audio sample rate for internal processing.", min_value=8000, step=100),

    # ── Pitch & activity ─────────────────────────────────────────────────
    SettingMeta("pitch_min_hz", "float", GROUP_PITCH,
                "Pitch detection floor in Hz (65.41 = C2).", min_value=20.0, max_value=200.0, step=0.01),
    SettingMeta("pitch_max_hz", "float", GROUP_PITCH,
                "Pitch detection ceiling in Hz (1046.5 = C6).", min_value=200.0, max_value=4000.0, step=0.1),
    SettingMeta("crepe_hop_ms", "number", GROUP_PITCH,
                "torchcrepe analysis hop length (ms). 10 ms = 100 frames/s.",
                min_value=1, max_value=100),
    SettingMeta("band_energy_min_hz", "float", GROUP_PITCH,
                "Lower bound of the amplitude-proxy band (Hz).", min_value=20.0, max_value=2000.0, step=1.0),
    SettingMeta("band_energy_max_hz", "float", GROUP_PITCH,
                "Upper bound of the amplitude-proxy band (Hz). 4000 covers harmonics/formants.",
                min_value=200.0, max_value=20000.0, step=10.0),
    SettingMeta("activity_quiet_confidence", "float", GROUP_PITCH,
                "Min CREPE confidence for a frame to count as 'quiet' in the noise-floor estimate (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_voiced_confidence", "float", GROUP_PITCH,
                "Min CREPE confidence for a frame to count as 'voiced' in the signal estimate (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_noise_percentile", "float", GROUP_PITCH,
                "Percentile of quiet frames used as the noise floor (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_noise_fallback_percentile", "float", GROUP_PITCH,
                "Percentile of all frames used as noise floor when no quiet frames exist (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_signal_percentile", "float", GROUP_PITCH,
                "Percentile of voiced frames used as the signal level (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_signal_fallback_percentile", "float", GROUP_PITCH,
                "Percentile of all frames used as signal fallback when no voiced frames exist (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("activity_threshold_ratio", "float", GROUP_PITCH,
                "Fraction of (signal - noise) added to the noise floor for the activity threshold (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),

    # ── Note segmentation ────────────────────────────────────────────────
    SettingMeta("note_min_confidence", "float", GROUP_NOTES,
                "Min CREPE confidence for vocal activity when trimming/splitting notes (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("note_fallback_confidence", "float", GROUP_NOTES,
                "Fallback confidence when no frame meets the activity threshold (0-1).",
                min_value=0.0, max_value=1.0, step=0.05),
    SettingMeta("note_dropout_gap_ms", "number", GROUP_NOTES,
                "Max gap (ms) between active frames still treated as one continuous note.",
                min_value=0, max_value=2000),
    SettingMeta("note_smooth_window", "number", GROUP_NOTES,
                "Median filter window (frames) for pitch smoothing. 5 removes vibrato.",
                min_value=1, max_value=50),
    SettingMeta("note_pitch_tolerance", "number", GROUP_NOTES,
                "Max semitone drift kept within a single note.", min_value=0, max_value=12),
    SettingMeta("note_min_duration_ms", "number", GROUP_NOTES,
                "Min duration (ms) for a pitch-change segment.", min_value=0, max_value=2000),
    SettingMeta("note_frame_step_ms", "number", GROUP_NOTES,
                "Fallback frame spacing (ms) for note end times.", min_value=1, max_value=100),
    SettingMeta("note_segment_plots", "checkbox", GROUP_NOTES,
                "Write matplotlib diagnostic plots to <temp>/note_segments_plots/ (slow, debug only)."),

    # ── Pauses ───────────────────────────────────────────────────────────
    SettingMeta("pause_min_silence_ms", "number", GROUP_PAUSES,
                "Minimum silence duration (ms) for pause detection.", min_value=0, max_value=10000),
    SettingMeta("pause_threshold_pct", "float", GROUP_PAUSES,
                "RMS energy threshold (% of 95th percentile) below which frames are silent.",
                min_value=0.1, max_value=100.0, step=0.5),

    # ── Output & debug ───────────────────────────────────────────────────
    SettingMeta("gap_lead_in_ms", "number", GROUP_OUTPUT,
                "Fallback lead-in before the first note for #GAP when no first beat is detected (ms).",
                min_value=0, max_value=10000),
    SettingMeta("linebreak_beat_offset", "number", GROUP_OUTPUT,
                "Beats before the next note to insert a line break (4 = one 4/4 measure).",
                min_value=0, max_value=32),
    SettingMeta("beat_resolution_multiplier", "number", GROUP_OUTPUT,
                "Scales the exported BPM for a finer Ultrastar beat grid (>=1).", min_value=1, max_value=16),
    SettingMeta("ffmpeg_audio_bitrate", "text", GROUP_OUTPUT,
                "FFmpeg audio bitrate for the extracted MP3 (e.g. 128k)."),
    SettingMeta("output_dir", "text", GROUP_OUTPUT,
                "Output directory for generated files (web jobs override this per job)."),
    SettingMeta("temp_dir", "text", GROUP_OUTPUT,
                "Temp directory for intermediates (web jobs override this per job)."),
    SettingMeta("debug_alignment", "checkbox", GROUP_OUTPUT,
                "Write alignment debug JSON/backtrace/pitch HTML to the temp dir."),
    SettingMeta("bpm_use_accompaniment", "checkbox", GROUP_OUTPUT,
                "Use the Demucs instrumental stem for BPM detection instead of the full mix."),
]

SETTINGS_BY_KEY: dict[str, SettingMeta] = {s.key: s for s in SETTINGS}
