"""Audio and machine-learning processing services."""

from .pauses import detect_pauses
from .pipeline import PipelineRunError, PipelineRunner, ValidationInput
from .pitch import (
    attach_pitch_to_words,
    get_midi_for_range,
    get_pitch_frames_for_range,
    hz_to_midi,
)

__all__ = [
    "attach_pitch_to_words",
    "detect_pauses",
    "get_midi_for_range",
    "get_pitch_frames_for_range",
    "hz_to_midi",
    "PipelineRunError",
    "PipelineRunner",
    "ValidationInput",
]
