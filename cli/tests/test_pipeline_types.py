"""Tests for shared pipeline data serialization."""

from cli.pipeline_types import PitchFrame, TranscribeResult, WordTimestamp


def test_transcribe_result_preserves_global_pitch_frames():
    frame = PitchFrame(time=0.1, midi=60, confidence=0.2, amplitude=0.05)
    result = TranscribeResult(
        words=[WordTimestamp("hello", 0.1, 0.5, 60)],
        language="en",
        vocals_path="vocals.mp3",
        accompaniment_path="accompaniment.mp3",
        pauses=[],
        pitch_frames=[frame],
    )

    restored = TranscribeResult.from_dict(result.to_dict())

    assert restored.pitch_frames == [frame]


def test_old_transcribe_result_recovers_word_pitch_frames():
    frame = PitchFrame(time=0.1, midi=60, confidence=0.8, amplitude=0.4)
    data = {
        "words": [WordTimestamp("hello", 0.1, 0.5, 60, [frame]).to_dict()],
        "language": "en",
        "vocals_path": "vocals.mp3",
        "accompaniment_path": "accompaniment.mp3",
        "pauses": [],
    }

    restored = TranscribeResult.from_dict(data)

    assert restored.pitch_frames == [frame]
