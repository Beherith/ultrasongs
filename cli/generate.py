"""Note generation: convert aligned syllables to Ultrastar .txt format."""

from cli.config import Config
from cli.logging_setup import get_logger
from cli.pipeline_types import AlignedSyllable, UltrastarMeta, UltrastarNote
from cli.ultrastar import build_ultrastar_txt, ms_to_beats

logger = get_logger("cli.generate")


def generate_ultrastar(
    aligned_syllables: list[AlignedSyllable],
    bpm: float,
    gap_ms: int,
    title: str,
    artist: str,
    mp3_filename: str,
    video_filename: str | None = None,
    first_beat_ms: float | None = None,
    config: Config | None = None,
) -> str:
    """Convert aligned syllables to Ultrastar .txt string.

    Handles:
        - Converting seconds to a high-resolution Ultrastar beat grid
        - Anchoring the grid to the detected first beat via #GAP
        - Capping every duration at the next note onset
        - Ensuring every singing note has at least one beat of duration
        - Placing line breaks between adjacent singing notes

    Args:
        aligned_syllables: Syllables with timestamps and MIDI from align_lyrics().
        bpm: Detected BPM.
        gap_ms: Gap in milliseconds before first note (fallback #GAP).
        title: Song title.
        artist: Artist name.
        mp3_filename: MP3 filename for the #MP3 header.
        video_filename: Optional video filename.
        first_beat_ms: Time in milliseconds of the song's first beat
            (BpmResult.first_beat_ms). Used as #GAP so the beat grid aligns
            with the actual groove. Falls back to first note minus gap_ms.
        config: Pipeline configuration.

    Returns:
        Complete Ultrastar .txt string.
    """
    if config is None:
        config = Config()

    ultra_notes: list[UltrastarNote] = []

    # Anchor the beat grid: #GAP is the offset from the audio start to the
    # first beat, so a note sung on the downbeat lands on an integer beat.
    if first_beat_ms is not None:
        gap = max(0, round(first_beat_ms))
    else:
        first_syl = next((s for s in aligned_syllables if not s.is_line_break), None)
        gap = max(0, round(first_syl.start * 1000 - gap_ms)) if first_syl else 0
    output_bpm = bpm * config.beat_resolution_multiplier

    # Quantize every onset once. Notes need distinct grid positions because an
    # Ultrastar note must have a duration of at least one beat.
    starts: list[int | None] = []
    previous_start: int | None = None
    for syl in aligned_syllables:
        if syl.is_line_break:
            starts.append(None)
            continue
        start = ms_to_beats(syl.start * 1000, output_bpm, gap)
        if previous_start is not None:
            start = max(start, previous_start + 1)
        starts.append(start)
        previous_start = start

    # Nearest following singing-note onset for duration and line-break caps.
    next_starts: list[int | None] = [None] * len(aligned_syllables)
    next_start: int | None = None
    for i in range(len(aligned_syllables) - 1, -1, -1):
        next_starts[i] = next_start
        if starts[i] is not None:
            next_start = starts[i]

    offset = config.linebreak_beat_offset * config.beat_resolution_multiplier

    for i, syl in enumerate(aligned_syllables):
        if syl.is_line_break:
            next_note_beat = next_starts[i]
            if next_note_beat is None:
                continue

            last = ultra_notes[-1] if ultra_notes else None
            previous_end = last.start_beat + last.duration if last and last.note_type == ":" else 0
            target = max(0, next_note_beat - offset)
            line_break_beat = min(next_note_beat, max(target, previous_end))

            ultra_notes.append(UltrastarNote(
                note_type="-",
                start_beat=line_break_beat,
                duration=0,
                pitch=0,
                syllable="",
            ))
            continue

        start_beat = starts[i]
        assert start_beat is not None
        end_beat = max(start_beat + 1, ms_to_beats(syl.end * 1000, output_bpm, gap))
        if next_starts[i] is not None:
            end_beat = min(end_beat, next_starts[i])
        duration = end_beat - start_beat

        ultra_notes.append(UltrastarNote(
            note_type=":",
            start_beat=start_beat,
            duration=duration,
            pitch=syl.midi,
            syllable=syl.syllable or "~",
        ))

    meta = UltrastarMeta(
        title=title,
        artist=artist,
        mp3=mp3_filename,
        bpm=output_bpm,
        gap=gap,
        video=video_filename,
    )

    txt = build_ultrastar_txt(ultra_notes, meta)
    logger.info(f"Generated {len(ultra_notes)} notes ({title} - {artist})")
    return txt
