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
    config: Config | None = None,
) -> str:
    """Convert aligned syllables to Ultrastar .txt string.

    Handles:
        - Converting seconds to beats via ms_to_beats()
        - Overlap prevention: adj_start = max(start_beat, prev_end + 1)
        - Line break handling: cap last note, insert break offset beats before next

    Args:
        aligned_syllables: Syllables with timestamps and MIDI from align_lyrics().
        bpm: Detected BPM.
        gap_ms: Gap in milliseconds before first note.
        title: Song title.
        artist: Artist name.
        mp3_filename: MP3 filename for the #MP3 header.
        video_filename: Optional video filename.
        config: Pipeline configuration.

    Returns:
        Complete Ultrastar .txt string.
    """
    if config is None:
        config = Config()

    ultra_notes: list[UltrastarNote] = []
    prev_end = -1

    # Calculate gap from first syllable
    first_syl = next((s for s in aligned_syllables if not s.is_line_break and s.start > 0), None)
    gap = max(0, round(first_syl.start * 1000 - gap_ms)) if first_syl else 0

    for syl in aligned_syllables:
        if syl.is_line_break:
            next_note_beat = ms_to_beats(syl.start * 1000, bpm, gap)

            # Cap the last note of this paragraph
            last = ultra_notes[-1] if ultra_notes else None
            if last and last.note_type != "-":
                max_dur = max(1, next_note_beat - 2 - last.start_beat)
                if last.duration > max_dur:
                    last.duration = max_dur
                    prev_end = last.start_beat + max_dur

            # Line break offset beats before next note
            line_break_beat = max(
                prev_end + 1 if prev_end >= 0 else 0,
                next_note_beat - config.linebreak_beat_offset,
            )
            ultra_notes.append(UltrastarNote(
                note_type="-",
                start_beat=line_break_beat,
                duration=0,
                pitch=0,
                syllable="",
            ))
            prev_end = line_break_beat
            continue

        start_beat = ms_to_beats(syl.start * 1000, bpm, gap)
        duration = max(1, ms_to_beats(syl.end * 1000, bpm, gap) - start_beat)
        adj_start = max(start_beat, prev_end + 1) if prev_end >= 0 else start_beat

        ultra_notes.append(UltrastarNote(
            note_type=":",
            start_beat=adj_start,
            duration=duration,
            pitch=syl.midi,
            syllable=syl.syllable,
        ))
        prev_end = adj_start + duration

    meta = UltrastarMeta(
        title=title,
        artist=artist,
        mp3=mp3_filename,
        bpm=bpm,
        gap=gap,
        video=video_filename,
    )

    txt = build_ultrastar_txt(ultra_notes, meta)
    logger.info(f"Generated {len(ultra_notes)} notes ({title} - {artist})")
    return txt
