"""Note generation: convert aligned syllables to Ultrastar .txt format."""

from cli.config import Config
from cli.logging_setup import get_logger
from cli.medley import detect_medley
from cli.pipeline_types import AlignedSyllable, UltrastarMeta, UltrastarNote
from cli.ultrastar import build_ultrastar_txt, creator_value, ms_to_beats

logger = get_logger("cli.generate")


def generate_ultrastar(
    aligned_syllables: list[AlignedSyllable],
    bpm: float,
    gap_ms: int,
    title: str,
    artist: str,
    mp3_filename: str,
    video_filename: str | None = None,
    vocals_filename: str | None = None,
    instrumental_filename: str | None = None,
    cover_filename: str | None = None,
    first_beat_ms: float | None = None,
    start_sec: float | None = None,
    end_ms: int | None = None,
    lyrics_text: str | None = None,
    config: Config | None = None,
) -> str:
    """Convert aligned syllables to Ultrastar .txt string.

    Handles:
        - Converting seconds to a high-resolution Ultrastar beat grid
        - Anchoring the grid to the detected first beat via #GAP
        - Capping every duration at the next note onset
        - Ensuring every singing note has at least one beat of duration
        - Placing line breaks between adjacent singing notes
        - Stamping a #CREATOR tag with the generation timestamp

    Args:
        aligned_syllables: Syllables with timestamps and MIDI from align_lyrics().
        bpm: Detected BPM.
        gap_ms: Gap in milliseconds before first note (fallback #GAP).
        title: Song title.
        artist: Artist name.
        mp3_filename: MP3 filename for the #MP3 header.
        video_filename: Optional video filename.
        vocals_filename: Optional vocals stem filename for the #VOCALS header.
        instrumental_filename: Optional accompaniment stem filename for the
            #INSTRUMENTAL header.
        cover_filename: Optional cover image filename for the #COVER header.
        first_beat_ms: Time in milliseconds of the song's first beat
            (BpmResult.first_beat_ms). Used as #GAP so the beat grid aligns
            with the actual groove. Falls back to first note minus gap_ms.
         start_sec: Song start in seconds from the audio start (acoustic
             energy bounds); written as the #START tag.
         end_ms: Song end in milliseconds from the audio start (acoustic
             energy bounds); written as the #END tag.
         lyrics_text: Plain lyrics the notes were aligned from. When given,
             the medley (largest, most frequently repeated section) is located
             and written as #MEDLEYSTARTBEAT/#MEDLEYENDBEAT, and its first
             occurrence as #PREVIEWSTART.
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

    medley_start_beat, medley_end_beat, preview_start = _detect_medley_beats(
        ultra_notes, lyrics_text, gap, output_bpm
    )

    meta = UltrastarMeta(
        title=title,
        artist=artist,
        mp3=mp3_filename,
        bpm=output_bpm,
        gap=gap,
        video=video_filename,
        vocals=vocals_filename,
        instrumental=instrumental_filename,
        cover=cover_filename,
        creator=creator_value(),
        start=start_sec,
        end_ms=end_ms,
        medley_start_beat=medley_start_beat,
        medley_end_beat=medley_end_beat,
        preview_start=preview_start,
    )

    txt = build_ultrastar_txt(ultra_notes, meta)
    logger.info(f"Generated {len(ultra_notes)} notes ({title} - {artist})")
    return txt


def _detect_medley_beats(
    ultra_notes: list[UltrastarNote],
    lyrics_text: str | None,
    gap: int,
    output_bpm: float,
) -> tuple[int | None, int | None, float | None]:
    """Locate the medley (chorus) in the lyrics and map it onto the beat grid.

    The medley is the largest, most frequently repeated block of lyric lines
    (see ``cli.medley.detect_medley``). When found, the first occurrence of
    the block defines the section:

    - ``medley_start_beat``: start beat of the first singing note of the block
    - ``medley_end_beat``: end beat (start + duration) of its last singing note
    - ``preview_start``: seconds from the audio start where the section begins

    Args:
        ultra_notes: Notes as generated for the Ultrastar file (line breaks
            included).
        lyrics_text: The plain lyrics the notes were aligned from, or None to
            skip medley detection.
        gap: ``#GAP`` in milliseconds (offset from audio start to beat 0).
        output_bpm: Exported BPM (already scaled by the beat resolution
            multiplier), used to convert beats to milliseconds.

    Returns:
        (medley_start_beat, medley_end_beat, preview_start); all None when no
        medley is detected or the mapping is not possible.
    """
    if lyrics_text is None:
        return None, None, None

    lyric_lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
    medley = detect_medley(lyric_lines)
    if medley is None:
        logger.info("No medley (repeated chorus section) detected")
        return None, None, None

    medley_start, medley_end = medley

    line_groups: list[list[UltrastarNote]] = [[]]
    for note in ultra_notes:
        if note.note_type == "-":
            line_groups.append([])
        else:
            line_groups[-1].append(note)
    while line_groups and not line_groups[-1]:
        line_groups.pop()

    if (
        len(line_groups) != len(lyric_lines)
        or not line_groups[medley_start]
        or not line_groups[medley_end]
    ):
        logger.warning(
            f"Medley found at lyric lines {medley_start + 1}-{medley_end + 1} but the "
            f"generated note lines ({len(line_groups)}) do not match the lyric lines "
            f"({len(lyric_lines)}); omitting medley tags"
        )
        return None, None, None

    first_note = line_groups[medley_start][0]
    last_note = line_groups[medley_end][-1]
    medley_start_beat = first_note.start_beat
    medley_end_beat = last_note.start_beat + last_note.duration
    beat_ms = 60000.0 / output_bpm
    preview_start = (gap + medley_start_beat * beat_ms) / 1000.0
    logger.info(
        f"Medley detected: lines {medley_start + 1}-{medley_end + 1} of {len(lyric_lines)} "
        f"(beats {medley_start_beat}-{medley_end_beat}, preview from {preview_start:.2f} s)"
    )
    return medley_start_beat, medley_end_beat, preview_start
