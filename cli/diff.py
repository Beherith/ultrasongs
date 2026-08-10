"""Diff tool: compare two Ultrastar .txt files."""

from dataclasses import dataclass, field
from pathlib import Path

from cli.logging_setup import get_logger
from cli.pipeline_types import UltrastarMeta, UltrastarNote
from cli.ultrastar import parse_ultrastar_txt

logger = get_logger("cli.diff")

# Tolerances
_BPM_TOLERANCE = 2.0       # BPM
_BEAT_TOLERANCE = 4        # beats
_DURATION_TOLERANCE = 4    # beats
_PITCH_TOLERANCE = 3       # semitones


@dataclass
class NoteDiff:
    idx: int
    orig_syllable: str
    gen_syllable: str
    syllable_match: bool
    orig_beat: int
    gen_beat: int
    beat_offset: int
    orig_duration: int
    gen_duration: int
    duration_diff: int
    orig_pitch: int
    gen_pitch: int
    pitch_diff: int
    passed: bool


@dataclass
class DiffReport:
    title_match: bool = False
    artist_match: bool = False
    bpm_orig: float = 0.0
    bpm_gen: float = 0.0
    bpm_diff: float = 0.0
    bpm_pass: bool = False
    gap_orig: int = 0
    gap_gen: int = 0
    gap_pass: bool = False
    orig_note_count: int = 0
    gen_note_count: int = 0
    note_count_pass: bool = False
    note_diffs: list[NoteDiff] = field(default_factory=list)
    linebreak_orig_count: int = 0
    linebreak_gen_count: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.title_match
            and self.artist_match
            and self.bpm_pass
            and self.gap_pass
            and self.note_count_pass
            and all(nd.passed for nd in self.note_diffs)
        )

    def print(self) -> None:
        """Print human-readable report to stdout."""
        print("=" * 60)
        print("ULTRASTAR DIFF REPORT")
        print("=" * 60)

        print(f"\nTitle:    {'PASS' if self.title_match else 'FAIL'}")
        print(f"Artist:   {'PASS' if self.artist_match else 'FAIL'}")
        print(f"BPM:      {self.bpm_orig:.1f} vs {self.bpm_gen:.1f} "
              f"(diff={self.bpm_diff:.1f}) {'PASS' if self.bpm_pass else 'FAIL'}")
        print(f"GAP:      {self.gap_orig} vs {self.gap_gen} "
              f"{'PASS' if self.gap_pass else 'FAIL'}")
        print(f"Notes:    {self.orig_note_count} vs {self.gen_note_count} "
              f"{'PASS' if self.note_count_pass else 'FAIL'}")
        print(f"Line breaks: {self.linebreak_orig_count} vs {self.linebreak_gen_count}")

        failed = [nd for nd in self.note_diffs if not nd.passed]
        if failed:
            print(f"\n{len(failed)} failed notes (showing first 20):")
            for nd in failed[:20]:
                print(
                    f"  [{nd.idx}] '{nd.orig_syllable}' vs '{nd.gen_syllable}' "
                    f"beat={nd.orig_beat}->{nd.gen_beat}({nd.beat_offset:+d}) "
                    f"dur={nd.orig_duration}->{nd.gen_duration}({nd.duration_diff:+d}) "
                    f"pitch={nd.orig_pitch}->{nd.gen_pitch}({nd.pitch_diff:+d})"
                )
        else:
            print(f"\nAll {len(self.note_diffs)} notes within tolerance.")

        print(f"\nOverall: {'PASS' if self.passed else 'FAIL'}")
        print("=" * 60)


def diff_ultrastar(original_path: Path, generated_path: Path) -> DiffReport:
    """Compare two Ultrastar .txt files.

    Args:
        original_path: Reference .txt file.
        generated_path: Generated .txt file to compare.

    Returns:
        DiffReport with pass/fail summary and per-note deltas.
    """
    orig_meta, orig_notes = parse_ultrastar_txt(original_path.read_text(encoding="utf-8"))
    gen_meta, gen_notes = parse_ultrastar_txt(generated_path.read_text(encoding="utf-8"))

    report = DiffReport()

    # Meta comparison
    report.title_match = orig_meta.title == gen_meta.title
    report.artist_match = orig_meta.artist == gen_meta.artist
    report.bpm_orig = orig_meta.bpm
    report.bpm_gen = gen_meta.bpm
    report.bpm_diff = abs(orig_meta.bpm - gen_meta.bpm)
    report.bpm_pass = report.bpm_diff <= _BPM_TOLERANCE
    report.gap_orig = orig_meta.gap
    report.gap_gen = gen_meta.gap
    report.gap_pass = orig_meta.gap == gen_meta.gap

    # Note counts
    orig_singing = [n for n in orig_notes if n.note_type != "-"]
    gen_singing = [n for n in gen_notes if n.note_type != "-"]
    orig_linebreaks = [n for n in orig_notes if n.note_type == "-"]
    gen_linebreaks = [n for n in gen_notes if n.note_type == "-"]

    report.orig_note_count = len(orig_singing)
    report.gen_note_count = len(gen_singing)
    report.note_count_pass = len(orig_singing) == len(gen_singing)
    report.linebreak_orig_count = len(orig_linebreaks)
    report.linebreak_gen_count = len(gen_linebreaks)

    # Per-note comparison (align by index)
    min_count = min(len(orig_singing), len(gen_singing))
    for i in range(min_count):
        o = orig_singing[i]
        g = gen_singing[i]

        beat_offset = g.start_beat - o.start_beat
        duration_diff = g.duration - o.duration
        pitch_diff = g.pitch - o.pitch

        nd = NoteDiff(
            idx=i,
            orig_syllable=o.syllable,
            gen_syllable=g.syllable,
            syllable_match=o.syllable == g.syllable,
            orig_beat=o.start_beat,
            gen_beat=g.start_beat,
            beat_offset=beat_offset,
            orig_duration=o.duration,
            gen_duration=g.duration,
            duration_diff=duration_diff,
            orig_pitch=o.pitch,
            gen_pitch=g.pitch,
            pitch_diff=pitch_diff,
            passed=(
                abs(beat_offset) <= _BEAT_TOLERANCE
                and abs(duration_diff) <= _DURATION_TOLERANCE
                and abs(pitch_diff) <= _PITCH_TOLERANCE
            ),
        )
        report.note_diffs.append(nd)

    logger.info(f"Diff complete: {min_count} notes compared, {sum(1 for nd in report.note_diffs if not nd.passed)} failed")
    return report
