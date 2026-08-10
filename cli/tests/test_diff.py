"""Tests for diff module."""

import tempfile
from pathlib import Path

import pytest
from cli.diff import diff_ultrastar
from cli.types import UltrastarMeta, UltrastarNote
from cli.ultrastar import build_ultrastar_txt


def _write_txt(path: Path, meta: UltrastarMeta, notes: list[UltrastarNote]) -> None:
    path.write_text(build_ultrastar_txt(notes, meta), encoding="utf-8")


class TestDiffUltrastar:
    def test_identical_files(self):
        meta = UltrastarMeta(title="Test", artist="Artist", mp3="t.mp3", bpm=120.0, gap=500)
        notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=4, pitch=60, syllable="hello"),
            UltrastarNote(note_type=":", start_beat=5, duration=4, pitch=62, syllable="world"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            orig = tmp / "orig.txt"
            gen = tmp / "gen.txt"
            _write_txt(orig, meta, notes)
            _write_txt(gen, meta, notes)

            report = diff_ultrastar(orig, gen)
            assert report.passed

    def test_bpm_within_tolerance(self):
        meta_orig = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        meta_gen = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=121.0, gap=500)
        notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=4, pitch=60, syllable="hi"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_txt(tmp / "o.txt", meta_orig, notes)
            _write_txt(tmp / "g.txt", meta_gen, notes)
            report = diff_ultrastar(tmp / "o.txt", tmp / "g.txt")
            assert report.bpm_pass

    def test_bpm_outside_tolerance(self):
        meta_orig = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        meta_gen = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=125.0, gap=500)
        notes = [
            UltrastarNote(note_type=":", start_beat=0, duration=4, pitch=60, syllable="hi"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_txt(tmp / "o.txt", meta_orig, notes)
            _write_txt(tmp / "g.txt", meta_gen, notes)
            report = diff_ultrastar(tmp / "o.txt", tmp / "g.txt")
            assert not report.bpm_pass

    def test_note_beat_offset_within_tolerance(self):
        meta = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        notes_orig = [
            UltrastarNote(note_type=":", start_beat=10, duration=4, pitch=60, syllable="hi"),
        ]
        notes_gen = [
            UltrastarNote(note_type=":", start_beat=13, duration=4, pitch=60, syllable="hi"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_txt(tmp / "o.txt", meta, notes_orig)
            _write_txt(tmp / "g.txt", meta, notes_gen)
            report = diff_ultrastar(tmp / "o.txt", tmp / "g.txt")
            assert report.note_diffs[0].passed

    def test_note_beat_offset_outside_tolerance(self):
        meta = UltrastarMeta(title="T", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        notes_orig = [
            UltrastarNote(note_type=":", start_beat=10, duration=4, pitch=60, syllable="hi"),
        ]
        notes_gen = [
            UltrastarNote(note_type=":", start_beat=20, duration=4, pitch=60, syllable="hi"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_txt(tmp / "o.txt", meta, notes_orig)
            _write_txt(tmp / "g.txt", meta, notes_gen)
            report = diff_ultrastar(tmp / "o.txt", tmp / "g.txt")
            assert not report.note_diffs[0].passed

    def test_different_titles(self):
        meta_orig = UltrastarMeta(title="Original", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        meta_gen = UltrastarMeta(title="Generated", artist="A", mp3="t.mp3", bpm=120.0, gap=500)
        notes = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_txt(tmp / "o.txt", meta_orig, notes)
            _write_txt(tmp / "g.txt", meta_gen, notes)
            report = diff_ultrastar(tmp / "o.txt", tmp / "g.txt")
            assert not report.title_match
