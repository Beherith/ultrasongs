"""Self-contained HTML reporting for pipeline inputs, intermediates, and output."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ultrasongs.domain.scoring import SimilarityResult
from ultrasongs.domain.ultrastar import UltrastarNote, UltrastarSong, beat_to_ms, beats_to_ms
from ultrasongs.domain.validation import ValidationOutcome

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def pitch_name(pitch: int) -> str:
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def build_pipeline_report(
    *,
    candidate: UltrastarSong | None = None,
    reference: UltrastarSong | None = None,
    transcription: Mapping[str, Any] | None = None,
    similarity: SimilarityResult | None = None,
    title: str | None = None,
    effective_config: Mapping[str, Any] | None = None,
    validation_outcome: Mapping[str, Any] | ValidationOutcome | None = None,
    report_options: Mapping[str, Any] | None = None,
) -> str:
    """Build one offline report from any available pipeline artifacts."""

    display_title = (
        title or _song_title(candidate) or _song_title(reference) or "UltraSongs pipeline report"
    )
    words = _valid_words(transcription)
    include_pitch_frames = bool(
        (report_options or {}).get("include_pitch_frames", True)
    )
    include_pauses = bool((report_options or {}).get("include_pauses", True))
    report_words = (
        words
        if include_pitch_frames
        else [{key: value for key, value in word.items() if key != "pitchFrames"} for word in words]
    )
    frames = (
        [frame for word in report_words for frame in _valid_frames(word)]
        if include_pitch_frames
        else []
    )
    pauses = _valid_pauses(transcription) if include_pauses else []
    sections = [
        _summary_section(candidate, reference, transcription, similarity, words, frames, pauses),
        _validation_section(validation_outcome),
        _score_section(similarity),
        _transcription_section(transcription, report_words, frames, pauses),
        _song_section("Final candidate", "candidate", candidate),
        _song_section("Reference song", "reference", reference),
        _config_section(effective_config),
    ]
    score_json = _safe_json(similarity.to_dict() if similarity is not None else None)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(display_title)} - pipeline report</title>
  <style>{_STYLES}</style>
</head>
<body>
<main>
  <header class="hero">
    <p class="eyebrow">UltraSongs validation</p>
    <h1>{_escape(display_title)}</h1>
    <p>Intermediate pitch and transcription data, generated notes, reference notes,
    and validation metrics.</p>
  </header>
  {"".join(sections)}
</main>
<script id="similarity-data" type="application/json">{score_json}</script>
</body>
</html>
"""


def _validation_section(
    outcome: Mapping[str, Any] | ValidationOutcome | None,
) -> str:
    if outcome is None:
        return ""
    if hasattr(outcome, "to_dict"):
        outcome = outcome.to_dict()
    if not isinstance(outcome, Mapping):
        return ""
    passed = bool(outcome.get("passed"))
    failures = outcome.get("failures") or ()
    status = "PASSED" if passed else "FAILED"
    css_class = "validation-pass" if passed else "validation-fail"
    details = ""
    if failures:
        items = "".join(f"<li>{_escape(item)}</li>" for item in failures)
        details = f"<ul>{items}</ul>"
    return (
        f'<section id="validation" class="{css_class}"><h2>Validation {status}</h2>'
        f"{details}</section>"
    )


def write_pipeline_report(path: str | Path, **kwargs: Any) -> Path:
    destination = Path(path)
    destination.write_text(build_pipeline_report(**kwargs), encoding="utf-8", newline="\n")
    return destination


def _summary_section(candidate, reference, transcription, similarity, words, frames, pauses) -> str:
    language = transcription.get("language") if transcription else None
    cards = [
        ("Candidate notes", str(len(candidate.notes)) if candidate else "not supplied"),
        ("Reference notes", str(len(reference.notes)) if reference else "not supplied"),
        ("Transcribed words", str(len(words))),
        ("Pitch frames", str(len(frames))),
        ("Pauses", str(len(pauses))),
        ("Language", str(language) if language else "unknown"),
        ("Matched notes", str(similarity.matched_notes) if similarity else "not scored"),
    ]
    body = "".join(
        f'<div class="card"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in cards
    )
    return f'<section id="summary"><h2>Summary</h2><div class="cards">{body}</div></section>'


def _score_section(result: SimilarityResult | None) -> str:
    if result is None:
        return _empty_section(
            "similarity", "Similarity score", "No reference comparison was supplied."
        )
    rows = [
        (
            "Matched",
            f"{result.matched_notes} / {min(result.reference_notes, result.candidate_notes)}",
        ),
        ("Matched ratio", _percent(result.matched_ratio)),
        ("Reference coverage", _percent(result.reference_coverage)),
        ("Candidate coverage", _percent(result.candidate_coverage)),
        ("Timing RMSE", _metric(result.timing_rmse_ms, "ms")),
        (
            "Timing median / max",
            _pair_metric(result.timing_median_error_ms, result.timing_max_error_ms, "ms"),
        ),
        ("Duration RMSE", _metric(result.duration_rmse_ms, "ms")),
        (
            "Duration median / max",
            _pair_metric(result.duration_median_error_ms, result.duration_max_error_ms, "ms"),
        ),
        ("Pitch distance", _metric(result.pitch_distance_semitones, "semitones")),
        (
            "Pitch median / max",
            _pair_metric(
                result.pitch_median_distance_semitones, result.pitch_max_distance_semitones, "st"
            ),
        ),
    ]
    warning = (
        ""
        if result.has_matches
        else '<p class="warning">No notes matched; error metrics are intentionally unavailable.</p>'
    )
    score_table = _table(("Metric", "Value"), rows)
    return f'<section id="similarity"><h2>Similarity score</h2>{warning}{score_table}</section>'


def _transcription_section(transcription, words, frames, pauses) -> str:
    if transcription is None:
        return _empty_section(
            "intermediate",
            "Intermediate transcription and pitch",
            "No transcription artifact was supplied.",
        )
    groups = _split_word_groups(words)
    charts = "".join(
        f'<article class="plot"><h3>Segment {index + 1}</h3>{_pitch_svg(group)}</article>'
        for index, group in enumerate(groups)
    )
    word_rows = [
        (
            str(word.get("word", "")),
            _number(word.get("start"), "s"),
            _number(word.get("end"), "s"),
            str(word.get("midi", "-")),
            pitch_name(int(word["midi"])) if _is_number(word.get("midi")) else "-",
            str(len(_valid_frames(word))),
        )
        for word in words
    ]
    pause_rows = [(_number(pause["start"], "s"), _number(pause["end"], "s")) for pause in pauses]
    word_table = _table(("Word", "Start", "End", "MIDI", "Note", "Frames"), word_rows)
    pause_table = (
        _table(("Start", "End"), pause_rows) if pause_rows else '<p class="empty">No pauses.</p>'
    )
    return f"""<section id="intermediate">
<h2>Intermediate transcription and pitch</h2>
<p class="section-note">{len(words)} words, {len(frames)} pitch frames,
{len(pauses)} detected pauses.</p>
{charts or '<p class="empty">No plottable pitch frames.</p>'}
<details><summary>Word timing and dominant pitch</summary>{word_table}</details>
<details><summary>Detected pauses</summary>{pause_table}</details>
</section>"""


def _song_section(heading: str, section_id: str, song: UltrastarSong | None) -> str:
    if song is None:
        return _empty_section(section_id, heading, f"No {heading.lower()} was supplied.")
    meta = song.metadata
    metadata_rows = [
        ("Title", meta.title),
        ("Artist", meta.artist),
        ("MP3", meta.mp3),
        ("BPM", f"{meta.bpm:g}"),
        ("GAP", f"{meta.gap_ms:g} ms"),
    ] + [(key, value) for key, value in meta.extras.items()]
    verse_blocks = []
    for index, verse in enumerate(song.verses()):
        lyric = "".join(note.lyric for note in verse)
        verse_blocks.append(
            f'<article class="plot"><h3>Line {index + 1}: {_escape(lyric)}</h3>'
            f"{_song_svg(song, verse)}</article>"
        )
    note_rows = [
        (
            note.note_type.value,
            f"{note.start_beat:g}",
            f"{beat_to_ms(note.start_beat, meta.bpm, meta.gap_ms):.1f} ms",
            f"{note.duration_beats:g}",
            f"{beats_to_ms(note.duration_beats, meta.bpm):.1f} ms",
            str(note.pitch),
            pitch_name(note.pitch),
            note.lyric,
        )
        for note in song.notes
    ]
    metadata_table = _table(("Header", "Value"), metadata_rows)
    note_headers = ("Type", "Beat", "Start", "Length", "Duration", "MIDI", "Note", "Lyric")
    note_table = _table(note_headers, note_rows)
    line_count = len(tuple(song.verses()))
    return f"""<section id="{section_id}"><h2>{_escape(heading)}</h2>
<div class="two-column">{metadata_table}<div class="card">
<span>Lines</span><strong>{line_count}</strong>
<span>Notes</span><strong>{len(song.notes)}</strong></div></div>
{"".join(verse_blocks) or '<p class="empty">No notes.</p>'}
<details><summary>All notes</summary>{note_table}</details>
</section>"""


def _config_section(config: Mapping[str, Any] | None) -> str:
    if config is None:
        return _empty_section(
            "configuration", "Effective configuration", "No configuration snapshot was supplied."
        )
    config_json = _escape(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    return (
        '<section id="configuration"><h2>Effective configuration</h2>'
        f"<pre>{config_json}</pre></section>"
    )


def _pitch_svg(words: Sequence[Mapping[str, Any]], width: int = 1000, height: int = 250) -> str:
    frames = [(word, frame) for word in words for frame in _valid_frames(word)]
    if not frames:
        return '<p class="empty">No pitch frames in this segment.</p>'
    times = [float(frame["time"]) for _, frame in frames]
    pitches = [int(frame["midi"]) for _, frame in frames]
    t_min, t_max = min(times), max(times)
    p_min, p_max = min(pitches), max(pitches)

    def x(value: float) -> float:
        return 55 + (float(value) - t_min) / max(t_max - t_min, 0.01) * (width - 75)

    def y(value: int) -> float:
        return 18 + (p_max - int(value)) / max(p_max - p_min, 1) * (height - 50)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pitch frames over time">']
    for pitch in range(p_min, p_max + 1):
        cy = y(pitch)
        parts.append(
            f'<line class="grid" x1="55" y1="{cy:.1f}" '
            f'x2="{width - 20}" y2="{cy:.1f}"/>'
            f'<text class="axis" x="50" y="{cy + 3:.1f}" '
            f'text-anchor="end">{pitch_name(pitch)}</text>'
        )
    for word, frame in frames:
        raw_confidence = frame.get("confidence", 0)
        confidence = _clamp(float(raw_confidence) if _is_number(raw_confidence) else 0)
        hue = confidence * 120
        label = (
            f"{word.get('word', '')} | {pitch_name(int(frame['midi']))} | "
            f"{float(frame['time']):.2f}s | confidence {confidence:.2f}"
        )
        parts.append(
            f'<circle cx="{x(frame["time"]):.1f}" '
            f'cy="{y(frame["midi"]):.1f}" r="2.2" '
            f'fill="hsl({hue:.0f} 80% 55%)" '
            f'opacity="{max(0.25, confidence):.2f}">'
            f"<title>{_escape(label)}</title></circle>"
        )
    parts.append(
        f'<text class="axis" x="55" y="{height - 6}">{t_min:.2f}s</text>'
        f'<text class="axis" x="{width - 20}" y="{height - 6}" '
        f'text-anchor="end">{t_max:.2f}s</text></svg>'
    )
    return "".join(parts)


def _song_svg(
    song: UltrastarSong, notes: Sequence[UltrastarNote], width: int = 1000, height: int = 210
) -> str:
    starts = [
        beat_to_ms(note.start_beat, song.metadata.bpm, song.metadata.gap_ms) for note in notes
    ]
    ends = [
        start + beats_to_ms(note.duration_beats, song.metadata.bpm)
        for start, note in zip(starts, notes, strict=True)
    ]
    pitches = [note.pitch for note in notes]
    t_min, t_max = min(starts), max(ends)
    p_min, p_max = min(pitches), max(pitches)

    def x(value: float) -> float:
        return 55 + (value - t_min) / max(t_max - t_min, 1) * (width - 75)

    def y(value: int) -> float:
        return 18 + (p_max - value) / max(p_max - p_min, 1) * (height - 50)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="UltraStar notes over time">'
    ]
    for pitch in range(p_min, p_max + 1):
        cy = y(pitch)
        parts.append(
            f'<line class="grid" x1="55" y1="{cy:.1f}" '
            f'x2="{width - 20}" y2="{cy:.1f}"/>'
            f'<text class="axis" x="50" y="{cy + 3:.1f}" '
            f'text-anchor="end">{pitch_name(pitch)}</text>'
        )
    for note, start, end in zip(notes, starts, ends, strict=True):
        left, right, cy = x(start), x(end), y(note.pitch)
        color = "#fb7185" if note.chorus else "#38bdf8"
        label = (
            f"{note.lyric} | {pitch_name(note.pitch)} | {start / 1000:.2f}s | {end - start:.1f}ms"
        )
        parts.append(
            f'<rect x="{left:.1f}" y="{cy - 7:.1f}" '
            f'width="{max(right - left, 2):.1f}" height="14" '
            f'rx="3" fill="{color}"><title>{_escape(label)}</title></rect>'
        )
    parts.append(
        f'<text class="axis" x="55" y="{height - 6}">{t_min / 1000:.2f}s</text>'
        f'<text class="axis" x="{width - 20}" y="{height - 6}" '
        f'text-anchor="end">{t_max / 1000:.2f}s</text></svg>'
    )
    return "".join(parts)


def _valid_words(transcription: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not transcription or not isinstance(transcription.get("words"), list):
        return []
    return [word for word in transcription["words"] if isinstance(word, Mapping)]


def _valid_frames(word: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = word.get("pitchFrames", word.get("pitch_frames", []))
    if not isinstance(raw, list):
        return []
    return [
        frame
        for frame in raw
        if isinstance(frame, Mapping)
        and _is_number(frame.get("time"))
        and _is_number(frame.get("midi"))
    ]


def _valid_pauses(transcription: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    raw = transcription.get("pauses", []) if transcription else []
    if not isinstance(raw, list):
        return []
    return [
        pause
        for pause in raw
        if isinstance(pause, Mapping)
        and _is_number(pause.get("start"))
        and _is_number(pause.get("end"))
    ]


def _split_word_groups(
    words: Sequence[Mapping[str, Any]], gap: float = 2.0, max_words: int = 36
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for word in words:
        start = float(word.get("start", 0)) if _is_number(word.get("start")) else 0
        last_end = (
            float(current[-1].get("end", start))
            if current and _is_number(current[-1].get("end"))
            else start
        )
        if current and (start - last_end > gap or len(current) >= max_words):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _empty_section(section_id: str, heading: str, message: str) -> str:
    return (
        f'<section id="{section_id}"><h2>{_escape(heading)}</h2>'
        f'<p class="empty">{_escape(message)}</p></section>'
    )


def _song_title(song: UltrastarSong | None) -> str | None:
    return song.metadata.title if song and song.metadata.title else None


def _metric(value: float | None, unit: str) -> str:
    return "unavailable" if value is None else f"{value:.2f} {unit}"


def _pair_metric(median: float | None, maximum: float | None, unit: str) -> str:
    return (
        "unavailable"
        if median is None or maximum is None
        else f"{median:.2f} / {maximum:.2f} {unit}"
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: Any, unit: str = "") -> str:
    return f"{float(value):.2f}{unit}" if _is_number(value) else "-"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


_STYLES = """
:root { color-scheme:dark; --bg:#08111f; --panel:#111d2f; --line:#263750;
  --text:#dbeafe; --muted:#8da2bd; --accent:#38bdf8; --hot:#fb7185; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }
main { width:min(1440px,96vw); margin:auto; padding:28px 0 60px; }
.hero { padding:32px; background:linear-gradient(135deg,#10243e,#271936);
  border:1px solid var(--line); border-radius:18px; }
.hero h1 { font-size:clamp(28px,5vw,54px); margin:0; }
.hero p { color:var(--muted); max-width:850px; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; color:var(--accent)!important;
  font-weight:700; }
section { margin-top:22px; padding:22px; background:var(--panel);
  border:1px solid var(--line); border-radius:14px; }
h2 { margin:0 0 16px; font-size:22px; }
h3 { font-size:14px; color:var(--muted); font-weight:600; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; }
.card { display:grid; gap:4px; padding:13px; background:#0b1627;
  border:1px solid var(--line); border-radius:10px; }
.card span { color:var(--muted); }
.card strong { font-size:21px; }
.two-column { display:grid; grid-template-columns:minmax(300px,2fr) minmax(140px,1fr);
  gap:14px; }
.plot { margin-top:14px; padding:12px; background:#091424; border:1px solid var(--line);
  border-radius:10px; overflow:hidden; }
.plot svg { width:100%; height:auto; display:block; }
.grid { stroke:#263750; stroke-width:.7; }
.axis { fill:#8da2bd; font:10px ui-monospace,monospace; }
.table-wrap { overflow:auto; }
table { width:100%; border-collapse:collapse; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left;
  white-space:nowrap; }
th { color:var(--accent); font-size:12px; text-transform:uppercase; }
td:last-child { white-space:normal; }
.empty, .section-note { color:var(--muted); }
.warning { padding:10px; border-left:4px solid var(--hot); background:#2a1520; }
.validation-pass { border-color:#22c55e; }
.validation-pass h2 { color:#86efac; }
.validation-fail { border-color:var(--hot); }
.validation-fail h2 { color:#fda4af; }
details { margin-top:16px; }
summary { cursor:pointer; color:var(--accent); }
pre { padding:14px; background:#091424; overflow:auto; border-radius:8px; }
@media(max-width:700px) { .two-column { grid-template-columns:1fr; }
  .hero, section { padding:16px; } }
"""
