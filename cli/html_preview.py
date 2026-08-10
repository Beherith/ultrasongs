"""Parse an UltraStar Deluxe .txt song file and render it as an HTML page with verse visualizations.

Supports optional pitch detection JSON overlay to show confidence-colored pitch frames
alongside the Ultrastar note bars.
"""

import json
import sys
from pathlib import Path

import html as html_module

from cli.logging_setup import get_logger

logger = get_logger("cli.html_preview")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_note(pitch: int) -> str:
    octave = pitch // 12 - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def beats_to_ms(beats: float, bpm: float) -> float:
    """Convert beat count to milliseconds. Each beat unit is a 16th note."""
    return beats * (60000.0 / bpm / 4)


def ms_to_sec(ms: float) -> str:
    """Format milliseconds as seconds (e.g. 32.65s)."""
    return f"{ms / 1000:.2f}s"


def sec_to_str(s: float) -> str:
    return f"{s:.2f}s"


def confidence_color(conf: float) -> str:
    """Map confidence 0..1 to a color: red (low) -> yellow (mid) -> green (high)."""
    if conf < 0.5:
        r, g = 255, int(conf * 2 * 200)
    else:
        r, g = int(255 - (conf - 0.5) * 2 * 255), 200 + int((conf - 0.5) * 2 * 55)
    return f"rgb({r},{g},0)"


def numpy_range(start: float, stop: float, step: float):
    """Simple range generator for floats."""
    while start <= stop:
        yield round(start, 10)
        start += step


def split_verses(words: list[dict], gap_threshold: float = 2.0) -> list[list[dict]]:
    """Split words into verses based on time gaps (seconds)."""
    verses = []
    current = []
    for w in words:
        if current and w["start"] - (current[-1]["end"]) > gap_threshold:
            verses.append(current)
            current = []
        current.append(w)
    if current:
        verses.append(current)
    return verses


# ── Ultrastar parsing ──────────────────────────────────────────────────────────


def parse_ultrastar(filepath: Path) -> dict:
    """Parse an Ultrastar .txt file into metadata, notes, and verses."""
    metadata = {}
    text = filepath.read_text(encoding="utf-8")

    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            metadata[key.strip()] = value.strip()

    bpm = float(metadata.get("BPM", 120))
    gap = float(metadata.get("GAP", 0))

    def convert_note(parts: list[str], is_chorus: bool) -> dict:
        return {
            "start": gap + beats_to_ms(float(parts[0]), bpm),
            "duration": beats_to_ms(float(parts[1]), bpm),
            "pitch": int(parts[2]),
            "lyric": parts[3],
            "chorus": is_chorus,
        }

    verses: list[list[dict]] = []
    current_verse: list[dict] = []

    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(":") or line.startswith("*"):
            parts = line[1:].split(None, 3)
            if len(parts) >= 4:
                current_verse.append(convert_note(parts, line.startswith("*")))
        elif line.startswith("-"):
            if current_verse:
                verses.append(current_verse)
                current_verse = []

    if current_verse:
        verses.append(current_verse)

    all_notes = [n for v in verses for n in v]
    return {"metadata": metadata, "notes": all_notes, "verses": verses}


# ── Pitch detection loading ───────────────────────────────────────────────────


def load_pitch_json(filepath: str | Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def match_pitch_frames_to_notes(
    notes: list[dict],
    pitch_words: list[dict],
) -> list[dict]:
    """Attach pitch frames from detection JSON to Ultrastar notes by time overlap.

    Returns a new list of notes with an optional ``pitchFrames`` key attached.
    """
    enriched = []
    for note in notes:
        note_copy = dict(note)
        start_s = note["start"] / 1000.0
        end_s = (note["start"] + note["duration"]) / 1000.0
        frames = []
        for pw in pitch_words:
            pw_start = pw.get("start", 0)
            pw_end = pw.get("end", 0)
            if pw_end <= start_s or pw_start >= end_s:
                continue
            for pf in pw.get("pitchFrames", []):
                ft = pf["time"]
                if start_s <= ft <= end_s:
                    frames.append(pf)
        if frames:
            note_copy["pitchFrames"] = frames
        enriched.append(note_copy)
    return enriched


# ── SVG building ──────────────────────────────────────────────────────────────


def build_verse_svg(
    verse: list[dict],
    width: int,
    height: int,
    *,
    show_pitch_frames: bool = False,
) -> str:
    """Build an SVG for a verse: X=time (ms), Y=pitch (inverted).

    When ``show_pitch_frames`` is True, confidence-colored pitch detection dots
    are overlaid on top of the note bars.
    """
    if not verse:
        return ""

    starts = [n["start"] for n in verse]
    ends = [n["start"] + n["duration"] for n in verse]
    pitches = [n["pitch"] for n in verse]

    t_min = min(starts)
    t_max = max(ends)
    p_min = min(pitches)
    p_max = max(pitches)

    # Collect pitch frames for range expansion
    all_frames = []
    for n in verse:
        all_frames.extend(n.get("pitchFrames", []))

    if all_frames:
        frame_pitches = [f["midi"] for f in all_frames]
        p_min = min(p_min, *frame_pitches)
        p_max = max(p_max, *frame_pitches)

    pad_left = 50
    pad_right = 16
    pad_top = 16
    pad_bottom = 24
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    t_range = max(t_max - t_min, 1)
    p_range = max(p_max - p_min, 1)

    def tx(ms: float) -> float:
        return pad_left + (ms - t_min) / t_range * plot_w

    def ty(pitch: int) -> float:
        return pad_top + (p_max - pitch) / p_range * plot_h

    def tx_from_sec(s: float) -> float:
        return tx(s * 1000.0)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;height:auto;display:block;">'
    )

    # Pitch grid
    for p in range(p_min, p_max + 1):
        y = ty(p)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="#2a2a4a" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{pad_left - 4}" y="{y + 3:.1f}" text-anchor="end" fill="#666" '
            f'font-size="8" font-family="monospace">{p}</text>'
        )

    # Time axis labels (ms)
    t_span = t_range
    if t_span > 20000:
        step = 5000
    elif t_span > 8000:
        step = 2000
    elif t_span > 3000:
        step = 1000
    else:
        step = 500

    for t in range(int(t_min // step) * step, int(t_max // step) * step + step, step):
        if t < t_min or t > t_max:
            continue
        x = tx(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height - pad_bottom}" '
            f'stroke="#1e1e3a" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 4}" text-anchor="middle" fill="#666" '
            f'font-size="8" font-family="monospace">{t / 1000:.1f}s</text>'
        )

    # Word-block background shading (from pitch frames)
    if show_pitch_frames:
        # Compute per-note time spans in seconds for background
        for n in verse:
            x1 = tx(n["start"])
            x2 = tx(n["start"] + n["duration"])
            parts.append(
                f'<rect x="{x1:.1f}" y="{pad_top}" width="{max(x2 - x1, 1):.1f}" '
                f'height="{plot_h}" fill="#0a0a1a" opacity="0.3" rx="2"/>'
            )

    # Note bars
    for n in verse:
        x = tx(n["start"])
        w = tx(n["start"] + n["duration"]) - x
        y = ty(n["pitch"])
        bar_h = max(plot_h / (p_range + 1) * 0.7, 8)
        fill = "#e94560" if n.get("chorus") else "#0f3460"
        stroke = "#ff6b81" if n.get("chorus") else "#4fc3f7"
        text_fill = "#fff" if n.get("chorus") else "#cde"
        rx = x + w / 2
        ry = y
        bw = max(w, 2)
        parts.append(
            f'<rect x="{x:.1f}" y="{(y - bar_h / 2):.1f}" width="{bw:.1f}" '
            f'height="{bar_h:.1f}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="0.8" '
            f'title="{html_module.escape(n["lyric"])} ({pitch_to_note(n["pitch"])}) '
            f'{ms_to_sec(n["start"])} +{n["duration"]:.0f}ms"/>'
        )
        lyric = html_module.escape(n["lyric"])
        fs = max(7, min(12, bw / (len(lyric) * 0.55)))
        parts.append(
            f'<text x="{rx:.1f}" y="{ry + fs * 0.35:.1f}" text-anchor="middle" '
            f'fill="{text_fill}" font-size="{fs:.1f}" font-family="sans-serif" '
            f'dominant-baseline="central">{lyric}</text>'
        )

    # Pitch detection dots overlay
    if show_pitch_frames:
        for n in verse:
            for pf in n.get("pitchFrames", []):
                x = tx_from_sec(pf["time"])
                y = ty(pf["midi"])
                r = 2.5
                opacity = max(0.3, pf["confidence"])
                color = confidence_color(pf["confidence"])
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
                    f'opacity="{opacity:.2f}" '
                    f'title="{html_module.escape(n["lyric"])} {pitch_to_note(pf["midi"])} '
                    f'@ {pf["time"]:.2f}s (conf: {pf["confidence"]:.2f})"/>'
                )

    # Axes
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" '
        f'stroke="#555" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
        f'y2="{height - pad_bottom}" stroke="#555" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{width / 2}" y="{height - 1}" text-anchor="middle" fill="#888" '
        f'font-size="8" font-family="monospace">time →</text>'
    )
    parts.append(
        f'<text x="8" y="{height / 2}" text-anchor="middle" fill="#888" font-size="8" '
        f'font-family="monospace" transform="rotate(-90,8,{height / 2})">pitch →</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Lyrics ─────────────────────────────────────────────────────────────────────


def build_verse_lyrics(verse: list[dict], *, has_pitch_frames: bool = False) -> str:
    """Build a lyrics line with timing annotations."""
    parts = []
    for n in verse:
        t = ms_to_sec(n["start"])
        lyric = html_module.escape(n["lyric"])
        note = pitch_to_note(n["pitch"])
        dur = f'{n["duration"]:.0f}ms'
        if has_pitch_frames and n.get("pitchFrames"):
            frames = n["pitchFrames"]
            avg_conf = sum(f["confidence"] for f in frames) / len(frames)
            title = f"{t} | {note} | {dur} | frames: {len(frames)} | avg conf: {avg_conf:.2f}"
        else:
            title = f"{t} | {note} | {dur}"
        parts.append(
            f'<span class="lyric-word" title="{title}">'
            f'<span class="lyric-time">[{t}]</span> {lyric}</span>'
        )
    return "\n".join(parts)


# ── Full HTML (Ultrastar + optional pitch overlay) ────────────────────────────


def build_html(data: dict, title: str) -> str:
    """Build the full HTML document.

    Accepts two data shapes:

    1. **Ultrastar parse result** (keys: ``metadata``, ``notes``, ``verses``) —
       optionally with ``pitchFrames`` attached to each note and a ``pitch_word_count``
       / ``pitch_frame_count`` summary.

    2. **Legacy pitch-detection JSON** (keys: ``words``, ``done``) — each word may
       carry ``pitchFrames``.  In this mode the function synthesizes note-like entries
       so the same SVG logic applies.
    """
    is_pitch_only = "words" in data and "metadata" not in data

    if is_pitch_only:
        return _build_pitch_only_html(data, title)
    else:
        return _build_ultrastar_html(data, title)


def _build_ultrastar_html(data: dict, title: str) -> str:
    meta = data["metadata"]
    notes = data["notes"]
    verses = data["verses"]
    has_frames = bool(notes[0].get("pitchFrames")) if notes else False
    pitch_word_count = data.get("pitch_word_count", 0)
    pitch_frame_count = data.get("pitch_frame_count", 0)

    meta_rows = ""
    for k, v in meta.items():
        meta_rows += f"<tr><th>{html_module.escape(k)}</th><td>{html_module.escape(v)}</td></tr>\n"

    total_dur = ms_to_sec(notes[-1]["start"] + notes[-1]["duration"]) if notes else "—"

    summary_parts = [
        f"Total duration: {total_dur}",
        f"{len(notes)} notes",
        f"{len(verses)} verses",
    ]
    if pitch_frame_count:
        summary_parts.append(f"{pitch_word_count} pitch words")
        summary_parts.append(f"{pitch_frame_count} pitch frames")
    summary_str = " | ".join(summary_parts)

    verse_blocks = ""
    for i, verse in enumerate(verses):
        verse_start = ms_to_sec(verse[0]["start"])
        verse_end = ms_to_sec(verse[-1]["start"] + verse[-1]["duration"])
        pitches = set(n["pitch"] for n in verse)
        p_min = min(pitches)
        p_max = max(pitches)

        # Expand range with pitch frames
        for n in verse:
            for pf in n.get("pitchFrames", []):
                p_min = min(p_min, pf["midi"])
                p_max = max(p_max, pf["midi"])

        svg_height = max(120, (p_max - p_min + 2) * 18 + 50)
        svg = build_verse_svg(verse, 960, svg_height, show_pitch_frames=has_frames)
        lyrics = build_verse_lyrics(verse, has_pitch_frames=has_frames)

        verse_info_parts = [
            f"{len(verse)} notes",
            f"Pitch range: {pitch_to_note(p_min)}–{pitch_to_note(p_max)}",
        ]
        verse_frames = sum(len(n.get("pitchFrames", [])) for n in verse)
        if verse_frames:
            verse_info_parts.append(f"{verse_frames} pitch frames")

        verse_blocks += f"""
<div class="verse-block">
<div class="verse-header">
  <span class="verse-num">Verse {i + 1}</span>
  <span class="verse-time">{verse_start} — {verse_end}</span>
  <span class="verse-info">{', '.join(verse_info_parts)}</span>
</div>
{svg}
<div class="verse-lyrics">{lyrics}</div>
</div>
"""

    legend_html = ""
    if has_frames:
        legend_html = """
<div class="legend">
  <span><span class="dot" style="background:rgb(0,255,0)"></span> High confidence</span>
  <span><span class="dot" style="background:rgb(255,200,0)"></span> Medium</span>
  <span><span class="dot" style="background:rgb(255,0,0)"></span> Low confidence</span>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)} — UltraStar Song</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Consolas, monospace; background: #1a1a2e; color: #e0e0e0; padding: 24px; }}
  h1 {{ text-align: center; color: #e94560; margin-bottom: 6px; font-size: 1.6em; }}
  .summary {{ text-align: center; color: #888; margin-bottom: 18px; font-size: 0.85em; }}
  table.meta {{ width: 100%; max-width: 600px; margin: 0 auto 24px; border-collapse: collapse; }}
  table.meta th {{ text-align: right; padding: 3px 12px; color: #e94560; font-size: 0.85em; border-bottom: 1px solid #333; }}
  table.meta td {{ padding: 3px 12px; color: #ccc; font-size: 0.85em; border-bottom: 1px solid #333; }}
  .verse-block {{ background: #16213e; border-radius: 8px; margin-bottom: 24px; overflow: hidden; border: 1px solid #2a2a4a; }}
  .verse-header {{ display: flex; gap: 16px; align-items: center; padding: 8px 14px; background: #0f3460; font-size: 0.8em; flex-wrap: wrap; }}
  .verse-num {{ font-weight: bold; color: #e94560; font-size: 1.05em; }}
  .verse-time {{ color: #4fc3f7; }}
  .verse-info {{ color: #888; }}
  .verse-lyrics {{ padding: 8px 14px 12px; font-size: 0.85em; line-height: 1.9; color: #bbb; }}
  .lyric-time {{ color: #e94560; font-size: 0.8em; font-family: monospace; }}
  .lyric-word {{ margin-right: 2px; }}
  svg text {{ user-select: none; }}
  svg rect {{ cursor: pointer; transition: opacity 0.15s; }}
  svg rect:hover {{ opacity: 0.7; }}
  svg circle {{ cursor: pointer; transition: r 0.15s; }}
  svg circle:hover {{ r: 5; opacity: 1 !important; }}
  .legend {{ display: flex; gap: 18px; justify-content: center; margin-bottom: 12px; font-size: 0.8em; color: #888; }}
  .legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>
<p class="summary">{summary_str}</p>

<table class="meta">
{meta_rows}
</table>

{legend_html}
{verse_blocks}
</body>
</html>"""


def _build_pitch_only_html(data: dict, title: str) -> str:
    """Handle legacy pitch-detection-only JSON (no Ultrastar data).

    This keeps backward compatibility with align.py's call to build_html(whisper_pitch_data, title).
    """
    words = data["words"]
    done = data.get("done", False)
    verses = split_verses(words)

    total_dur = sec_to_str(words[-1]["end"]) if words else "—"
    total_frames = sum(len(w.get("pitchFrames", [])) for w in words)

    verse_blocks = ""
    for i, verse in enumerate(verses):
        v_start = sec_to_str(verse[0]["start"])
        v_end = sec_to_str(verse[-1]["end"])
        pitches = set()
        for w in verse:
            pitches.add(w["midi"])
            for pf in w.get("pitchFrames", []):
                pitches.add(pf["midi"])
        p_min = min(pitches)
        p_max = max(pitches)
        svg_height = max(120, (p_max - p_min + 2) * 18 + 50)

        # Convert pitch-only words to note-like dicts for the shared SVG builder
        note_verse = []
        for w in verse:
            note_verse.append({
                "start": w["start"] * 1000,
                "duration": (w["end"] - w["start"]) * 1000,
                "pitch": w["midi"],
                "lyric": w["word"],
                "chorus": False,
                "pitchFrames": w.get("pitchFrames", []),
            })

        svg = build_verse_svg(note_verse, 960, svg_height, show_pitch_frames=True)

        # Lyrics line
        lyric_parts = []
        for w in verse:
            t = sec_to_str(w["start"])
            lyric = html_module.escape(w["word"])
            note = pitch_to_note(w["midi"])
            frames = w.get("pitchFrames", [])
            avg_conf = sum(f["confidence"] for f in frames) / max(len(frames), 1)
            lyric_parts.append(
                f'<span class="lyric-word" title="{t} | {note} | conf: {avg_conf:.2f}">'
                f'<span class="lyric-time">[{t}]</span> {lyric}</span>'
            )
        lyrics = "\n".join(lyric_parts)

        verse_blocks += f"""
<div class="verse-block">
<div class="verse-header">
  <span class="verse-num">Verse {i + 1}</span>
  <span class="verse-time">{v_start} — {v_end}</span>
  <span class="verse-info">{len(verse)} words | {sum(len(w.get('pitchFrames', [])) for w in verse)} frames | Pitch: {pitch_to_note(p_min)}–{pitch_to_note(p_max)}</span>
</div>
{svg}
<div class="verse-lyrics">{lyrics}</div>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)} — Pitch Detection</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Consolas, monospace; background: #1a1a2e; color: #e0e0e0; padding: 24px; }}
  h1 {{ text-align: center; color: #e94560; margin-bottom: 6px; font-size: 1.6em; }}
  .summary {{ text-align: center; color: #888; margin-bottom: 18px; font-size: 0.85em; }}
  .verse-block {{ background: #16213e; border-radius: 8px; margin-bottom: 24px; overflow: hidden; border: 1px solid #2a2a4a; }}
  .verse-header {{ display: flex; gap: 16px; align-items: center; padding: 8px 14px; background: #0f3460; font-size: 0.8em; flex-wrap: wrap; }}
  .verse-num {{ font-weight: bold; color: #e94560; font-size: 1.05em; }}
  .verse-time {{ color: #4fc3f7; }}
  .verse-info {{ color: #888; }}
  .verse-lyrics {{ padding: 8px 14px 12px; font-size: 0.85em; line-height: 1.9; color: #bbb; }}
  .lyric-time {{ color: #e94560; font-size: 0.8em; font-family: monospace; }}
  .lyric-word {{ margin-right: 2px; }}
  svg text {{ user-select: none; }}
  svg circle {{ cursor: pointer; transition: r 0.15s; }}
  svg circle:hover {{ r: 5; opacity: 1 !important; }}
  .legend {{ display: flex; gap: 18px; justify-content: center; margin-bottom: 12px; font-size: 0.8em; color: #888; }}
  .legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>
<p class="summary">Duration: {total_dur} | {len(words)} words | {total_frames} pitch frames | Done: {done}</p>

<div class="legend">
  <span><span class="dot" style="background:rgb(0,255,0)"></span> High confidence</span>
  <span><span class="dot" style="background:rgb(255,200,0)"></span> Medium</span>
  <span><span class="dot" style="background:rgb(255,0,0)"></span> Low confidence</span>
</div>

{verse_blocks}
</body>
</html>"""


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_preview(
    txt_path: Path,
    output_html: Path | None = None,
    pitch_json_path: Path | str | None = None,
) -> Path:
    """Generate an HTML preview from an Ultrastar .txt file.

    Args:
        txt_path: Path to the Ultrastar .txt file.
        output_html: Optional output path. Defaults to <txt_stem>.html next to the .txt file.
        pitch_json_path: Optional path to pitch detection JSON to overlay.

    Returns:
        Path to the generated HTML file.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Ultrastar file not found: {txt_path}")

    data = parse_ultrastar(txt_path)
    title = data["metadata"].get("TITLE", txt_path.stem)

    if pitch_json_path is not None:
        pitch_data = load_pitch_json(pitch_json_path)
        pitch_words = pitch_data.get("words", [])
        data["notes"] = match_pitch_frames_to_notes(data["notes"], pitch_words)
        # Rebuild verses from enriched notes
        data["verses"] = [
            [n for n in v] for v in data["verses"]
        ]
        # Re-match per-verse (verses share note references, so enriching notes is enough)
        pitch_word_count = len(pitch_words)
        pitch_frame_count = sum(len(w.get("pitchFrames", [])) for w in pitch_words)
        data["pitch_word_count"] = pitch_word_count
        data["pitch_frame_count"] = pitch_frame_count

    if output_html is None:
        output_html = txt_path.with_suffix(".html")
    else:
        output_html = Path(output_html)

    html_content = build_html(data, title)
    output_html.write_text(html_content, encoding="utf-8")

    frame_info = ""
    if pitch_json_path is not None:
        frame_info = f", {data.get('pitch_frame_count', 0)} pitch frames"

    logger.info(
        f"HTML preview: {output_html}  ({len(data['notes'])} notes, {len(data['verses'])} verses{frame_info})"
    )
    return output_html


def main():
    """CLI entry-point: ``python html_preview.py <song.txt> [output.html] [--pitch detection.json]``"""
    if len(sys.argv) < 2:
        print("Usage: python html_preview.py <song.txt> [output.html] [--pitch detection.json]")
        sys.exit(1)

    txt_path = Path(sys.argv[1])
    if not txt_path.exists():
        print(f"Error: {txt_path} not found.")
        sys.exit(1)

    output_html = None
    pitch_json = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--pitch" and i + 1 < len(args):
            pitch_json = args[i + 1]
            i += 2
        else:
            if output_html is None:
                output_html = args[i]
            i += 1

    result = generate_preview(txt_path, output_html, pitch_json)
    print(f"Written: {result}")


if __name__ == "__main__":
    main()
