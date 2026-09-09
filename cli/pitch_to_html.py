#!/usr/bin/env python3
"""Render pitch detection JSON as a scrollable HTML page with verse visualizations."""

import sys
import json
import html as html_module
from pathlib import Path

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch_to_note(pitch: int) -> str:
    octave = pitch // 12 - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"

def sec_to_str(s):
    return f"{s:.2f}s"

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def split_verses(words, gap_threshold=2.0):
    """Split words into verses based on time gaps."""
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

def confidence_color(conf):
    """Map confidence 0..1 to a color: red (low) -> yellow (mid) -> green (high)."""
    if conf < 0.5:
        r, g = 255, int(conf * 2 * 200)
    else:
        r, g = int(255 - (conf - 0.5) * 2 * 255), 200 + int((conf - 0.5) * 2 * 55)
    return f"rgb({r},{g},0)"

def build_verse_svg(verse, width, height):
    """Build SVG: X=time (seconds), Y=pitch. Each word's pitch frames drawn as dots."""
    if not verse:
        return ""

    # Collect all pitch frames
    all_frames = []
    for w in verse:
        for pf in w.get("pitchFrames", []):
            all_frames.append(pf)

    if not all_frames:
        return ""

    times = [f["time"] for f in all_frames]
    pitches = [f["midi"] for f in all_frames]

    t_min = min(times)
    t_max = max(times)
    p_min = min(pitches)
    p_max = max(pitches)

    pad_left = 50
    pad_right = 16
    pad_top = 16
    pad_bottom = 24
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    t_range = max(t_max - t_min, 0.01)
    p_range = max(p_max - p_min, 1)

    def tx(t):
        return pad_left + (t - t_min) / t_range * plot_w

    def ty(p):
        return pad_top + (p_max - p) / p_range * plot_h

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">')

    # Pitch grid
    for p in range(p_min, p_max + 1):
        y = ty(p)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#2a2a4a" stroke-width="0.5"/>')
        parts.append(f'<text x="{pad_left - 4}" y="{y + 3:.1f}" text-anchor="end" fill="#666" font-size="8" font-family="monospace">{p}</text>')

    # Time axis labels
    t_span = t_range
    if t_span > 30:
        step = 5
    elif t_span > 10:
        step = 2
    elif t_span > 4:
        step = 1
    elif t_span > 1.5:
        step = 0.5
    else:
        step = 0.2

    for t in numpy_range(t_min, t_max, step):
        x = tx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height - pad_bottom}" stroke="#1e1e3a" stroke-width="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 4}" text-anchor="middle" fill="#666" font-size="8" font-family="monospace">{t:.1f}s</text>')

    # Draw word blocks (background rectangles)
    for w in verse:
        x1 = tx(w["start"])
        x2 = tx(w["end"])
        parts.append(f'<rect x="{x1:.1f}" y="{pad_top}" width="{max(x2 - x1, 1):.1f}" height="{plot_h}" fill="#0a0a1a" opacity="0.4" rx="2"/>')

    # Draw pitch frames as small circles colored by confidence
    for w in verse:
        for pf in w.get("pitchFrames", []):
            x = tx(pf["time"])
            y = ty(pf["midi"])
            r = 2.5
            opacity = max(0.3, pf["confidence"])
            color = confidence_color(pf["confidence"])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" opacity="{opacity:.2f}" '
                f'title="{html_module.escape(w["word"])} {pitch_to_note(pf["midi"])} @ {pf["time"]:.2f}s (conf: {pf["confidence"]:.2f})"/>'
            )

    # Draw word labels on the SVG
    for w in verse:
        x1 = tx(w["start"])
        x2 = tx(w["end"])
        cx = (x1 + x2) / 2
        w_dur = w["end"] - w["start"]
        lyric = html_module.escape(w["word"])
        fs = max(7, min(11, (x2 - x1) / (len(lyric) * 0.5)))
        # Place label at the dominant pitch
        dominant_y = ty(w["midi"])
        parts.append(
            f'<text x="{cx:.1f}" y="{dominant_y + fs * 0.4:.1f}" text-anchor="middle" '
            f'fill="#fff" font-size="{fs:.1f}" font-family="sans-serif" font-weight="bold" '
            f'dominant-baseline="central" opacity="0.9">{lyric}</text>'
        )

    # Axes
    parts.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#555" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#555" stroke-width="1"/>')
    parts.append(f'<text x="{width / 2}" y="{height - 1}" text-anchor="middle" fill="#888" font-size="8" font-family="monospace">time →</text>')
    parts.append(f'<text x="8" y="{height / 2}" text-anchor="middle" fill="#888" font-size="8" font-family="monospace" transform="rotate(-90,8,{height / 2})">pitch →</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def numpy_range(start, stop, step):
    """Simple range generator for floats."""
    while start <= stop:
        yield round(start, 10)
        start += step


def build_verse_lyrics(verse):
    """Build lyrics line with timing annotations."""
    parts = []
    for w in verse:
        t = sec_to_str(w["start"])
        lyric = html_module.escape(w["word"])
        note = pitch_to_note(w["midi"])
        avg_conf = sum(f["confidence"] for f in w.get("pitchFrames", [])) / max(len(w.get("pitchFrames", [1])), 1)
        parts.append(
            f'<span class="lyric-word" title="{t} | {note} | conf: {avg_conf:.2f}">'
            f'<span class="lyric-time">[{t}]</span> {lyric}</span>'
        )
    return "\n".join(parts)


def build_html(data: dict, title: str) -> str:
    words = data["words"]
    done = data.get("done", False)
    verses = split_verses(words)

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
        svg = build_verse_svg(verse, 960, svg_height)
        lyrics = build_verse_lyrics(verse)

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

    total_dur = sec_to_str(words[-1]["end"]) if words else "—"
    total_frames = sum(len(w.get("pitchFrames", [])) for w in words)

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python pitch_to_html.py <detection.json> [output.html]")
        sys.exit(1)

    infile = sys.argv[1]
    if not Path(infile).exists():
        print(f"Error: {infile} not found.")
        sys.exit(1)

    data = load_json(infile)
    title = Path(infile).stem

    outfile = sys.argv[2] if len(sys.argv) > 2 else title + ".html"
    html_content = build_html(data, title)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html_content)

    verses = split_verses(data["words"])
    print(f"Written: {outfile}  ({len(data['words'])} words, {len(verses)} verses)")


if __name__ == "__main__":
    main()
