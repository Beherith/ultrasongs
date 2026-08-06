#!/usr/bin/env python3
"""Parse an UltraStar Deluxe .txt song file and render it as a scrollable HTML page with verse visualizations."""

import sys
import html as html_module
from pathlib import Path

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch_to_note(pitch: int) -> str:
    octave = pitch // 12 - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"

def beats_to_ms(beats, bpm):
    """Convert beat count to milliseconds. Each beat unit is a 16th note (1/4 of a quarter-note beat at BPM)."""
    return beats * (60000.0 / bpm / 4)

def ms_to_sec(ms):
    """Format milliseconds as seconds (e.g. 32.65s)."""
    return f"{ms / 1000:.2f}s"

def parse_ultrastar(filepath: str) -> dict:
    metadata = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                metadata[key.strip()] = value.strip()

    bpm = float(metadata.get("BPM", 120))
    gap = float(metadata.get("GAP", 0))

    def convert_note(parts, is_chorus):
        return {
            "start": gap + beats_to_ms(float(parts[0]), bpm),
            "duration": beats_to_ms(float(parts[1]), bpm),
            "pitch": int(parts[2]),
            "lyric": parts[3],
            "chorus": is_chorus,
        }

    verses = []
    current_verse = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
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

def build_verse_svg(verse, width, height):
    """Build an SVG for a verse: X=time, Y=pitch (inverted so high pitch = top)."""
    if not verse:
        return ""

    starts = [n["start"] for n in verse]
    ends = [n["start"] + n["duration"] for n in verse]
    pitches = [n["pitch"] for n in verse]

    t_min = min(starts)
    t_max = max(ends)
    p_min = min(pitches)
    p_max = max(pitches)

    pad_left = 50
    pad_right = 16
    pad_top = 16
    pad_bottom = 24
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    t_range = max(t_max - t_min, 1)
    p_range = max(p_max - p_min, 1)

    def tx(ms):
        return pad_left + (ms - t_min) / t_range * plot_w

    def ty(pitch):
        return pad_top + (p_max - pitch) / p_range * plot_h

    # Build SVG elements
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">')

    # Background grid lines for pitch
    for p in range(p_min, p_max + 1):
        y = ty(p)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#2a2a4a" stroke-width="0.5"/>')
        parts.append(f'<text x="{pad_left - 4}" y="{y + 3:.1f}" text-anchor="end" fill="#666" font-size="8" font-family="monospace">{p}</text>')

    # Time axis labels
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
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height - pad_bottom}" stroke="#1e1e3a" stroke-width="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 4}" text-anchor="middle" fill="#666" font-size="8" font-family="monospace">{t / 1000:.1f}s</text>')

    # Draw notes
    for n in verse:
        x = tx(n["start"])
        w = tx(n["start"] + n["duration"]) - x
        y = ty(n["pitch"])
        bar_h = max(plot_h / (p_range + 1) * 0.7, 8)
        fill = "#e94560" if n["chorus"] else "#0f3460"
        stroke = "#ff6b81" if n["chorus"] else "#4fc3f7"
        text_fill = "#fff" if n["chorus"] else "#cde"
        rx = x + w / 2
        ry = y
        bw = max(w, 2)
        parts.append(
            f'<rect x="{x:.1f}" y="{(y - bar_h / 2):.1f}" width="{bw:.1f}" '
            f'height="{bar_h:.1f}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="0.8" '
            f'title="{html_module.escape(n["lyric"])} ({pitch_to_note(n["pitch"])}) {ms_to_sec(n["start"])} +{n["duration"]:.0f}ms"/>'
        )
        lyric = html_module.escape(n["lyric"])
        fs = max(7, min(12, bw / (len(lyric) * 0.55)))
        parts.append(
            f'<text x="{rx:.1f}" y="{ry + fs * 0.35:.1f}" text-anchor="middle" '
            f'fill="{text_fill}" font-size="{fs:.1f}" font-family="sans-serif" '
            f'dominant-baseline="central">{lyric}</text>'
        )

    # Axes
    parts.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#555" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#555" stroke-width="1"/>')
    parts.append(f'<text x="{width / 2}" y="{height - 1}" text-anchor="middle" fill="#888" font-size="8" font-family="monospace">time →</text>')
    parts.append(f'<text x="8" y="{height / 2}" text-anchor="middle" fill="#888" font-size="8" font-family="monospace" transform="rotate(-90,8,{height / 2})">pitch →</text>')

    parts.append("</svg>")
    return "\n".join(parts)

def build_verse_lyrics(verse):
    """Build a lyrics line with timing annotations."""
    parts = []
    for n in verse:
        t = ms_to_sec(n["start"])
        lyric = html_module.escape(n["lyric"])
        parts.append(f'<span class="lyric-word" title="{t} | {pitch_to_note(n["pitch"])} | {n["duration"]:.0f}ms"><span class="lyric-time">[{t}]</span> {lyric}</span>')
    return "\n".join(parts)

def build_html(data: dict, title: str) -> str:
    meta = data["metadata"]
    notes = data["notes"]
    verses = data["verses"]

    meta_rows = ""
    for k, v in meta.items():
        meta_rows += f"<tr><th>{html_module.escape(k)}</th><td>{html_module.escape(v)}</td></tr>\n"

    total_dur = ms_to_sec(notes[-1]["start"] + notes[-1]["duration"]) if notes else "—"

    verse_blocks = ""
    for i, verse in enumerate(verses):
        verse_start = ms_to_sec(verse[0]["start"])
        verse_end = ms_to_sec(verse[-1]["start"] + verse[-1]["duration"])
        pitches = set(n["pitch"] for n in verse)
        p_min = min(pitches)
        p_max = max(pitches)
        svg_height = max(120, (p_max - p_min + 2) * 18 + 50)
        svg = build_verse_svg(verse, 960, svg_height)
        lyrics = build_verse_lyrics(verse)

        verse_blocks += f"""
<div class="verse-block">
<div class="verse-header">
  <span class="verse-num">Verse {i + 1}</span>
  <span class="verse-time">{verse_start} — {verse_end}</span>
  <span class="verse-info">{len(verse)} notes | Pitch range: {pitch_to_note(p_min)}–{pitch_to_note(p_max)}</span>
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
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>
<p class="summary">Total duration: {total_dur} | {len(notes)} notes | {len(verses)} verses</p>

<table class="meta">
{meta_rows}
</table>

{verse_blocks}
</body>
</html>"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python ultrastar_to_html.py <song.txt> [output.html]")
        sys.exit(1)

    infile = sys.argv[1]
    if not Path(infile).exists():
        print(f"Error: {infile} not found.")
        sys.exit(1)

    data = parse_ultrastar(infile)
    title = data["metadata"].get("TITLE", Path(infile).stem)

    outfile = sys.argv[2] if len(sys.argv) > 2 else Path(infile).stem + ".html"
    html_content = build_html(data, title)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Written: {outfile}  ({len(data['notes'])} notes, {len(data['verses'])} verses)")

if __name__ == "__main__":
    main()
