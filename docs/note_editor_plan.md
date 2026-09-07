# Ultrasongs Visual Note Editor — Implementation Plan

A self-contained, in-browser editor for hand-tuning a near-final Ultrastar `.txt`.
It generates one HTML file per song (no server, no build step) that visually matches
the SVGs produced by `cli/html_preview.py`, adds a live FFT background, and supports
drag/resize/split/merge/delete/gold/lyric editing with an undo stack and
audio + MIDI + metronome playback.

Status: **approved design** (decisions locked with the user), ready to implement.

---

## 1. Goals

- Open a generated `.txt` and fix its notes by ear and by eye, one lyric line at a time.
- The page must look like the existing preview SVG (note bars, pitch dots, whisper-word
  labels, beat grid, axes, annotated lyric line) so the user recognizes the picture.
- An FFT spectrogram (from the **vocals stem**) sits in the background, **2 octaves** tall,
  aligned so a frequency row lines up with the matching MIDI note row.
- Fast, low-friction local workflow: `python -m cli edit …` → double-click the HTML →
  load the vocals MP3 → edit → download the corrected `.txt`.

## 2. Non-goals / out of scope (v1)

- No server, no multi-user, no auth, no persistence backend (save = download).
- No editing of the header (`#BPM`, `#GAP`, `#TITLE`, …) — read-only, displayed only.
- No redo (explicitly "useless" per the user).
- No adding/removing **line breaks** (`-` notes); the line count is fixed for the session.
  (Split/merge/delete act on sung notes only.)
- No audio resampling, normalization, or effects.
- No mobile/touch optimization (mouse + keyboard first; pointer events work on touch
  incidentally).
- No i18n; UI text is English.

---

## 3. Requirements (refined)

| ID | Requirement | Refinement |
|----|-------------|------------|
| R1 | Visual editor for manual tweaks to a near-final `.txt` | Edits notes in place; header preserved verbatim on save. |
| R2 | Very closely resemble the CLI `.svg` | Reuse the exact `build_verse_svg` visual language (colors, strokes, fonts, layout, tooltips). See §9. |
| R3 | One "page" per lyrics line | A "line" = the run of sung notes between two `-` line-break notes. Prev/next page nav + keyboard. |
| R4 | FFT in the background, only 2 octaves | Live Web-Audio STFT of the vocals stem, rendered to a `<canvas>`; y-axis = log₂ frequency spanning a 24-semitone window aligned to the MIDI rows. See §8. |
| R5 | Drag note up/down, left/right | Vertical = pitch (MIDI), horizontal = start time (beat). Snapped (integer MIDI, integer beat). See §10. |
| R6 | Drag edge of note to lengthen | Right-edge handle changes `duration` (min 1 beat, cannot overlap the next note). Optional left-edge handle adjusts start. |
| R7 | Double-click or button to change lyrics | Double-click a note (or toolbar button) → inline popover edits the syllable, preserving Ultrastar space semantics. |
| R8 | Split or merge two adjacent notes | Split at playhead/midpoint; merge selected with next. Exact semantics in §11. |
| R9 | Delete a note | Remove the selected sung note. |
| R10 | Mark note as "gold note" (`*`) | Toggle `:` ↔ `*` on the selected note. |
| R11 | Undo stack (no redo) | Snapshot-per-mutation stack, `Ctrl+Z`, capped at 100, in-memory. |
| R12 | Display processing timing info just like the SVG | Pitch dots (amplitude color, confidence opacity), whisper-word labels + arrowheads, word-block shading, 4/4 beat grid, time + pitch axes, `[time] word` lyric annotations, hover tooltips. Data from `--pitch` JSON. |
| R13 | Play audio + notes as MIDI-like tone + beat metronome on spacebar | Web Audio: audio buffer from line start + per-note oscillator tones + per-beat metronome clicks (accented downbeat), from `#BPM`/`#GAP`; moving playhead + active-note highlight. |

---

## 4. Invocation / CLI

```
python -m cli edit --txt <path/to/song.txt> \
    [--pitch <path/to/whisperx_pitch.json>] \
    [--vocals <path/to/vocals.mp3>] \
    [--embed-audio] \
    [--output <path/to/song_editor.html>]
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--txt` | yes | The Ultrastar `.txt` to edit. Resolved for `#MP3`/`#VIDEO` relative to its own directory (for hints only). |
| `--pitch` | no | Path to a `whisperx_pitch.json` (shape: `{"words":[{word,start,end,midi,characters[],pitchFrames[]}], "done":bool}`). Enables R12 overlays. If omitted, `pitch` payload is `null` and the editor still works (no dots/labels). |
| `--vocals` | no | Path to the vocals stem. **Not embedded by default**; only its basename is embedded as `audioHint` so the UI can tell the user which file to load. With `--embed-audio` the file is base64-embedded. |
| `--embed-audio` | no | Base64-embed the `--vocals` MP3 into the HTML (truly single-file, no picker). Default **off** (keeps the HTML small). Errors clearly if `--embed-audio` is given without `--vocals`. |
| `--output` | no | Output HTML path. Default: `<txt_stem>_editor.html` next to the `.txt`. |

Wiring: new `edit` subparser + `_cmd_edit(args, config)` in `cli/__main__.py`, lazy-importing
`cli.editor.generate_editor`. `config` is loaded but only used for nothing critical (the editor
is header-driven); keep the signature consistent with the other `_cmd_*` adapters.

Exit codes: `0` on success (log the written path), `1` on error (missing file, bad `--embed-audio`).

The command mirrors `preview` (`cli/__main__.py:89-93`, `_cmd_preview`).

---

## 5. Generated file: structure & embedded data

`generate_editor` writes a single HTML document: the template `cli/editor_template.html` with
two placeholders substituted:

- `__EDITOR_DATA__` → `json.dumps(payload, ensure_ascii=False)` (see schema below).
- `__AUDIO_B64__` → base64 string or `""` (only non-empty with `--embed-audio`).

The template inlines all CSS and JS (no external requests, no CDN). It reads
`window.EDITOR_DATA` at load.

### 5.1 Payload schema (`window.EDITOR_DATA`)

```jsonc
{
  "rawHeader": [                       // verbatim leading '#' lines, in order
    "#TITLE:Diggy Diggy Hole",
    "#ARTIST:Yogscast",
    "#MP3:Diggy Diggy Hole.mp3",
    "#COVER:Diggy Diggy Hole.jpg",     // preserved even though build_ultrastar_txt doesn't know it
    "#BPM:121.3",
    "#GAP:32646"
  ],
  "meta": {
    "title": "Diggy Diggy Hole",
    "artist": "Yogscast",
    "bpm": 121.3,                      // exactly as in #BPM — used verbatim (pipeline files already bake in beat_resolution_multiplier)
    "gap": 32646,                      // int ms, from #GAP
    "mp3": "Diggy Diggy Hole.mp3",     // from #MP3 (may be "")
    "video": null                      // from #VIDEO (may be null)
  },
  "bpm": 121.3,
  "gap": 32646,
  "beatMs": 123.66…,                   // 60000 / bpm / 4  = one .txt beat unit (16th note at the listed BPM) — the note SNAP grid
  "pulseMs": 494.64…,                  // 60000 / bpm       = one quarter note (4x the beat unit) — the visual grid + metronome pulse
  "notes": [                           // full ordered note list, exactly as parsed
    { "id": 0,  "type": ":", "start": 0,  "dur": 2, "pitch": 17, "syl": "Broth" },
    { "id": 1,  "type": ":", "start": 2,  "dur": 2, "pitch": 17, "syl": " ers" },
    ...
    { "id": 14, "type": "-", "start": 14, "dur": 0, "pitch": 0,  "syl": "" },
    { "id": 15, "type": ":", "start": 16, "dur": 4, "pitch": 25, "syl": "Swing," }
  ],
  "srcName": "diggy_diggy_hole.txt",      // basename of the input .txt — the download filename
  "pitchWords": [                         // FULL whisper word list (only when --pitch given; else [])
    {
      "word": "Broth",
      "start_ms": 32646.0,                // word timing, absolute song time (ms)
      "end_ms": 32700.0,
      "midi": 49,
      "frames": [                         // per-character/word pitch frames of this word
        { "t_ms": 32650.0, "m": 50, "conf": 0.48, "amp": 0.175 }, ...
      ]
    }, ...
  ],
  "audioHint": "Diggy Diggy Hole_vocals.mp3",   // basename to suggest, or "" 
  "audioB64": "",                          // "" unless --embed-audio
  "audioMime": "audio/mpeg"                // set when --embed-audio
}
```

Notes on the schema:

- `notes` is the **single source of truth**. `start`/`dur` are **beats** (integers), `pitch` is
  MIDI (int). The JS derives milliseconds: `start_ms = gap + start*beatMs`,
  `end_ms = start_ms + dur*beatMs`.
- `-` line-break notes have `type "-"`, `dur 0`, `pitch 0`, `syl ""`. They are **not**
  rendered or editable (no drag/split/merge/delete/gold/lyric); the JS groups sung notes
  into lyric-line "pages" delimited by them.
- `syl` preserves leading/trailing spaces exactly as parsed (Ultrastar word-join semantics).
- `pitchWords` is the **full** word/frame list (absolute song time in ms) — a compact copy of
  `whisperx_pitch.json` with seconds converted to ms. The JS slices it into the **live** page
  window (`[first_note_start_ms - 200, last_note_end_ms + 400]`, cached per
  page+window) each render. Because slicing happens at render time, the overlays stay
  correct under any edit that changes the line structure (e.g. deleting every note of a
  line). The overlays do **not** move when a note is dragged — they show where processing
  detected pitch/words, which is exactly what the user aligns notes against. Total payload
  size equals a pre-sliced per-page design (same frames, just not partitioned).
- Two distinct "beat" concepts — do not confuse them:
  - **`beatMs = 60000 / bpm / 4`** = one **`.txt` beat unit** (a 16th note at the listed BPM).
    Note `start`/`dur` are counted in this unit; this is the **snap grid** for dragging (§10).
    Matches `html_preview.beats_to_ms` (`60000/bpm/4`) and `ultrastar.ms_to_beats`.
  - **`pulseMs = 60000 / bpm`** = one **quarter note** (4× the beat unit). This is the spacing of
    the **visual beat-grid lines** and the **metronome clicks** (§9.2, §13.3), so the metronome
    lands exactly on the visible grid lines — matching `build_verse_svg` (`beat_ms = 60000/bpm`).
- Because the file's `#BPM` already includes `beat_resolution_multiplier`, **no** extra scaling
  is applied at edit time.

### 5.2 Raw header extraction

`extract_raw_header(text) -> (raw_header_lines, body_text)`:
- Iterate lines from the top; while `line.strip().startswith("#")`, append the line **verbatim**
  (no strip) to `raw_header_lines`. Stop at the first non-`#` line (start of body).
- `body_text` is the remainder (unused for output, kept for tests/diagnostics).
- This guarantees **lossless** header round-trip: `#COVER`, `#BACKGROUND`, `#CREATOR`, ordering,
  and formatting are all preserved on save, unlike `build_ultrastar_txt` (which emits only
  TITLE/ARTIST/MP3/VIDEO/BPM/GAP).

---

## 6. Coordinate system & geometry (exact)

One shared layout is used by **both** the background `<canvas>` and the overlay `<svg>` so their
plot areas coincide exactly. All values are in the internal coordinate space; the SVG uses
`viewBox="0 0 W H"` and scales to 100% container width; the canvas is drawn at the same
pixel size then CSS-scaled identically.

| Constant | Value | Meaning |
|----------|-------|---------|
| `W` | `1280` | Internal width |
| `H` | `480` | Internal height |
| `PAD_L` | `56` | Left padding (MIDI labels) |
| `PAD_R` | `24` | Right padding |
| `PAD_T` | `20` | Top padding |
| `PAD_B` | `44` | Bottom padding (time axis + whisper labels) |
| `PLOT_W` | `W - PAD_L - PAD_R` = `1200` | Plot width |
| `PLOT_H` | `H - PAD_T - PAD_B` = `416` | Plot height |
| `MARGIN_L_MS` | `200` | Time margin left of first note |
| `MARGIN_R_MS` | `400` | Time margin right of last note |
| `MIDI_MIN` | `24` | Global pitch floor (C1) for clamping |
| `MIDI_MAX` | `96` | Global pitch ceiling (C7) for clamping |

Mappings (match `html_preview.build_verse_svg:233-240`):

```
t_range = max(t1 - t0, 1)          # ms
p_range = max(winHi - winLo, 1)    # semitones

tx(ms)    = PAD_L + (ms - t0) / t_range * PLOT_W
ty(midi)  = PAD_T + (winHi - midi) / p_range * PLOT_H
```

Per-page time window (`t0`, `t1`), computed **live** in JS from the current notes + the page's
frames/words:

```
cand_starts = [note.start_ms for sung notes] + [f.t_ms for frames] + [w.start_ms for words]
cand_ends   = [note.end_ms   for sung notes] + [f.t_ms for frames] + [w.end_ms   for words]
t0 = max(0, min(cand_starts) - MARGIN_L_MS)
t1 = max(t0 + 250, max(cand_ends) + MARGIN_R_MS)
```

## 7. MIDI window (the "2 octaves")

Computed **live** in JS per page from the current sung notes (frames may nudge the edges):

```
note_lo = min(note.pitch); note_hi = max(note.pitch)
if note_hi - note_lo <= 24:
    mid     = round((note_lo + note_hi) / 2)
    win_lo  = mid - 12
    win_hi  = mid + 12
    if win_lo > note_lo: win_lo = note_lo;        win_hi = win_lo + 24
    if win_hi < note_hi: win_hi = note_hi;        win_lo = win_hi - 24
    win_lo  = clamp(win_lo, MIDI_MIN, MIDI_MAX - 24)
    win_hi  = win_lo + 24
else:                                    # line genuinely spans > 2 octaves
    win_lo, win_hi = note_lo, note_hi
```

- Normal case: exactly **24 semitones** (2 octaves).
- If a line's notes span more than 2 octaves, the window expands to fit (the "2 octaves" rule is
  the common case; this is the documented exception).
- Pitch frames outside `[win_lo, win_hi]` are **clipped** (not drawn).
- Dragging a note clamps pitch to `[win_lo, win_hi]` (§10).

`freq(midi) = 440 * 2 ** ((midi - 69) / 12)`; `midi_of(f) = 69 + 12 * log2(f / 440)`.
Because log₂(f) is linear in MIDI, a log₂-frequency axis over `[win_lo, win_hi]` aligns 1:1 with
the MIDI note rows — this is what makes the FFT background line up with the note bars.

---

## 8. FFT / STFT background (in-browser)

Rendered to a `<canvas>` positioned exactly under the SVG plot area. Computed **lazily** when a
page becomes visible (and cached per page index until its notes' window changes).

### 8.1 Signal
- Source: the decoded `AudioBuffer` of the **loaded vocals MP3** (mono mixdown: average channels).
- Segment: samples covering `[t0, t1]` (the page time window). If the file is shorter, pad with
  zeros / clamp.

### 8.2 STFT parameters (mirror `cli/align.py:_audio_spectrogram`)
| Param | Value |
|-------|-------|
| `N_FFT` | `2048` (Hann window) |
| `HOP` | `512` (75 % overlap) |
| Window | Hann: `0.5 - 0.5*cos(2π n / (N_FFT-1))` |
| Magnitude | `|FFT|` (amplitude) |
| dB | `20 * log10(x + 1e-7)` |
| dB floor/top (per page) | 5th / 99.5th percentile of all frame dB values |
| Color | `magma` (see §8.4) |

- `n_frames = 1 + max(0, (seg_len - N_FFT) // HOP)`; frame `k` time = `t0 + k*HOP/sr` (sec) → ms.
- A compact iterative **radix-2 FFT** (real input, `N_FFT` points) is included in the template
  (~120 lines, no dependencies). `freq(bin) = bin * sr / N_FFT`.

### 8.3 MIDI-binned rows (alignment trick)
Instead of a linear-frequency spectrogram, each STFT frame is binned **per MIDI row** so rows
align exactly with note rows:

```
for m in range(win_lo, win_hi + 1):
    f_lo = freq(m - 0.5); f_hi = freq(m + 0.5)
    bin_lo = floor(f_lo / (sr / N_FFT)); bin_hi = ceil(f_hi / (sr / N_FFT))
    energy[m] = sum(mag[bin_lo:bin_hi])      # coherent energy across the semitone band
db[m][k] = 20*log10(energy[m][k] + 1e-7)
```

Result: an `(n_midi_rows × n_frames)` matrix, `n_midi_rows = winHi - winLo + 1`.

### 8.4 Colormap
- Embed a **64-entry `magma` LUT** (RGB) in the template, generated once from
  `matplotlib.colormaps["magma"]` during implementation (one-off script) and hard-coded.
- Value `v = (db - floor) / (top - floor)` clamped to `[0,1]`; color = LUT interpolate at
  `v * 63`.
- Build an offscreen `n_frames × n_midi_rows` `ImageData`, then `drawImage` it scaled to the plot
  rect (`imageSmoothingEnabled = true` for a smooth look). Fast and crisp.

### 8.5 Canvas ↔ SVG alignment
- The canvas is sized `PLOT_W × PLOT_H` (internal units) and positioned at `(PAD_L, PAD_T)`.
- Canvas row `r` (0=top) ↔ MIDI `winHi - r`; canvas column `c` ↔ time `t0 + (c / PLOT_W) * t_range`.
- The SVG overlays at the same origin/size, so `tx`/`ty` land on the same pixels. Both are wrapped
  in a single relatively-positioned container scaled together.

---

## 9. SVG overlay rendering spec (matches `build_verse_svg`)

Draw order (back → front): word-block shading, beat grid, pitch grid + labels, time axis,
note bars (+ lyric text), whisper-word labels, pitch dots, axes, playhead.

Colors/functions are ported verbatim from `cli/html_preview.py`:
`NOTE_NAMES`, `pitch_to_note` (`:21`), `confidence_color` (`:41`), `amplitude_color` (`:50`),
`beats_to_ms` (`:27`), `ms_to_sec` (`:34`).

1. **Word-block shading** (only when `pitch.words` present): for each whisper word, a
   `<rect x=tx(w.start_ms) y=PAD_T width=max(tx(w.end_ms)-tx(w.start_ms),1) height=PLOT_H
   fill=#0a0a1a opacity=0.3 rx=2>` (`html_preview.py:312-316`).
2. **Beat grid (4/4)** (`html_preview.py:285-307`): grid lines spaced `pulseMs = 60000/bpm`
   (one quarter note = 4× the `.txt` beat unit);
   for `beat_idx` where `beat_time = gap + beat_idx*pulseMs` falls in `[t0, t1]`:
   - downbeat (`beat_idx % 4 == 0`): `stroke=rgba(255,255,255,0.35) stroke-width=1.5`
   - else: `stroke=rgba(255,255,255,0.12) stroke-width=0.8`
   - full-height vertical line within the plot.
3. **Pitch grid + labels** (`html_preview.py:249-257`): for `p in win_lo..winHi`:
   `line stroke=#2a2a4a width=0.5`; label `text fill=#666 font-size=8 font-family=monospace
   x=PAD_L-4 text-anchor=end` showing `p`; additionally show `pitch_to_note(p)` when `p % 12 == 0`.
4. **Time axis** (`html_preview.py:260-282`): step by `t_range` (`>20000→5000, >8000→2000,
   >3000→1000, else 500` ms); vertical `line stroke=#1e1e3a width=0.5`; label
   `fill=#666 font-size=8` at `y=H-4`, text `{t/1000:.1f}s`.
5. **Note bars** (`html_preview.py:321-349`): for each sung note:
   - `x=tx(start_ms)`, `w=max(tx(end_ms)-x, 2)`, `y=ty(pitch)`,
     `bar_h = max(PLOT_H / (p_range + 1) * 0.7, 8)`.
   - Normal (`:`): `fill=#0f3460 stroke=#4fc3f7`; gold (`*`): `fill=#e94560 stroke=#ff6b81`;
     text fill `#cde` / `#fff` respectively. `rx=2 stroke-width=0.8`.
   - Lyric text centered: `font-size = max(7, min(12, w / (len(syl)*0.55)))`,
     `dominant-baseline=central`, `user-select=none`.
   - Hover `title`: `{syl} ({pitch_to_note(pitch)}) {ms_to_sec(start_ms)} +{dur}ms`.
   - Selected note: add `stroke=#ffffff stroke-width=2` (selection highlight).
6. **Whisper-word labels** (`html_preview.py:352-394`): for each page word, green `#4caf50`:
   - label `font-style=italic font-size=max(6,min(9, bw/(len*0.55))) opacity=0.85` centered at the
     word's mid-x, `y` just below the note band.
   - underline `line stroke=#4caf50 width=1.2 opacity=0.5` from `tx(w.start_ms)` to `tx(w.end_ms)`
     with arrowhead `<polygon>` triangles (w=5, h=3.5) at each end, `opacity=0.5`.
7. **Pitch dots** (`html_preview.py:397-414`): for each frame in `[t0,t1]`∩`[win_lo,winHi]`:
   - `circle cx=tx(f.t_ms) cy=ty(f.m) r=2.5`,
     `fill=amplitude_color(f.amp, page.ampMin, page.ampMax)`,
     `opacity=max(0.15, f.conf)`.
   - Hover `title`: `{word} {pitch_to_note(f.m)} @ {f.t_ms/1000:.2f}s (conf, amp)`.
8. **Axes** (`html_preview.py:417-432`): left + bottom `line stroke=#555 width=1`;
   `text "time →"` bottom-center; `text "pitch →"` rotated -90° at `x=8`.
9. **Annotated lyric line** (below the plot, `build_verse_lyrics` `:441-461`): for each sung note
   in the page, a `<span class="lyric-word">` with `[t]` (`ms_to_sec`, `color=#e94560`) + syllable;
   hover `title` = `{t} | {pitch_to_note} | {dur}ms` (+ `avg conf` when frames exist).
10. **Playhead**: a `line` (full plot height) at `tx(playhead_ms)`, `stroke=#ffd54a width=1.5`,
    drawn last; plus a small triangle marker at the top. Hidden when not playing.

Everything is re-rendered (or the affected element updated) on state change; the FFT canvas is
**not** re-rendered during a drag (it's time-fixed).

---

## 10. Interaction model

Pointer events (`pointerdown/move/up`) with `setPointerCapture`. A note bar has:
- a **body hit area** (the bar rect, extended ±4 px vertically for easy grabbing) → move drag;
- a **right-edge handle** (8 px-wide zone at the bar's right, `cursor: ew-resize`) → duration drag;
- an optional **left-edge handle** (8 px, `cursor: ew-resize`) → start drag (keeps end fixed).

### 10.1 Selection
- `click` a note → select it (white highlight, §9.5). `click` empty plot → deselect.
- The toolbar's contextual actions (Split/Merge/Delete/Gold/Lyric) act on the selection.

### 10.2 Drag move (R5)
On `pointerdown` on a bar body: `pushUndo()` once; capture `orig_start_beat`, `orig_pitch`.
On `pointermove`:
```
d_ms   = (dx / PLOT_W) * t_range
d_beat = round(d_ms / beatMs)
new_start_beat = clamp(orig_start_beat + d_beat, prev_note_end_beat, next_note_start_beat - 1)
d_midi = round(-(dy) / (PLOT_H / p_range))          # up (dy<0) = higher pitch
new_pitch = clamp(orig_pitch + d_midi, win_lo, win_hi)
```
- `prev_note_end_beat` = end beat of the previous sung note (or 0); `next_note_start_beat` = start
  beat of the next sung note (or +∞). Notes **never overlap**.
- Live update: bar `x`, `y`, lyric label position, selection highlight. No FFT/dot re-render.
On `pointerup`: commit (no extra undo push).

Snapping is to **integer beats** and **integer MIDI** (the grid). A "free" (no-snap) toggle is a
stretch goal, **off** by default.

### 10.3 Drag right edge (R6, lengthen)
`pushUndo()` once; capture `orig_dur_beat`.
```
d_beat  = round((dx / PLOT_W) * t_range / beatMs)
new_dur = clamp(orig_dur_beat + d_beat, 1, next_note_start_beat - start_beat)
```
- Min duration **1 beat**; cannot extend past the next note's start.
- Live update: bar `width`, lyric label centering.

### 10.4 Drag left edge (optional)
`pushUndo()` once; capture `orig_start_beat`, `end_beat`.
```
new_start = clamp(orig_start_beat + d_beat, prev_note_end_beat, end_beat - 1)
new_dur   = end_beat - new_start
```

---

## 11. Edit operations (exact semantics)

All operate on the **selected** sung note `N` (start `s`, dur `d`, pitch `p`, syl `y`, type `t`).
Each first calls `pushUndo()`. Beat values stay integers; results re-verify the no-overlap rule.

### 11.1 Split (R8a)
- Split point `b`: the playhead beat if the playhead is inside `N`, else `round(s + d/2)`.
- Guard: `d >= 2` and `s < b < s + d`; otherwise the action is a no-op (button disabled).
- Result (two notes, same type and pitch):
  - `A`: `start=s, dur=b-s, pitch=p, syl=y`
  - `B`: `start=b, dur=(s+d)-b, pitch=p, syl="~"`   (unvoiced placeholder; user double-clicks to set)
- `B` is inserted immediately after `A`. (A long sung note is conventionally the first half keeps
  the syllable; the second half is a gap to be relabeled.)

### 11.2 Merge (R8b)
- Merge `N` with the **next** note `X` (the following note in the list).
- Guard: `X` exists and is a sung note (not `-`); otherwise disabled.
- Result: `start=s_N, dur=d_N + d_X, pitch=p_N, syl=y_N + y_X, type=t_N`. `X` is removed.
  - Pitch takes the **first** note's; syllables are **concatenated verbatim** (preserving any
    internal space semantics). The user can re-edit afterward.
- (Stretch: `Shift+M` merges with the previous note — not in v1.)

### 11.3 Delete (R9)
- Remove `N` from the list. Line breaks are never deleted by this action.
- If the page becomes empty of sung notes, the page still renders (empty plot + lyric "—").

### 11.4 Gold note (R10)
- Toggle `t`: `":"` ↔ `"*"`. (Sung notes only.)

### 11.5 Edit lyric (R7)
- Double-click `N` (or toolbar **Lyric** button / key `L`) opens a popover `<input>` anchored near
  the note, prefilled with `y`.
- On Enter / blur: `pushUndo()`, set `syl` to the input value (trimmed of accidental leading/trailing
  **double** spaces is NOT applied — the value is taken verbatim to respect semantics).
- Popover hint text: `leading space = new word · trailing space = end word · ~ = unvoiced`.
- Escape / outside-click cancels (no undo push).

### 11.6 Keyboard map
| Key | Action |
|-----|--------|
| `Space` | Play/stop the current line (R13). |
| `Ctrl+Z` | Undo (R11). |
| `←` / `→` (or `PgUp`/`PgDn`) | Previous / next line (R3). |
| `S` | Split selected. |
| `M` | Merge selected with next. |
| `G` | Toggle gold. |
| `L` | Edit lyric of selected. |
| `Delete` / `Backspace` | Delete selected. |
| `Esc` | Close popover / deselect / stop playback. |

(Keyboard handlers ignore key events while the lyric popover input has focus, except `Esc`/`Enter`.)

---

## 12. Undo stack (R11)

- `undoStack: string[]` of serialized note arrays (`JSON.stringify(notes)`), **max 100** (shift
  oldest when full). **No redo.**
- `pushUndo()` is called **once per mutation**: at `pointerdown` for a drag gesture (so a whole
  drag is one undo step), and immediately before each discrete op (split/merge/delete/gold/lyric).
- `undo()`: pop the last snapshot, `notes = JSON.parse(snapshot)`, clear selection, re-render the
  current page. If the stack is empty, no-op.
- The stack is in-memory only (lost on reload) — acceptable for a tweak session; the downloaded
  `.txt` is the durable artifact.

---

## 13. Audio playback + MIDI + metronome (R13)

One shared `AudioContext` (created/resumed on first user gesture to satisfy autoplay policy).
The loaded vocals file is decoded once via `decodeAudioData` into an `AudioBuffer` (also the FFT
source, §8).

### 13.1 Line playback
- Play range: `line_start_ms = first_note.start_ms`, `line_end_ms = last_note.end_ms`.
- `Space` (when stopped): 
  ```
  ctx.resume(); t0ctx = ctx.currentTime + 0.05;
  src = ctx.createBufferSource(); src.buffer = buffer;
  src.start(t0ctx, line_start_ms / 1000);        // offset into the file
  scheduleNotes(t0ctx); scheduleMetronome(t0ctx);
  playing = true; rafLoop();
  ```
- `Space` (when playing) or `Esc`: stop all scheduled sources (`src.stop()`), cancel rAF,
  hide playhead, `playing = false`.
- `requestAnimationFrame` loop: `elapsed = ctx.currentTime - t0ctx`;
  `playhead_ms = line_start_ms + elapsed*1000`; draw playhead at `tx(playhead_ms)`; highlight the
  note whose `[start_ms, end_ms)` contains `playhead_ms`; when `elapsed >= (line_end_ms -
  line_start_ms)/1000`, auto-stop.
- **Click-to-seek** (stretch): clicking the timeline sets the playhead and (if playing) restarts
  from that offset. v1 may omit; playhead is display-only when not playing.

### 13.2 MIDI-like note tones
For each sung note in the page, schedule relative to `t0ctx`:
```
f     = 440 * 2 ** ((note.pitch - 69) / 12)
start = t0ctx + (note.start_ms - line_start_ms) / 1000
dur   = note.dur_ms / 1000
osc   = ctx.createOscillator(); osc.type = "triangle"; osc.frequency.value = f
gain  = ctx.createGain()
peak  = 0.16
gain.gain.setValueAtTime(0.0001, start)
gain.gain.linearRampToValueAtTime(peak, start + 0.015)      # 15 ms attack
gain.gain.setValueAtTime(peak, start + dur - 0.06)
gain.gain.linearRampToValueAtTime(0.0001, start + dur)      # 60 ms release
osc.connect(gain).connect(masterGain); osc.start(start); osc.stop(start + dur + 0.02)
```
- `masterGain` (→ destination) at `0.9` to leave headroom with the audio + metronome.
- Gold notes may be slightly brighter (stretch: `peak 0.20`); v1 uses the same `peak`.

### 13.3 Beat metronome
- Beats anchored by `#GAP` on the **quarter-note pulse** (`pulseMs = 60000/bpm`, 4× the `.txt`
  beat unit) so clicks land on the visible grid lines: pulse `n` is at `gap + n*pulseMs` (absolute).
  Within `[line_start_ms, line_end_ms]`, for each such pulse `n`:
  ```
  t     = t0ctx + (beat_time_ms - line_start_ms) / 1000
  down  = (n % 4 == 0)
  freq  = down ? 1320 : 880
  gain  = down ? 0.22 : 0.14
  # short click: oscillator, 25 ms, fast exponential decay
  ```
- Downbeat (quarter-note) is accented (higher frequency + gain), matching the SVG's stronger
  downbeat grid line.

### 13.4 Reference display
- A transport line under the toolbar shows: `Line {i}/{N}`, current playhead time
  (`ms_to_sec`), selected note (`pitch_to_note`, `start`, `dur`), BPM, GAP, beat counter.

---

## 14. Save / download (R1)

- **Download .txt** button (and `Ctrl+S`): serialize the current `notes` + `rawHeader` and trigger
  a browser download via `Blob` + `<a download>`.
- Filename: the original `.txt` basename (so re-downloading overwrites the working copy). The
  browser may append `(1)` on collision — user's choice.
- Encoding: **UTF-8** (`Blob` default). Source files read via `read_text_fallback` (utf-8/cp1252/
  latin-1) are re-emitted as UTF-8 — safe, and an improvement for non-ASCII.

### 14.1 Serialization format (must match `build_ultrastar_txt` body)
```
serialize_editor_txt(rawHeader, notes) -> str:
    header = "\n".join(rawHeader)
    body = []
    for n in notes:
        if n.type == "-":
            body.append(f"- {n.start}")
        else:
            body.append(f"{n.type} {n.start} {n.dur} {n.pitch} {n.syl}")
    return f"{header}\n\n" + "\n".join(body) + "\nE\n"
```
- This is implemented **in Python** (`cli/editor.py`) and unit-tested; the JS mirror uses the same
  exact f-string shapes so downloads are byte-consistent with the tested Python output.
- `rawHeader` is emitted **verbatim** (R1 lossless header). The body uses the canonical one-space
  separator, identical to `build_ultrastar_txt` (`cli/ultrastar.py:68-73`).

---

## 15. Layout & page navigation (R3)

- **Header bar** (top, full width): song `TITLE — ARTIST`, `BPM`, `GAP`, note/line counts, and a
  "load audio" status chip.
- **Audio loader** (shown until audio is ready): a drop zone + `<input type=file accept="audio/*">`
  with the text `Load the vocals audio to enable the FFT background + playback. Suggested: {audioHint}`.
  If `audioB64` is present (`--embed-audio`), decode it automatically and skip this step.
- **Toolbar** (per session, above the page):
  `[◀ Prev] [Next ▶]  [Play/Stop]  |  [Split] [Merge] [Delete] [Gold] [Lyric]  |  [Undo]  |  [Download .txt]`
  + line indicator `Line {i}/{N}` + selection readout. Contextual buttons disable when no valid
  selection (e.g., Merge disabled if next is a line break; Split disabled if `dur < 2`).
- **Page**: the canvas+svg stack (§6/§8/§9) + the annotated lyric line (§9.9) + transport (§13.4).
- **Navigation**: Prev/Next buttons + `←`/`→`; switching pages recomputes the window, (re)builds
  the FFT (cached), resets the playhead, and re-renders the SVG.
- Dark theme matching the preview (`body #1a1a2e`, panels `#16213e`, accent `#e94560`/`#4fc3f7`).

---

## 16. File layout

### New
| File | Purpose |
|------|---------|
| `cli/editor.py` | Editor generation: raw-header extraction, full `pitchWords` embedding (s→ms), payload build, magma-LUT + template injection, optional base64 audio embed, `serialize_editor_txt`. |
| `cli/editor_template.html` | The SPA: inlined CSS + JS (layout, FFT/STFT + radix-2 FFT, canvas background, SVG overlay, drag/split/merge/delete/gold/lyric, undo, Web Audio playback + MIDI + metronome, download). |
| `cli/tests/test_editor.py` | Python-pure tests (see §18). |

### Modified
| File | Change |
|------|--------|
| `cli/__main__.py` | Add `edit` subparser + `_cmd_edit`. |
| `AGENTS.md` | Document the `edit` subcommand, new files, and the editor's data flow. |
| `docs/note_editor_plan.md` | This plan (new). |

No new Python dependencies (editor.py is stdlib + existing `cli` modules). The SPA uses only
vanilla JS + Web Audio (no framework, no CDN).

---

## 17. `cli/editor.py` — function signatures

```python
def extract_raw_header(text: str) -> tuple[list[str], str]:
    """Leading '#' lines (verbatim) + remainder body text."""

def load_pitch_words(pitch_path: Path) -> list[dict]:
    """Load whisperx_pitch.json words; [] if path is None/missing. Normalize keys to
    {word, start_ms, end_ms, midi, frames:[{t_ms,m,conf,amp}]} (seconds → ms).
    The FULL list is embedded as `pitchWords`; the JS slices it into the live page window."""

def build_payload(txt_path, pitch_path, audio_hint) -> dict:
    """Assemble the full EDITOR_DATA dict (§5.1)."""

def serialize_editor_txt(raw_header: list[str], notes: list[UltrastarNote]) -> str:
    """Header + note body + 'E' (§14.1)."""

def embed_audio_b64(path: Path) -> tuple[str, str]:
    """(base64, mime) for --embed-audio; raises FileNotFoundError if missing."""

def generate_editor(txt_path: Path, output_html: Path | None,
                    pitch_json_path: Path | None, vocals_hint: Path | None,
                    embed_audio: bool) -> Path:
    """Generate the self-contained HTML; returns the written path."""
```

Reuse from existing modules: `read_text_fallback`, `parse_ultrastar_txt`, `UltrastarNote`,
`UltrastarMeta` (`cli/ultrastar.py`). `pitch_to_note`, `amplitude_color`, `confidence_color`,
`NOTE_NAMES` (`cli/html_preview.py`) are mirrored (not imported) in the JS, as are the FFT
parameter constants from `cli/align.py` (`_FFT_WINDOW=2048`, `_FFT_HOP=512`).

---

## 18. Tests — `cli/tests/test_editor.py` (Python-pure)

The JS (FFT, drag, undo, Web Audio, download) is **not** unit-tested in Python; it is covered by
a Node DOM-stub smoke harness (init render, FFT row alignment on a synthetic 440 Hz buffer,
drag/split/undo/download) plus the manual smoke test (§21). The Python side is fully tested:

1. **`extract_raw_header`**
   - Splits header vs body at the first non-`#` line.
   - Preserves `#COVER`/`#BACKGROUND`/`#CREATOR` and exact formatting (no strip).
   - Handles a file with no blank line between header and body, and CRLF line endings.
2. **`load_pitch_words`**
    - Seconds → ms conversion for word start/end and per-frame `t_ms`.
    - `None` path and missing file both return `[]`.
3. **ms↔beat at file BPM**
   - `start_ms = gap + start*beatMs` matches `html_preview.beats_to_ms` for sample notes.
    - `beatMs = 60000/bpm/4` with a `beat_resolution_multiplier`-scaled BPM (e.g., 242.6 → 61.83 ms/beat; `pulseMs` → 247.32 ms).
4. **`serialize_editor_txt` round-trip**
   - `serialize(parse(file))` re-parses to **identical** notes (type/start/dur/pitch/syl incl. spaces).
   - **Idempotent**: `serialize(parse(serialize(parse(f)))) == serialize(parse(f))`.
   - Header emitted verbatim (incl. non-standard tags) and followed by exactly one blank line + `E`.
   - Gold notes (`*`) and line breaks (`-`) survive.
5. **`build_payload` shape**
    - Contains `srcName`, `rawHeader`, `meta`, `bpm`, `gap`, `beatMs`, `pulseMs`, `notes`,
      `pitchWords`, `audioHint`, `audioB64`.
    - `pitchWords` is the full converted word list when `--pitch` is given, `[]` when not.
    - `notes[i].start/dur/pitch` are ints; `-` notes have `dur 0 pitch 0`.
6. **`generate_editor`**
   - Writes an HTML file containing the `EDITOR_DATA` JSON (parseable back) and the `audioHint`.
   - Default output name `<stem>_editor.html` next to the `.txt`; `--output` honored.
   - `--embed-audio` embeds a non-empty `audioB64` + correct `audioMime`; without `--vocals` it errors.
   - Uses a real sample: the committed `test_song_reference_ultrastar_file.txt`.

---

## 19. Edge cases & risks

| Risk | Mitigation |
|------|-----------|
| Canvas/SVG misalignment (off-by-padding) | Single shared layout-constants object; both layers use identical `tx/ty` and origin (§6, §8.5). Verified visually in smoke test. |
| FFT row ↔ MIDI row mismatch | MIDI-binned energy (§8.3) guarantees exact row alignment; no interpolation between layers. |
| `#BPM` already scaled by `beat_resolution_multiplier` | Use the file's `#BPM` directly with the 16th-note `beatMs`; no double scaling (§5.1). |
| Non-UTF-8 source (cp1252 umlauts) | `read_text_fallback` on load; UTF-8 on download (safe, documented §14). |
| Header loss on save | Raw-header verbatim preservation (§5.2, §14) instead of `build_ultrastar_txt`. |
| Notes overlapping after edits | Every drag/merge clamps to neighbor boundaries (§10, §11); min duration 1 beat. |
| Autoplay policy blocks AudioContext | `ctx.resume()` on the spacebar/user gesture (§13.1). |
| Large songs / memory | Only the visible page's FFT is computed (cached); one decoded `AudioBuffer` (mono, a few MB). |
| `--pitch` JSON from a different run than the `.txt` | Documented: pass the matching `tmp/<stem>_…whisperx_pitch.json`. Mismatched overlays are still time-correct, just less meaningful. |
| Line spans > 2 octaves | Window expands to fit (documented exception, §7). |
| Split syllable ambiguity | Second half defaults to `~` (unvoiced) for the user to relabel (§11.1). |
| JS not unit-tested | Python-pure logic (serialization, parsing, payload) is tested; JS verified by the Node DOM-stub harness + smoke test (§21). |
| `decodeAudioData` fails on some encodings | Vocals stem is MP3 (universally decodable); show a clear error banner on failure. |

---

## 20. Build order

1. **`cli/editor.py` core** — `extract_raw_header`, `load_pitch_words`, `build_payload`
   (full `pitchWords` embedding), `serialize_editor_txt`, `embed_audio_b64`, `generate_editor`
   (magma-LUT + `__EDITOR_DATA__` injection) + the `edit` subcommand wiring.
   **Tests: header, pitch-words, ms↔beat, round-trip, payload, generation.**
2. **`cli/editor_template.html` render** — layout constants, `tx/ty`, page window + MIDI window,
   SVG overlay (grid/axes/beat grid/note bars/whisper labels/pitch dots/annotated lyric),
   audio loader, page navigation. (Static render of the first page, no editing yet.)
3. **FFT background** — radix-2 FFT + STFT + MIDI binning + magma LUT + canvas alignment.
4. **Interactions** — selection, drag move/resize (both edges), snapping + clamps, live update.
5. **Edit ops + undo** — split/merge/delete/gold/lyric popover, undo stack, keyboard map.
6. **Audio** — decode, line playback, MIDI tones, metronome, playhead + active-note highlight.
7. **Save** — download `.txt` (Python-tested serializer mirrored in JS).
8. **Polish + docs** — transport readout, theme, Node DOM-stub smoke harness (§21), AGENTS.md,
   full `pytest cli/tests/`.

Each step is independently smoke-testable against the committed test song.

---

## 21. Smoke test

### 21a. Automated (Node DOM-stub harness)

The app script is extracted from the generated HTML and executed under Node with DOM/Web Audio
stubs, then driven programmatically (no browser needed). It asserts:
- init render (note bars, whisper labels, pitch dots, playhead, lyric line, line/notes chips);
- audio decode + **FFT row alignment**: a synthetic 440 Hz buffer lights the MIDI-69 row, not the
  floor (catches the canvas-row inversion), stretched to the full plot;
- drag a note → download → the serialized `.txt` reflects the move; undo → reverts;
- re-select + split → `dur/2` halves, second half `~`;
- Space play/stop (Web Audio stubbed) does not throw; page navigation updates the line indicator.

Run it by extracting `<script>` #2 from the generated HTML to a `.js` file and `node harness.js app.js`.

### 21b. Manual (browser)

```
python -m cli edit --txt test_song_reference_ultrastar_file.txt \
    --pitch tmp/whisperx_pitch.json \
    --vocals tmp/test_song_full_audio_vocals.mp3 \
    --output tmp/editor_smoke.html
```
1. Open `tmp/editor_smoke.html`; load `tmp/test_song_full_audio_vocals.mp3` in the drop zone.
2. Verify page 1 renders: FFT background aligned (a C4 row lines up with the MIDI-60 grid line),
   note bars, pitch dots, whisper labels, beat grid, `[time] word` lyric line.
3. Navigate lines with `→`; confirm the FFT recomputes and the window stays ~2 octaves.
4. Drag a note up/down and left/right; confirm snapping to beats/semitones and no overlap.
5. Drag the right edge; confirm min 1 beat and no crossing the next note.
6. Double-click a note; change its syllable; confirm the lyric line updates.
7. Split a long note (playhead/midpoint); confirm two notes, second = `~`.
8. Merge the two; confirm combined span, first pitch, concatenated syllable.
9. Toggle gold on a note; confirm color flips to the chorus palette.
10. Delete a note; confirm removal.
11. `Ctrl+Z` repeatedly; confirm each step reverts (and stops at the initial state).
12. `Space`; confirm audio plays from the line, MIDI tones track the notes, metronome clicks on
    beats with accented downbeats, and the playhead + active-note highlight move; `Space` stops.
13. **Download .txt**; diff against the original with `python -m cli diff` (should be within
    tolerance) and confirm the header (incl. `#COVER`) is preserved byte-for-byte.

---

## 22. Acceptance criteria

- `python -m cli edit …` produces a self-contained HTML that opens with no server and no network.
- The page visually matches `html_preview` output for the same song (note bars, dots, labels, grid).
- All of R1–R13 work as specified; undo is reliable and capped; no note overlap is possible.
- Downloaded `.txt` re-parses to the edited notes with the original header preserved.
- `pytest cli/tests/` passes (existing + `test_editor.py`).
