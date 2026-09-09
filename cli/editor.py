"""Generate a self-contained, in-browser Ultrastar note editor from a .txt file.

The editor is a single HTML file (no server, no build step) that visually matches
``cli/html_preview.py`` and lets the user hand-tune notes by drag/resize/split/merge/
delete/gold/lyric with an undo stack, a live FFT background, and audio + MIDI +
metronome playback. See ``docs/note_editor_plan.md``.
"""

import base64
import json
import mimetypes
from pathlib import Path

from cli.pipeline_types import UltrastarNote
from cli.ultrastar import parse_ultrastar_txt, read_text_fallback

_TEMPLATE_PATH = Path(__file__).parent / "editor_template.html"

# C2 (~65 Hz). The editor's pitch auto-zoom clamps its window, so notes below
# this are drawn off-plot; sung notes are shifted up by whole octaves at load.
MIN_SING_PITCH = 36


def lift_pitch_to_c2(pitch: int) -> int:
    """Shift a MIDI pitch up by whole octaves until it is at least C2 (36)."""
    if pitch >= MIN_SING_PITCH:
        return pitch
    return pitch + ((MIN_SING_PITCH - pitch + 11) // 12) * 12


def extract_raw_header(text: str) -> tuple[list[str], str]:
    """Split leading ``#`` header lines (verbatim) from the note body.

    Line endings are normalized to LF first so the header lines carry no stray
    carriage returns; the output is always re-serialized with LF. This preserves
    non-standard tags (``#COVER``, ``#BACKGROUND``, ``#CREATOR``, ...) verbatim,
    unlike ``build_ultrastar_txt`` which only knows the six standard tags.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip().startswith("#"):
        i += 1
    return lines[:i], "\n".join(lines[i:])


def note_to_payload(note: UltrastarNote, index: int) -> dict:
    """Convert a parsed note to the compact editor payload form."""
    return {
        "id": index,
        "type": note.note_type,
        "start": note.start_beat,
        "dur": note.duration,
        "pitch": note.pitch,
        "syl": note.syllable,
    }


def load_pitch_words(pitch_path: str | Path | None) -> list[dict]:
    """Load ``whisperx_pitch.json`` words, converting seconds to milliseconds.

    Returns ``[]`` when the path is ``None`` or the file is missing. Each word is
    ``{word, start_ms, end_ms, midi, frames:[{t_ms, m, conf, amp}]}``.
    """
    if pitch_path is None:
        return []
    path = Path(pitch_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[dict] = []
    for w in data.get("words", []):
        frames = [
            {
                "t_ms": pf.get("time", 0) * 1000.0,
                "m": pf.get("midi", 0),
                "conf": pf.get("confidence", 0.0),
                "amp": pf.get("amplitude", 0.0),
            }
            for pf in w.get("pitchFrames", [])
        ]
        words.append(
            {
                "word": w.get("word", ""),
                "start_ms": w.get("start", 0) * 1000.0,
                "end_ms": w.get("end", 0) * 1000.0,
                "midi": w.get("midi", 0),
                "frames": frames,
            }
        )
    return words


def build_payload(
    txt_path: str | Path,
    pitch_path: str | Path | None = None,
    audio_hint: str = "",
) -> dict:
    """Assemble the full ``EDITOR_DATA`` payload (see plan §5.1)."""
    text = read_text_fallback(Path(txt_path))
    raw_header, _ = extract_raw_header(text)
    meta, notes = parse_ultrastar_txt(text)
    audio_hint = Path(audio_hint).name if audio_hint else ""
    bpm = meta.bpm
    gap = meta.gap
    beat_ms = 60000.0 / bpm / 4
    pulse_ms = 60000.0 / bpm
    notes_payload = []
    for i, n in enumerate(notes):
        payload = note_to_payload(n, i)
        if n.note_type != "-":
            payload["pitch"] = lift_pitch_to_c2(n.pitch)
        notes_payload.append(payload)
    pitch_words = load_pitch_words(pitch_path)
    return {
        "srcName": Path(txt_path).name,
        "rawHeader": raw_header,
        "meta": {
            "title": meta.title,
            "artist": meta.artist,
            "mp3": meta.mp3,
            "bpm": bpm,
            "gap": gap,
            "video": meta.video,
        },
        "bpm": bpm,
        "gap": gap,
        "beatMs": beat_ms,
        "pulseMs": pulse_ms,
        "notes": notes_payload,
        "pitchWords": pitch_words,
        "audioHint": audio_hint or "",
        "audioB64": "",
        "audioMime": "",
    }


def serialize_editor_txt(raw_header: list[str], notes_payload: list[dict]) -> str:
    """Serialize the raw header + compact notes to an Ultrastar ``.txt`` string.

    The body is byte-identical in shape to ``build_ultrastar_txt``; the header is
    the caller's verbatim raw lines (preserving non-standard tags). This is the
    Python mirror of the JS download serializer (plan §14).
    """
    header = "\n".join(raw_header)
    body_lines = []
    for n in notes_payload:
        if n["type"] == "-":
            body_lines.append(f"- {n['start']}")
        else:
            body_lines.append(f"{n['type']} {n['start']} {n['dur']} {n['pitch']} {n['syl']}")
    body = "\n".join(body_lines)
    return f"{header}\n\n{body}\nE\n"


def embed_audio_b64(path: str | Path) -> tuple[str, str]:
    """Return ``(base64, mime)`` for ``--embed-audio``. Raises ``FileNotFoundError``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    data = p.read_bytes()
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "audio/mpeg"
    return base64.b64encode(data).decode("ascii"), mime


def generate_editor(
    txt_path: str | Path,
    output_html: str | Path | None = None,
    pitch_json_path: str | Path | None = None,
    vocals_hint: str | Path | None = None,
    embed_audio: bool = False,
    output_json: str | Path | None = None,
) -> Path:
    """Generate the editor HTML and its matching JSON payload; return the HTML path."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Ultrastar file not found: {txt_path}")

    audio_hint = Path(vocals_hint).name if vocals_hint else ""
    payload = build_payload(txt_path, pitch_json_path, audio_hint)
    if embed_audio:
        if not vocals_hint:
            raise ValueError("--embed-audio requires --vocals")
        b64, mime = embed_audio_b64(vocals_hint)
        payload["audioB64"] = b64
        payload["audioMime"] = mime

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__EDITOR_DATA__", data_json)

    out = Path(output_html) if output_html else txt_path.with_name(txt_path.stem + "_editor.html")
    json_out = Path(output_json) if output_json else out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    json_out.write_text(data_json + "\n", encoding="utf-8")
    return out
