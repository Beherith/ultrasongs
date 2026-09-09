"""Tests for the self-contained HTML note editor generator (cli/editor.py)."""

import base64
import json
import re
from pathlib import Path

import pytest

from cli.editor import (
    build_payload,
    embed_audio_b64,
    extract_raw_header,
    generate_editor,
    load_pitch_words,
    note_to_payload,
    serialize_editor_txt,
)
from cli.ultrastar import parse_ultrastar_txt

_TEMPLATE = Path(__file__).resolve().parents[1] / "editor_template.html"

HEADER = (
    "#TITLE:Test Song\n"
    "#ARTIST:Tester\n"
    "#MP3:test.mp3\n"
    "#COVER:test.jpg\n"
    "#BACKGROUND:test.jpg\n"
    "#BPM:120.00\n"
    "#GAP:500\n"
    "#CREATOR:Me\n"
)
BODY = (
    "\n"
    ": 0 4 60 hello\n"
    ": 4 4 62  world\n"
    "- 10\n"
    ": 12 4 64 how\n"
    "* 16 4 66  are\n"
    "- 22\n"
    ": 24 4 68  you\n"
    "E\n"
)
CONTENT = HEADER + BODY


def _write_txt(tmp_path, content: str = CONTENT):
    p = tmp_path / "song.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _parse(text: str):
    return parse_ultrastar_txt(text)


class TestExtractRawHeader:
    def test_splits_at_first_non_header(self):
        raw, body = extract_raw_header(CONTENT)
        assert raw == HEADER.split("\n")[:-1]
        assert body.startswith("\n: 0 4 60 hello")

    def test_preserves_nonstandard_tags_verbatim(self):
        raw, _ = extract_raw_header(CONTENT)
        assert "#COVER:test.jpg" in raw
        assert "#BACKGROUND:test.jpg" in raw
        assert "#CREATOR:Me" in raw

    def test_no_strip_of_header_lines(self):
        # Header lines are detected via strip() but preserved VERBATIM (leading and
        # trailing whitespace are kept).
        raw, _ = extract_raw_header("  #TITLE: X\n#BPM:1\n: 0 1 60 a\nE\n")
        assert raw == ["  #TITLE: X", "#BPM:1"]
        raw2, _ = extract_raw_header("#TITLE: X \n#BPM:1\nE\n")
        assert raw2[0] == "#TITLE: X "

    def test_crlf_normalized_to_lf(self):
        raw, body = extract_raw_header("#TITLE:A\r\n#BPM:1\r\n: 0 1 60 a\r\nE\r\n")
        assert raw == ["#TITLE:A", "#BPM:1"]
        assert "\r" not in body

    def test_header_only_no_blank_line(self):
        raw, body = extract_raw_header("#TITLE:A\n#BPM:1\n: 0 1 60 a\nE\n")
        assert raw == ["#TITLE:A", "#BPM:1"]
        assert body.startswith(": 0 1 60 a")

    def test_empty_header(self):
        raw, body = extract_raw_header(": 0 1 60 a\nE\n")
        assert raw == []
        assert body == ": 0 1 60 a\nE\n"


class TestLoadPitchWords:
    def test_converts_seconds_to_ms(self, tmp_path):
        p = tmp_path / "pitch.json"
        p.write_text(
            json.dumps(
                {
                    "words": [
                        {
                            "word": "hi",
                            "start": 0.5,
                            "end": 1.0,
                            "midi": 60,
                            "pitchFrames": [
                                {"time": 0.6, "midi": 60, "confidence": 0.8, "amplitude": 0.4}
                            ],
                        }
                    ],
                    "done": True,
                }
            ),
            encoding="utf-8",
        )
        words = load_pitch_words(p)
        assert words[0]["start_ms"] == pytest.approx(500.0)
        assert words[0]["end_ms"] == pytest.approx(1000.0)
        assert words[0]["frames"][0]["t_ms"] == pytest.approx(600.0)
        assert words[0]["frames"][0]["amp"] == pytest.approx(0.4)

    def test_none_returns_empty(self):
        assert load_pitch_words(None) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_pitch_words(tmp_path / "nope.json") == []


class TestBuildPayload:
    def test_shape_and_values(self, tmp_path):
        payload = build_payload(_write_txt(tmp_path), None, "")
        assert payload["bpm"] == 120.0
        assert payload["gap"] == 500
        assert payload["beatMs"] == pytest.approx(125.0)
        assert payload["pulseMs"] == pytest.approx(500.0)
        assert payload["meta"]["title"] == "Test Song"
        assert payload["meta"]["video"] is None
        assert payload["rawHeader"][3] == "#COVER:test.jpg"
        assert len(payload["notes"]) == 7
        # No --pitch given -> empty full word list (JS slices it per rendered window)
        assert payload["pitchWords"] == []
        # Line-break notes carry zero duration/pitch
        dash = next(n for n in payload["notes"] if n["type"] == "-")
        assert dash["dur"] == 0 and dash["pitch"] == 0

    def test_pitch_words_embedded_full(self, tmp_path):
        txt = _write_txt(tmp_path)
        pitch = tmp_path / "pitch.json"
        pitch.write_text(
            json.dumps(
                {
                    "words": [
                        {
                            "word": "hello",
                            "start": 0.5,
                            "end": 1.0,
                            "midi": 60,
                            "pitchFrames": [
                                {"time": 0.6, "midi": 60, "confidence": 0.8, "amplitude": 0.4}
                            ],
                        }
                    ],
                    "done": True,
                }
            ),
            encoding="utf-8",
        )
        payload = build_payload(txt, pitch, "")
        assert len(payload["pitchWords"]) == 1
        assert payload["pitchWords"][0]["start_ms"] == pytest.approx(500.0)
        assert payload["pitchWords"][0]["frames"][0]["t_ms"] == pytest.approx(600.0)

    def test_audio_hint_basename(self, tmp_path):
        payload = build_payload(_write_txt(tmp_path), None, "C:/some/dir/vocals.mp3")
        assert payload["audioHint"] == "vocals.mp3"


class TestBuildPayloadSyllableSpaces:
    def test_preserves_syllable_word_spaces(self, tmp_path):
        txt = _write_txt(tmp_path, "#BPM:120\n\n: 0 4 60 he \n: 4 4 62  llo\nE\n")
        payload = build_payload(txt, None, "")
        assert payload["notes"][0]["syl"] == "he "
        assert payload["notes"][1]["syl"] == " llo"


class TestBuildPayloadPitchLift:
    def test_lifts_notes_below_c2_by_whole_octaves(self, tmp_path):
        txt = _write_txt(
            tmp_path,
            "#TITLE:Low\n#ARTIST:A\n#MP3:a.mp3\n#BPM:120\n#GAP:0\n"
            "\n: 0 4 12 a\n: 4 4 21 b\n: 8 4 35 c\n: 12 4 36 d\nE\n",
        )
        payload = build_payload(txt, None, "")
        assert [n["pitch"] for n in payload["notes"]] == [36, 45, 47, 36]

    def test_rest_notes_keep_zero_pitch(self, tmp_path):
        txt = _write_txt(tmp_path, "#BPM:120\n\n: 0 4 12 a\n- 8\n: 12 4 36 b\nE\n")
        payload = build_payload(txt, None, "")
        dash = next(n for n in payload["notes"] if n["type"] == "-")
        assert dash["pitch"] == 0 and dash["dur"] == 0
        assert [n["pitch"] for n in payload["notes"] if n["type"] != "-"] == [36, 36]

    def test_js_loader_preserves_word_spaces_and_lifts_low_pitch(self, tmp_path):
        html = generate_editor(_write_txt(tmp_path)).read_text(encoding="utf-8")
        parse_body = html.split("function parseUltrastarTxt(content){", 1)[1].split("\n  }", 1)[0]
        assert ".exec(line)" in parse_body
        assert ".exec(trimmed)" not in parse_body
        assert "minSingPitch=36" in parse_body
        assert "while(n.pitch<minSingPitch) n.pitch+=12" in parse_body


class TestSerializeEditorTxt:
    def test_round_trip_preserves_notes(self):
        meta, notes = _parse(CONTENT)
        raw, _ = extract_raw_header(CONTENT)
        payload_notes = [note_to_payload(n, i) for i, n in enumerate(notes)]
        serialized = serialize_editor_txt(raw, payload_notes)
        meta2, notes2 = _parse(serialized)
        assert len(notes2) == len(notes)
        for a, b in zip(notes, notes2):
            assert (a.note_type, a.start_beat, a.duration, a.pitch, a.syllable) == (
                b.note_type, b.start_beat, b.duration, b.pitch, b.syllable,
            )
        assert meta2.bpm == meta.bpm and meta2.gap == meta.gap

    def test_header_preserved_verbatim(self):
        raw, _ = extract_raw_header(CONTENT)
        payload_notes = [note_to_payload(n, i) for i, n in enumerate(_parse(CONTENT)[1])]
        serialized = serialize_editor_txt(raw, payload_notes)
        raw2, _ = extract_raw_header(serialized)
        assert raw2 == raw
        assert "#COVER:test.jpg" in raw2 and "#CREATOR:Me" in raw2

    def test_idempotent(self):
        raw, _ = extract_raw_header(CONTENT)
        payload_notes = [note_to_payload(n, i) for i, n in enumerate(_parse(CONTENT)[1])]
        once = serialize_editor_txt(raw, payload_notes)
        raw2, _ = extract_raw_header(once)
        payload_notes2 = [note_to_payload(n, i) for i, n in enumerate(_parse(once)[1])]
        twice = serialize_editor_txt(raw2, payload_notes2)
        assert once == twice

    def test_round_trip_preserves_trailing_word_space(self):
        text = "#BPM:120\n\n: 0 4 60 he \n: 4 4 62 llo\nE\n"
        raw, _ = extract_raw_header(text)
        payload_notes = [note_to_payload(n, i) for i, n in enumerate(_parse(text)[1])]
        serialized = serialize_editor_txt(raw, payload_notes)
        assert ": 0 4 60 he " in serialized
        _, notes2 = _parse(serialized)
        assert [n.syllable for n in notes2] == ["he ", "llo"]

    def test_gold_and_line_break_survive(self):
        raw, _ = extract_raw_header(CONTENT)
        payload_notes = [note_to_payload(n, i) for i, n in enumerate(_parse(CONTENT)[1])]
        serialized = serialize_editor_txt(raw, payload_notes)
        assert "* 16 4 66  are" in serialized
        assert "- 10" in serialized and "- 22" in serialized


class TestEmbedAudioB64:
    def test_returns_b64_and_mime(self, tmp_path):
        p = tmp_path / "v.mp3"
        p.write_bytes(b"fake-mp3")
        b64, mime = embed_audio_b64(p)
        assert b64 == base64.b64encode(b"fake-mp3").decode("ascii")
        assert mime == "audio/mpeg"

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            embed_audio_b64(tmp_path / "nope.mp3")


class TestGenerateEditor:
    def test_writes_default_name(self, tmp_path):
        txt = _write_txt(tmp_path)
        out = generate_editor(txt)
        assert out == txt.with_name("song_editor.html")
        assert out.exists()
        assert txt.with_name("song_editor.json").exists()

    def test_substitutes_data(self, tmp_path):
        txt = _write_txt(tmp_path)
        out = generate_editor(txt)
        html = out.read_text(encoding="utf-8")
        assert "__EDITOR_DATA__" not in html
        assert "__MAGMA__" not in html
        assert "EDITOR_DATA" in html
        assert "Test Song" in html
        assert "#COVER:test.jpg" in html
        m = re.search(r"window\.EDITOR_DATA = \(function \(\) \{", html)
        assert m, "EDITOR_DATA script block not found"
        mj = re.search(r"try \{ return (.*); \}\n\s*catch", html)
        assert mj, "embedded EDITOR_DATA payload not found"
        data = json.loads(mj.group(1).replace("<\\/", "</"))
        json_data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert json_data == data
        assert data["meta"]["title"] == "Test Song"
        assert data["pitchWords"] == []
        m2 = re.search(r"const MEGA=(\[\[.*?\]\]);", html, re.DOTALL)
        assert m2, "magma LUT not embedded"
        assert len(json.loads(m2.group(1))) == 64

    def test_template_standalone_fallback(self):
        tpl = _TEMPLATE.read_text(encoding="utf-8")
        assert "window.EDITOR_DATA = (function () {" in tpl
        assert "try { return __EDITOR_DATA__; }" in tpl
        fallback = tpl.split("catch (e) {", 1)[1].split("})();", 1)[0]
        assert "title: \"(empty)\"" in fallback
        assert "bpm: 120" in fallback and "gap: 0" in fallback
        assert "notes: []" in fallback and "rawHeader: []" in fallback
        assert 'audioB64: ""' in fallback

    def test_double_click_edits_lyrics_with_native_event(self, tmp_path):
        html = generate_editor(_write_txt(tmp_path)).read_text(encoding="utf-8")
        assert 'svgEl.addEventListener("dblclick",e=>' in html
        assert 'const id=g?+g.getAttribute("data-id"):selectedId;' in html
        assert "openLyricPopover(note);" in html
        assert 'e.pointerType==="touch"&&lastTap' in html
        pointerdown = html.split('svgEl.addEventListener("pointerdown",e=>', 1)[1].split(
            'svgEl.addEventListener("pointermove",e=>', 1
        )[0]
        assert "updateSelectionUi();" in pointerdown
        assert "renderPage();" not in pointerdown

    def test_add_remove_line_buttons(self, tmp_path):
        html = generate_editor(_write_txt(tmp_path)).read_text(encoding="utf-8")
        assert 'id="addLineBtn"' in html
        assert 'id="delLineBtn"' in html
        assert "function addLine(){" in html
        assert "function removeLine(){" in html
        assert 'syl:"New Line"' in html
        assert 'document.getElementById("addLineBtn").onclick=addLine;' in html
        assert 'document.getElementById("delLineBtn").onclick=removeLine;' in html

    def test_load_vocals_button(self, tmp_path):
        html = generate_editor(_write_txt(tmp_path)).read_text(encoding="utf-8")
        assert 'id="vocalsBtn"' in html
        assert 'id="vocalsInput"' in html
        assert 'document.getElementById("vocalsBtn").onclick=()=>vocalsInput.click();' in html
        assert 'vocalsInput.onchange=e=>{ onFile(e.target.files[0]); e.target.value=""; };' in html

    def test_output_path_honored(self, tmp_path):
        txt = _write_txt(tmp_path)
        target = tmp_path / "sub" / "custom.html"
        out = generate_editor(txt, output_html=target)
        assert out == target and out.exists()
        assert target.with_suffix(".json").exists()

    def test_json_output_path_honored(self, tmp_path):
        txt = _write_txt(tmp_path)
        target = tmp_path / "custom" / "payload.json"
        generate_editor(txt, output_json=target)
        assert json.loads(target.read_text(encoding="utf-8"))["meta"]["title"] == "Test Song"

    def test_embed_audio(self, tmp_path):
        txt = _write_txt(tmp_path)
        voc = tmp_path / "vocals.mp3"
        voc.write_bytes(b"fake-mp3-bytes")
        out = generate_editor(txt, vocals_hint=voc, embed_audio=True)
        html = out.read_text(encoding="utf-8")
        assert base64.b64encode(b"fake-mp3-bytes").decode("ascii") in html
        assert "audio/mpeg" in html

    def test_embed_audio_requires_vocals(self, tmp_path):
        txt = _write_txt(tmp_path)
        with pytest.raises(ValueError):
            generate_editor(txt, embed_audio=True)

    def test_missing_txt_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_editor(tmp_path / "nope.txt")
