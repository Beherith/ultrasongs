from __future__ import annotations

import json
from html.parser import HTMLParser

from ultrasongs.domain.reporting import build_pipeline_report
from ultrasongs.domain.scoring import compare_songs
from ultrasongs.domain.ultrastar import parse_ultrastar_text
from ultrasongs.domain.validation import ValidationOutcome


class SectionCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.section_ids: set[str] = set()

    def handle_starttag(self, tag, attrs) -> None:
        values = dict(attrs)
        if tag == "section" and values.get("id"):
            self.section_ids.add(values["id"])


def _song(title: str, body: str):
    return parse_ultrastar_text(
        f"#TITLE:{title}\n#ARTIST:Test artist\n#MP3:test.mp3\n#BPM:120\n#GAP:100\n{body}\nE\n",
        strict=True,
    )


def test_combined_report_contains_every_pipeline_section_and_embedded_score() -> None:
    reference = _song("Reference", ": 0 2 17 Hel\n: 2 2 19 lo\n- 5")
    candidate = _song("Candidate", ": 0 2 29 Hel\n: 3 2 21 lo\n- 6")
    score = compare_songs(reference, candidate)
    transcription = {
        "done": True,
        "language": "en",
        "words": [
            {
                "word": "Hello",
                "start": 0.1,
                "end": 0.5,
                "midi": 53,
                "pitchFrames": [
                    {"time": 0.1, "midi": 53, "confidence": 0.25},
                    {"time": 0.2, "midi": 54, "confidence": 0.9},
                ],
            }
        ],
        "pauses": [{"start": 0.6, "end": 1.0}],
    }

    report = build_pipeline_report(
        reference=reference,
        candidate=candidate,
        transcription=transcription,
        similarity=score,
        effective_config={"pitch": {"confidence": 0.2}},
    )

    collector = SectionCollector()
    collector.feed(report)
    assert collector.section_ids == {
        "summary",
        "similarity",
        "intermediate",
        "candidate",
        "reference",
        "configuration",
    }
    assert "Pitch frames over time" in report
    assert "UltraStar notes over time" in report
    assert "Timing RMSE" in report
    score_payload = report.split('<script id="similarity-data" type="application/json">', 1)[
        1
    ].split("</script>", 1)[0]
    assert json.loads(score_payload)["matched_notes"] == 2


def test_report_escapes_all_uploaded_text_and_script_json() -> None:
    attack = '</script><script>alert("bad")</script>'
    candidate = _song(attack, f": 0 2 17 {attack}")
    transcription = {
        "language": attack,
        "words": [{"word": attack, "start": 0, "end": 1, "midi": 17, "pitchFrames": []}],
    }

    report = build_pipeline_report(candidate=candidate, transcription=transcription)

    assert report.count("<script") == 1
    assert attack not in report
    assert "&lt;/script&gt;" in report


def test_report_shows_validation_pass_fail_and_reasons() -> None:
    song = _song("Song", ": 0 2 17 hi")
    score = compare_songs(song, song)
    outcome = ValidationOutcome(False, ("Timing exceeded the threshold",), score)

    report = build_pipeline_report(
        candidate=song,
        reference=song,
        similarity=score,
        validation_outcome=outcome,
    )

    assert 'id="validation"' in report
    assert "Validation FAILED" in report
    assert "Timing exceeded the threshold" in report


def test_missing_optional_artifacts_produce_useful_empty_sections() -> None:
    report = build_pipeline_report(title="Empty run")
    assert "No transcription artifact was supplied" in report
    assert "No final candidate was supplied" in report
    assert "No reference song was supplied" in report
    assert "No reference comparison was supplied" in report
    assert (
        json.loads(
            report.split('<script id="similarity-data" type="application/json">', 1)[1].split(
                "</script>", 1
            )[0]
        )
        is None
    )


def test_report_options_can_hide_pitch_frames_and_pauses() -> None:
    transcription = {
        "words": [
            {
                "word": "hi",
                "start": 0,
                "end": 1,
                "midi": 60,
                "pitchFrames": [{"time": 0.5, "midi": 60, "confidence": 0.9}],
            }
        ],
        "pauses": [{"start": 1, "end": 2}],
    }

    report = build_pipeline_report(
        transcription=transcription,
        report_options={"include_pitch_frames": False, "include_pauses": False},
    )

    assert "1 words, 0 pitch frames" in report
    assert "0 detected pauses" in report
    assert "confidence 0.90" not in report


def test_legacy_ultrastar_wrapper_writes_combined_report(tmp_path) -> None:
    from ultrastar_to_html import main

    source = tmp_path / "song.txt"
    output = tmp_path / "song.html"
    source.write_text("#TITLE:Wrapper\n#BPM:120\n: 0 2 17 hi\nE\n", encoding="utf-8")

    assert main([str(source), str(output)]) == 0
    rendered = output.read_text(encoding="utf-8")
    assert "Final candidate" in rendered
    assert "Wrapper" in rendered


def test_legacy_pitch_wrapper_writes_combined_report(tmp_path) -> None:
    from pitch_to_html import main

    source = tmp_path / "pitch.json"
    output = tmp_path / "pitch.html"
    source.write_text(
        json.dumps(
            {"words": [{"word": "hi", "start": 0, "end": 1, "midi": 17, "pitchFrames": []}]}
        ),
        encoding="utf-8",
    )

    assert main([str(source), str(output)]) == 0
    assert "Intermediate transcription and pitch" in output.read_text(encoding="utf-8")
