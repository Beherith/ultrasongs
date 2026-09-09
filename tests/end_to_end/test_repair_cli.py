from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ultrasongs.domain.ultrastar import parse_ultrastar_file


@pytest.mark.e2e
def test_real_repair_cli(
    pytestconfig: pytest.Config,
    tmp_path: Path,
) -> None:
    """Run the same command users run, with explicit local MP3 and TXT fixtures."""

    audio_option = pytestconfig.getoption("--e2e-audio")
    song_option = pytestconfig.getoption("--e2e-song")
    if not audio_option or not song_option:
        pytest.skip("pass --e2e-audio and --e2e-song to run the real repair pipeline")

    audio = Path(audio_option).resolve()
    song = Path(song_option).resolve()
    assert audio.is_file(), f"missing --e2e-audio fixture: {audio}"
    assert song.is_file(), f"missing --e2e-song fixture: {song}"

    workspace = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "exports"
    command = [sys.executable, "-m", "ultrasongs", "repair"]
    config_option = pytestconfig.getoption("--e2e-config")
    if config_option:
        command.extend(["--config", str(Path(config_option).resolve())])
    command.extend(
        [
            "--audio",
            str(audio),
            "--song",
            str(song),
            "--output-dir",
            str(output_root),
            "--json",
        ]
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(workspace / "src"), environment.get("PYTHONPATH", "")])
    )
    environment["ULTRASONGS_PATHS__PROJECTS_DIR"] = str(tmp_path / "projects")
    environment["ULTRASONGS_PATHS__TEMP_DIR"] = str(tmp_path / "work")
    environment["ULTRASONGS_PATHS__EXPORTS_DIR"] = str(output_root)
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=7_200,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    bundles = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    bundle = bundles[0]
    scores_path = next(bundle.glob("*-scores.json"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["similarity"]["matched_notes"] > 0
    assert Path(scores["artifacts"]["reference_song"]).read_bytes() == song.read_bytes()
    assert parse_ultrastar_file(scores["artifacts"]["updated_song"]).notes

    report = Path(scores["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "Final candidate" in report
    assert "Reference song" in report
    assert "Similarity score" in report
    with zipfile.ZipFile(scores["artifacts"]["archive"]) as archive:
        assert any(name.lower().endswith(".txt") for name in archive.namelist())

    manifests = list((tmp_path / "projects").glob("prj_*/manifests/run_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["stages"]
    assert all(stage["status"] == "succeeded" for stage in manifest["stages"])
