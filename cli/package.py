"""Output packaging: write .txt, copy MP3/video, optionally create ZIP."""

import shutil
import zipfile
from pathlib import Path
from typing import Sequence

from cli.logging_setup import get_logger

logger = get_logger("cli.package")

# Intermediate file extensions included in the ZIP (skips .mp3 — already in
# the ZIP as title/vocals/accompaniment — and the note_segments_plots/ dir).
INTERMEDIATE_SUFFIXES = {".json", ".html", ".txt"}
INTERMEDIATE_SKIP_DIRS = {"note_segments_plots"}


def collect_intermediates(temp_path: Path, since_ts: float | None = None) -> list[Path]:
    """Collect intermediate temp files (JSON/HTML/TXT) for inclusion in the ZIP.

    With ``since_ts``, only files modified at or after ``since_ts - 2`` seconds
    are returned, so a shared temp dir does not leak files from other runs.
    """
    temp_path = Path(temp_path)
    if not temp_path.is_dir():
        return []
    cutoff = since_ts - 2 if since_ts is not None else None
    found: list[Path] = []
    for f in sorted(temp_path.rglob("*")):
        if not f.is_file():
            continue
        if any(part in INTERMEDIATE_SKIP_DIRS for part in f.relative_to(temp_path).parts[:-1]):
            continue
        if f.suffix.lower() not in INTERMEDIATE_SUFFIXES:
            continue
        if cutoff is not None and f.stat().st_mtime < cutoff:
            continue
        found.append(f)
    return found


def package_output(
    txt_content: str,
    mp3_path: Path,
    output_dir: Path,
    name: str,
    video_path: Path | None = None,
    vocals_path: Path | None = None,
    accompaniment_path: Path | None = None,
    extra_files: Sequence[Path] = (),
    intermediates_prefix: str = "intermediates",
) -> Path:
    """Package generated files into an output directory.

    Creates (all named after ``name``, typically ``<artist> - <title>``):
        output_dir/
            <name>.txt
            <name>.mp3     (copy of mp3_path)
            <name><ext>    (optional video, from video_path)
            <name>_vocals.mp3      (optional)
            <name>_accompaniment.mp3  (optional)
            <name>_editor.html, <name>_editor.json  (when already generated)
            <name>.zip     (ZIP of all above plus extra_files under intermediates/)

    Args:
        txt_content: The Ultrastar .txt string.
        mp3_path: Path to the MP3 file.
        output_dir: Directory to write output.
        name: Output base name (used for all filenames).
        video_path: Optional video file.
        vocals_path: Optional vocals stem.
        accompaniment_path: Optional accompaniment stem.
        extra_files: Optional intermediate files, written into the ZIP under
            intermediates/<name>.
        intermediates_prefix: ZIP directory for extra_files.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{name}.txt"
    txt_path.write_text(txt_content, encoding="utf-8")
    logger.info(f"Written: {txt_path}")

    mp3_out = output_dir / f"{name}.mp3"
    shutil.copy2(mp3_path, mp3_out)
    logger.info(f"Copied: {mp3_out}")

    if video_path and video_path.exists():
        video_out = output_dir / f"{name}{video_path.suffix.lower()}"
        shutil.copy2(video_path, video_out)
        logger.info(f"Copied: {video_out}")

    if vocals_path and Path(vocals_path).exists():
        vocals_out = output_dir / f"{name}_vocals.mp3"
        shutil.copy2(vocals_path, vocals_out)
        logger.info(f"Copied: {vocals_out}")

    if accompaniment_path and Path(accompaniment_path).exists():
        acc_out = output_dir / f"{name}_accompaniment.mp3"
        shutil.copy2(accompaniment_path, acc_out)
        logger.info(f"Copied: {acc_out}")

    # Create ZIP
    zip_path = output_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.iterdir():
            if f.suffix.lower() == ".zip":
                continue
            zf.write(f, f.name)
        for f in extra_files:
            f = Path(f)
            if f.is_file():
                zf.write(f, f"{intermediates_prefix}/{f.name}")
    logger.info(f"Created ZIP: {zip_path}")

    return output_dir
