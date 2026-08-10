"""Output packaging: write .txt, copy MP3/video, optionally create ZIP."""

import shutil
import zipfile
from pathlib import Path

from cli.logging_setup import get_logger

logger = get_logger("cli.package")


def package_output(
    txt_content: str,
    mp3_path: Path,
    output_dir: Path,
    title: str,
    video_path: Path | None = None,
    vocals_path: Path | None = None,
    accompaniment_path: Path | None = None,
) -> Path:
    """Package generated files into an output directory.

    Creates:
        output_dir/
            <title>.txt
            <title>.mp3     (copy of mp3_path)
            <title>.mp4     (optional, from video_path)
            vocals.mp3      (optional)
            accompaniment.mp3  (optional)
            <title>.zip     (ZIP of all above)

    Args:
        txt_content: The Ultrastar .txt string.
        mp3_path: Path to the MP3 file.
        output_dir: Directory to write output.
        title: Song title (used for filenames).
        video_path: Optional video file.
        vocals_path: Optional vocals stem.
        accompaniment_path: Optional accompaniment stem.

    Returns:
        Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{title}.txt"
    txt_path.write_text(txt_content, encoding="utf-8")
    logger.info(f"Written: {txt_path}")

    mp3_out = output_dir / f"{title}.mp3"
    shutil.copy2(mp3_path, mp3_out)
    logger.info(f"Copied: {mp3_out}")

    if video_path and video_path.exists():
        video_out = output_dir / video_path.name
        shutil.copy2(video_path, video_out)
        logger.info(f"Copied: {video_out}")

    if vocals_path and Path(vocals_path).exists():
        vocals_out = output_dir / "vocals.mp3"
        shutil.copy2(vocals_path, vocals_out)
        logger.info(f"Copied: {vocals_out}")

    if accompaniment_path and Path(accompaniment_path).exists():
        acc_out = output_dir / "accompaniment.mp3"
        shutil.copy2(accompaniment_path, acc_out)
        logger.info(f"Copied: {acc_out}")

    # Create ZIP
    zip_path = output_dir / f"{title}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.iterdir():
            if f.suffix.lower() == ".zip":
                continue
            zf.write(f, f.name)
    logger.info(f"Created ZIP: {zip_path}")

    return output_dir
