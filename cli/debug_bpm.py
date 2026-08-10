"""Debug script: split an MP3 into 30 s chunks and compare BPM estimates."""

import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np


CHUNK_DURATION = 30  # seconds


def detect_bpm(audio_path: Path) -> dict:
    """Return BPM estimate and raw tempo vector for a file."""
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo_estimates = librosa.feature.tempo(y=audio, sr=sr)
    return {
        "bpm": float(np.median(tempo_estimates)),
        "mean": float(np.mean(tempo_estimates)),
        "std": float(np.std(tempo_estimates)),
        "min": float(np.min(tempo_estimates)),
        "max": float(np.max(tempo_estimates)),
        "samples": len(audio),
        "duration": len(audio) / sr,
        "tempo_vector": tempo_estimates,
    }


def split_mp3(input_path: Path, chunk_duration: int, output_dir: Path) -> list[Path]:
    """Split an MP3 into fixed-length chunks using FFmpeg. Returns chunk paths."""
    chunks: list[Path] = []
    t = 0
    idx = 0

    while True:
        out = output_dir / f"chunk_{idx:04d}.mp3"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ss", str(t),
            "-t", str(chunk_duration),
            "-acodec", "libmp3lame",
            "-ab", "128k",
            "-ar", "44100",
            "-ac", "1",
            str(out),
        ]
        print(f"  [ffmpeg] {' '.join(cmd[6:])}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            print(f"  [ffmpeg] {line.rstrip()}")
        proc.stdout.close()
        proc.wait()
        if proc.returncode != 0:
            break
        if not out.exists() or out.stat().st_size < 4096:
            # Empty output (FFmpeg writes a ~1KiB MP3 header even with no audio)
            out.unlink(missing_ok=True)
            break
        chunks.append(out)
        idx += 1
        t += chunk_duration

    return chunks


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python debug_bpm.py <path_to_mp3>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    print(f"Input: {input_path} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print()

    # Overall BPM
    print("=" * 60)
    print("OVERALL BPM")
    print("=" * 60)
    overall = detect_bpm(input_path)
    print(f"  Duration : {overall['duration']:.2f}s  ({overall['samples']} samples @ 22050 Hz)")
    print(f"  Median   : {overall['bpm']:.2f} BPM")
    print(f"  Mean     : {overall['mean']:.2f}")
    print(f"  Std dev  : {overall['std']:.2f}")
    print(f"  Range    : {overall['min']:.2f} - {overall['max']:.2f}")
    print()

    # Split into chunks
    with tempfile.TemporaryDirectory(prefix="bpm_debug_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        print(f"Splitting into {CHUNK_DURATION}s chunks ...")
        chunks = split_mp3(input_path, CHUNK_DURATION, tmpdir_path)
        print(f"  -> {len(chunks)} chunks\n")

        # Per-chunk BPM
        print("=" * 60)
        print("PER-CHUNK BPM")
        print("=" * 60)
        print(f"  {'#':>4}  {'File':<20}  {'Duration':>8}  {'Median':>8}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print("  " + "-" * 84)

        chunk_infos: list[dict] = []
        for i, chunk in enumerate(chunks):
            info = detect_bpm(chunk)
            chunk_infos.append(info)
            print(
                f"  {i+1:>4}  {chunk.name:<20}  "
                f"{info['duration']:>8.2f}s  "
                f"{info['bpm']:>8.2f}  "
                f"{info['mean']:>8.2f}  "
                f"{info['std']:>8.2f}  "
                f"{info['min']:>8.2f}  "
                f"{info['max']:>8.2f}"
            )

        print()

        # Summary
        chunk_bpms = [info["bpm"] for info in chunk_infos]
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Overall BPM       : {overall['bpm']:.2f}")
        print(f"  Chunk avg BPM     : {np.mean(chunk_bpms):.2f}")
        print(f"  Chunk median BPM  : {np.median(chunk_bpms):.2f}")
        print(f"  Chunk std BPM     : {np.std(chunk_bpms):.2f}")
        print(f"  Chunk range       : {min(chunk_bpms):.2f} - {max(chunk_bpms):.2f}")
        print(f"  Deviation from overall: {abs(np.mean(chunk_bpms) - overall['bpm']):.2f}")


if __name__ == "__main__":
    main()
