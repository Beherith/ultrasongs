"""Test script: octave-folded spectrogram for a vocals MP3.

Decodes a vocals MP3, splits it into fixed-length chunks (5 s by default),
and renders one PNG per chunk with two panels:

1. Focus panel (220-440 Hz): the base octave 220-440 Hz spectrogram with
   every other octave band (55-110, 110-220, 440-880, 880-1760, 1760-3520,
   3520-4400) stretched onto the base band's grid and additively summed on
   top of it.
2. Reference panel: the plain 55-4400 Hz spectrogram on a log2 frequency
   axis, with the 220-440 Hz base octave highlighted.

FFT sizing (per run, computed from the sample rate):
- n_fft is the smallest size >= 2048 for which 220 Hz and 440 Hz fall
  exactly on bin edges, so the 220-440 Hz band is covered by whole bins.
- hop is the largest integer divisor of n_fft that is <= n_fft/4
  (guarantees hop divides the FFT size and >= 75% window overlap).

Usage (from the repo root):
    python octave_spectrogram.py tmp/test_song_full_audio_vocals.mp3
    python octave_spectrogram.py vocals.mp3 --chunk-sec 5 --outdir tmp/octave_plots
"""

import argparse
import shutil
from math import gcd, lcm
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cli.ffmpeg_pcm import extract_pcm

FREQ_MIN = 55.0
FREQ_MAX = 4400.0
BASE_LO = 220.0
BASE_HI = 440.0
MIN_NFFT = 2048
DB_EPS = 1e-12

OCTAVE_EDGE_TICKS = [55, 110, 220, 440, 880, 1760, 3520, 4400]


def choose_fft_size(sample_rate: int) -> int:
    """Smallest n_fft >= MIN_NFFT placing 220 Hz and 440 Hz on bin edges."""
    need = lcm(
        sample_rate // gcd(220, sample_rate),
        sample_rate // gcd(440, sample_rate),
    )
    return need * max(1, -(-MIN_NFFT // need))


def choose_hop(n_fft: int) -> int:
    """Largest divisor of n_fft that is <= n_fft/4 (>= 75% overlap)."""
    for hop in range(n_fft // 4, 0, -1):
        if n_fft % hop == 0:
            return hop
    return 1


def build_octave_bands() -> list[tuple[float, float]]:
    """Octave bands covering FREQ_MIN-FREQ_MAX, anchored on the base octave."""
    lower: list[tuple[float, float]] = []
    lo = BASE_LO
    while lo / 2.0 >= FREQ_MIN - 1e-9:
        lo /= 2.0
        lower.append((lo, lo * 2.0))
    upper: list[tuple[float, float]] = []
    hi = BASE_HI
    while hi < FREQ_MAX - 1e-9:
        upper.append((hi, min(hi * 2.0, FREQ_MAX)))
        hi *= 2.0
    return list(reversed(lower)) + [(BASE_LO, BASE_HI)] + upper


def stretch_to_edges(amplitude: np.ndarray, centers: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Interpolate (n_frames x n_bins) amplitude sampled at `centers` onto
    `edges` (Hz) along the frequency axis, linearly per time frame."""
    i0 = np.clip(np.searchsorted(centers, edges, side="right") - 1, 0, len(centers) - 1)
    i1 = np.minimum(i0 + 1, len(centers) - 1)
    denom = centers[i1] - centers[i0]
    frac = np.clip((edges - centers[i0]) / np.where(denom == 0, 1.0, denom), 0.0, 1.0)
    return amplitude[:, i0] * (1.0 - frac) + amplitude[:, i1] * frac


def build_folded(spec: np.ndarray, centers: np.ndarray, bin_width: float,
                 bands: list[tuple[float, float]]) -> np.ndarray:
    """Sum all octave bands onto the base octave's cell grid.

    Returns (n_frames x K) linear amplitude, where K is the number of bins
    spanning 220-440 Hz. The base band uses its exact bins; other bands are
    interpolated onto the same K-cell boundary grid and added.
    """
    n_frames = spec.shape[0]
    k = int(round((BASE_HI - BASE_LO) / bin_width))
    total = np.zeros((n_frames, k))
    for lo, hi in bands:
        if hi - lo == BASE_HI - BASE_LO:
            i0 = int(round(lo / bin_width))
            i1 = int(round(hi / bin_width))
            cells = spec[:, i0:i1]
        else:
            band_edges = np.linspace(lo, hi, k + 1)
            cells = np.diff(stretch_to_edges(spec, centers, band_edges), axis=1)
            cells = np.clip(cells, 0.0, None)
        if cells.shape[1] != k:
            raise RuntimeError(f"band {lo}-{hi} Hz gave {cells.shape[1]} cells, expected {k}")
        total = total + cells
    return total


def render_chunk(
    index: int,
    label: str,
    spec: np.ndarray,
    centers: np.ndarray,
    bin_width: float,
    times: np.ndarray,
    bands: list[tuple[float, float]],
    n_fft: int,
    hop: int,
    outdir: Path,
) -> Path:
    k = int(round((BASE_HI - BASE_LO) / bin_width))
    folded = build_folded(spec, centers, bin_width, bands)
    folded_db = 20.0 * np.log10(folded + DB_EPS)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 0.85],
        hspace=0.22,
        left=0.05, right=0.97, top=0.92, bottom=0.06,
    )
    ax_fold = fig.add_subplot(gs[0])
    ax_ref = fig.add_subplot(gs[1])

    im = ax_fold.imshow(
        folded_db,
        extent=[times[0], times[-1], BASE_LO, BASE_HI],
        aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_fold, label="20*log10(summed linear amplitude)")
    ax_fold.set_ylabel("base octave (Hz)")
    band_list = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in bands)
    ax_fold.set_title(
        f"Folded octaves onto 220-440 Hz  (bands: {band_list})",
        fontsize=10,
    )
    ax_fold.grid(True, alpha=0.2)

    mask = (centers >= FREQ_MIN) & (centers <= FREQ_MAX)
    ref_db = 20.0 * np.log10(spec[:, mask] + DB_EPS)
    im2 = ax_ref.imshow(
        ref_db,
        extent=[times[0], times[-1], FREQ_MIN, FREQ_MAX],
        aspect="auto", origin="lower", cmap="magma",
    )
    ax_ref.set_yscale("log", base=2)
    ax_ref.set_ylim(FREQ_MIN, FREQ_MAX)
    ax_ref.set_yticks(OCTAVE_EDGE_TICKS)
    ax_ref.set_yticklabels([f"{v}" for v in OCTAVE_EDGE_TICKS])
    ax_ref.axhspan(BASE_LO, BASE_HI, color="red", alpha=0.15, zorder=2)
    ax_ref.axhline(BASE_LO, color="white", ls="--", lw=0.8, alpha=0.7)
    ax_ref.axhline(BASE_HI, color="white", ls="--", lw=0.8, alpha=0.7)
    ax_ref.set_ylabel("frequency (Hz, log2)")
    ax_ref.set_title(f"Reference spectrogram {FREQ_MIN:.0f}-{FREQ_MAX:.0f} Hz", fontsize=10)
    ax_ref.grid(True, alpha=0.2)

    overlap = (1.0 - hop / n_fft) * 100.0
    fig.suptitle(
        f"{label}   chunk {index + 1}: {times[0]:.2f}-{times[-1]:.2f}s   "
        f"n_fft={n_fft}  bin={bin_width:g} Hz  hop={hop} ({overlap:.0f}% overlap)",
        fontsize=11,
    )

    fname = outdir / f"chunk_{index:02d}_{times[0]:.2f}-{times[-1]:.2f}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def main() -> None:
    ap = argparse.ArgumentParser(description="Octave-folded spectrogram for a vocals MP3.")
    ap.add_argument("mp3", type=Path, help="Vocals MP3 file")
    ap.add_argument("--chunk-sec", type=float, default=5.0, help="Chunk length in seconds (default 5)")
    ap.add_argument("--outdir", type=Path, default=Path("tmp/octave_plots"), help="Output directory")
    ap.add_argument("--sr", type=int, default=44100, help="Decode sample rate (default 44100)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")

    n_fft = choose_fft_size(args.sr)
    bin_width = args.sr / n_fft
    for target in (220.0, 440.0):
        if abs(target / bin_width - round(target / bin_width)) > 1e-9:
            raise RuntimeError(f"{target} Hz does not land on a bin edge (bin={bin_width:g} Hz)")
    hop = choose_hop(n_fft)
    bands = build_octave_bands()

    print(f"sr={args.sr}  n_fft={n_fft}  bin={bin_width:g} Hz  "
          f"220Hz=bin {220 / bin_width:.0f}  440Hz=bin {440 / bin_width:.0f}  "
          f"hop={hop} (divides n_fft, overlap={(1 - hop / n_fft) * 100:.0f}%)")
    print(f"octave bands: {', '.join(f'{lo:.0f}-{hi:.0f}' for lo, hi in bands)}")

    pcm = extract_pcm(args.mp3, args.sr)
    audio = np.frombuffer(pcm, dtype=np.float32).astype(np.float64)
    duration = len(audio) / args.sr
    centers = np.fft.rfftfreq(n_fft, 1.0 / args.sr)
    print(f"decoded {duration:.2f}s from {args.mp3}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    chunk_samples = int(round(args.chunk_sec * args.sr))
    written: list[Path] = []
    index = 0
    pos = 0
    while pos < len(audio):
        seg = audio[pos:pos + chunk_samples]
        if len(seg) < n_fft:
            print(f"skipping tail of {len(seg) / args.sr:.2f}s (< n_fft={n_fft})")
            break
        n_frames = 1 + (len(seg) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop
        frames = seg[idx] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frames, axis=1))
        times = pos / args.sr + (np.arange(n_frames) * hop) / args.sr
        fname = render_chunk(index, args.mp3.stem, spec, centers, bin_width, times,
                             bands, n_fft, hop, args.outdir)
        written.append(fname)
        print(f"wrote {fname}")
        index += 1
        pos += chunk_samples

    print(f"done: {len(written)} chunk(s) in {args.outdir}")


if __name__ == "__main__":
    main()
