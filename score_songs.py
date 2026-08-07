#!/usr/bin/env python3
"""Score two UltraStar song files for similarity on timing, duration, and pitch."""

import sys
import math
from pathlib import Path

def parse_ultrastar(filepath: str) -> dict:
    metadata = {}
    notes = []

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

    def beats_to_ms(beats):
        return beats * (60000.0 / bpm / 4)

    # First pass: collect raw time values to detect unit (beats vs milliseconds)
    raw_times = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(":") or line.startswith("*"):
                parts = line[1:].split(None, 3)
                if len(parts) >= 4:
                    raw_times.append(float(parts[0]))
            elif line.startswith("-") and raw_times:
                break

    # Heuristic: if first-verse time range < 20, values are in beat units;
    # otherwise they are raw milliseconds.
    times_are_ms = (max(raw_times) - min(raw_times) >= 20) if raw_times else False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(":") or line.startswith("*"):
                parts = line[1:].split(None, 3)
                if len(parts) < 4:
                    continue
                raw_start = float(parts[0])
                raw_dur = float(parts[1])
                pitch = int(parts[2])
                lyric = parts[3]

                if times_are_ms:
                    start_ms = gap + raw_start
                    dur_ms = raw_dur
                else:
                    start_ms = gap + beats_to_ms(raw_start)
                    dur_ms = beats_to_ms(raw_dur)

                notes.append({
                    "start": start_ms,
                    "duration": dur_ms,
                    "pitch": pitch,
                    "lyric": lyric,
                    "chorus": line.startswith("*"),
                })

    return {"metadata": metadata, "notes": notes, "times_are_ms": times_are_ms}


def align_notes(notes_a, notes_b):
    """Match notes by lyric text, tracking indices to avoid reusing notes."""
    used_b = set()
    pairs = []

    for a in notes_a:
        best_idx = None
        best_dist = float("inf")
        for i, b in enumerate(notes_b):
            if i in used_b:
                continue
            if b["lyric"].strip() == a["lyric"].strip():
                dist = abs(b["start"] - a["start"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        if best_idx is not None:
            used_b.add(best_idx)
            pairs.append((a, notes_b[best_idx]))

    return pairs


def score_timing(pairs):
    """Mean squared error of start-time differences (ms)."""
    if not pairs:
        return 0.0
    errors = [(a["start"] - b["start"]) ** 2 for a, b in pairs]
    mse = sum(errors) / len(errors)
    return math.sqrt(mse)


def score_duration(pairs):
    """Mean squared error of duration differences (ms)."""
    if not pairs:
        return 0.0
    errors = [(a["duration"] - b["duration"]) ** 2 for a, b in pairs]
    mse = sum(errors) / len(errors)
    return math.sqrt(mse)


def octave_corrected_distance(pitch_a, pitch_b):
    """Chromatic distance between pitch classes (octave-invariant)."""
    a_class = pitch_a % 12
    b_class = pitch_b % 12
    diff = abs(a_class - b_class)
    return min(diff, 12 - diff)


def score_pitch(pairs):
    """Mean octave-corrected pitch distance in semitones."""
    if not pairs:
        return 0.0
    distances = [octave_corrected_distance(a["pitch"], b["pitch"]) for a, b in pairs]
    return sum(distances) / len(distances)


def main():
    if len(sys.argv) < 3:
        print("Usage: python score_songs.py <song1.txt> <song2.txt>")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]

    for f in (file_a, file_b):
        if not Path(f).exists():
            print(f"Error: {f} not found.")
            sys.exit(1)

    data_a = parse_ultrastar(file_a)
    data_b = parse_ultrastar(file_b)

    title_a = data_a["metadata"].get("TITLE", Path(file_a).stem)
    title_b = data_b["metadata"].get("TITLE", Path(file_b).stem)
    bpm_a = data_a["metadata"].get("BPM", "—")
    bpm_b = data_b["metadata"].get("BPM", "—")
    unit_a = "ms" if data_a.get("times_are_ms") else "beats"
    unit_b = "ms" if data_b.get("times_are_ms") else "beats"

    pairs = align_notes(data_a["notes"], data_b["notes"])

    total_a = len(data_a["notes"])
    total_b = len(data_b["notes"])
    matched = len(pairs)

    rmse_time = score_timing(pairs)
    rmse_dur = score_duration(pairs)
    avg_pitch = score_pitch(pairs)

    print(f"{'='*52}")
    print(f"  UltraStar Song Similarity Score")
    print(f"{'='*52}")
    print(f"  File A: {title_a}  (BPM {bpm_a}, {total_a} notes, times in {unit_a})")
    print(f"  File B: {title_b}  (BPM {bpm_b}, {total_b} notes, times in {unit_b})")
    print(f"  Matched: {matched} / {min(total_a, total_b)} notes")
    print(f"{'-'*52}")
    print(f"  Timing RMSE:        {rmse_time:>10.2f} ms")
    print(f"  Duration RMSE:      {rmse_dur:>10.2f} ms")
    print(f"  Pitch distance:     {avg_pitch:>10.2f} semitones (octave-corrected)")
    print(f"{'='*52}")

    if pairs:
        time_errors = sorted([(a["start"] - b["start"]) ** 2 for a, b in pairs])
        dur_errors = sorted([(a["duration"] - b["duration"]) ** 2 for a, b in pairs])
        pch_dists = sorted([octave_corrected_distance(a["pitch"], b["pitch"]) for a, b in pairs])

        print(f"  Timing  — median: {math.sqrt(time_errors[len(time_errors)//2]):.1f} ms, "
              f"max: {math.sqrt(time_errors[-1]):.1f} ms")
        print(f"  Duration— median: {math.sqrt(dur_errors[len(dur_errors)//2]):.1f} ms, "
              f"max: {math.sqrt(dur_errors[-1]):.1f} ms")
        print(f"  Pitch   — median: {pch_dists[len(pch_dists)//2]:.1f} st, "
              f"max: {pch_dists[-1]:.1f} st")
        print(f"{'='*52}")


if __name__ == "__main__":
    main()
