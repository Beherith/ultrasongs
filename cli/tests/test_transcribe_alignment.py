"""Tests that per-frame band energy aligns exactly with torchcrepe pitch frames.

torchcrepe (``pad=True``) emits ``1 + len(audio) // hop`` frames, and frame i
analyzes the window ``[i*hop - 512, i*hop + 512)`` of the 16 kHz audio
(zero-padded 512 samples on both sides of its 1024-sample window).  Frame i is
therefore *centered* on ``i*hop``.  ``_compute_band_energy`` must use the same
windows and the same count so that ``energies[i]`` describes the same region as
pitch frame i.
"""

import numpy as np
import pytest

from cli.transcribe import CREPE_SR, _compute_band_energy

HOP = 160  # 10 ms hop at 16 kHz (crepe_hop_ms=10)
FRAME = 1024  # torchcrepe WINDOW_SIZE
FMIN, FMAX = 60.0, 1000.0


def _crepe_frame_count(length: int) -> int:
    """Frame count torchcrepe emits for ``length`` samples with pad=True."""
    return 1 + length // HOP


def _crepe_window(audio: np.ndarray, i: int) -> np.ndarray:
    """The exact 1024-sample region torchcrepe feeds to its frame i."""
    start = i * HOP - FRAME // 2
    win = np.zeros(FRAME, dtype=np.float32)
    lo = max(start, 0)
    hi = min(start + FRAME, len(audio))
    if hi > lo:
        win[lo - start: hi - start] = audio[lo:hi]
    return win


def _reference_energy(audio: np.ndarray, i: int) -> float:
    """Independent single-frame re-implementation of the band energy."""
    seg = _crepe_window(audio, i) * np.hanning(FRAME)
    mags = np.abs(np.fft.rfft(seg))
    freq_res = CREPE_SR / FRAME
    b0 = max(0, int(FMIN / freq_res))
    b1 = min(int(FMAX / freq_res) + 1, mags.size)
    return float(np.sqrt(np.sum(mags[b0:b1] ** 2) / max(b1 - b0, 1)))


def _burst_audio(burst_start: int = 16000, duration: int = 640, total: int = 160000) -> np.ndarray:
    """Silence with a 220 Hz windowed burst, entirely inside the 60-1000 Hz band."""
    audio = np.zeros(total, dtype=np.float32)
    t = np.arange(duration)
    audio[burst_start:burst_start + duration] = (
        0.5 * np.sin(2 * np.pi * 220.0 * t / CREPE_SR) * np.hanning(duration)
    )
    return audio


class TestFrameCount:
    @pytest.mark.parametrize(
        "length",
        [0, 1, 159, 160, 511, 512, 513, 1023, 1024, 1025, 3201, 16037, 441007],
    )
    def test_raw_count_matches_torchcrepe_formula(self, length):
        audio = np.zeros(length, dtype=np.float32)
        assert len(_compute_band_energy(audio, CREPE_SR, HOP)) == _crepe_frame_count(length)

    def test_n_frames_explicit(self):
        audio = np.zeros(32000, dtype=np.float32)
        n = _crepe_frame_count(32000)
        assert len(_compute_band_energy(audio, CREPE_SR, HOP, n_frames=n)) == n


class TestExactFrameAlignment:
    @pytest.mark.parametrize("length", [161, 513, 1025, 4321, 16037, 32000])
    def test_every_frame_matches_independent_reference(self, length):
        rng = np.random.default_rng(length)
        audio = (rng.standard_normal(length) * 0.1).astype(np.float32)

        energies = _compute_band_energy(audio, CREPE_SR, HOP)
        assert len(energies) == _crepe_frame_count(length)
        for i, e in enumerate(energies):
            assert float(e) == pytest.approx(_reference_energy(audio, i), rel=1e-5, abs=1e-7)

    def test_burst_support_lands_exactly_on_the_matching_frames(self):
        # Burst spans samples [16000, 16640) = [1.000 s, 1.040 s).
        # A frame window [i*160-512, i*160+512) overlaps the burst iff
        # i*160+512 > 16000 and i*160-512 < 16640, i.e. i in [97, 108).
        audio = _burst_audio()
        energies = _compute_band_energy(audio, CREPE_SR, HOP)

        assert (energies[:97] == 0).all()
        assert (energies[97:108] > 0).all()
        assert (energies[108:] == 0).all()
        # Burst center 16320 == frame 102's center
        assert 101 <= int(np.argmax(energies)) <= 103

    def test_head_and_tail_frames_use_partial_windows_not_zero(self):
        # With a non-zero tone over the whole file, the first and last frames
        # see partial (but not empty) windows, so their energy is > 0 even
        # though older start-aligned code produced a short/zero tail.
        t = np.arange(3207, dtype=np.float32)
        audio = (0.2 * np.sin(2 * np.pi * 220.0 * t / CREPE_SR)).astype(np.float32)
        energies = _compute_band_energy(audio, CREPE_SR, HOP)
        n = _crepe_frame_count(len(audio))
        assert len(energies) == n
        assert energies[0] > 0
        assert energies[-1] > 0
        assert float(energies[-1]) == pytest.approx(_reference_energy(audio, n - 1), rel=1e-5)

    def test_n_frames_padding_and_truncation(self):
        audio = np.ones(32000, dtype=np.float32)
        n = _crepe_frame_count(32000)
        base = _compute_band_energy(audio, CREPE_SR, HOP, n_frames=n)

        padded = _compute_band_energy(audio, CREPE_SR, HOP, n_frames=n + 5)
        assert len(padded) == n + 5
        assert np.allclose(padded[:n], base)
        assert (padded[n:] == 0).all()

        truncated = _compute_band_energy(audio, CREPE_SR, HOP, n_frames=n - 3)
        assert len(truncated) == n - 3
        assert np.allclose(truncated, base[: n - 3])


class TestAgainstRealTorchcrepe:
    def test_frames_captured_from_preprocess_match_centered_windows(self, monkeypatch):
        """Assert torchcrepe's own frame extraction, not just the docs."""
        torchcrepe = pytest.importorskip("torchcrepe")
        import torch

        audio = (np.random.default_rng(1).standard_normal(32037) * 0.1).astype(np.float32)
        wav = torch.from_numpy(audio).unsqueeze(0)
        length = len(audio)
        total = _crepe_frame_count(length)

        captured = []
        orig_unfold = torch.nn.functional.unfold

        def spy(x, *args, **kwargs):
            out = orig_unfold(x, *args, **kwargs)
            captured.append(x)
            return out

        monkeypatch.setattr(torch.nn.functional, "unfold", spy)
        list(torchcrepe.preprocess(wav, CREPE_SR, HOP, batch_size=total, device="cpu", pad=True))
        monkeypatch.undo()

        assert len(captured) == 1
        chunk = captured[0].reshape(-1)
        # chunk = padded audio trimmed to the end of the last frame's window
        assert chunk.numel() == (total - 1) * HOP + FRAME
        # left half is the 512-sample zero pad; real audio follows it exactly
        assert (chunk[: FRAME // 2] == 0).all()
        assert torch.equal(chunk[FRAME // 2: FRAME // 2 + length], wav[0])
        n_right_pad = chunk.numel() - (FRAME // 2 + length)
        if n_right_pad > 0:
            assert (chunk[-n_right_pad:] == 0).all()
        # every sampled frame window matches the centered [i*hop-512, i*hop+512)
        for i in (0, 1, 100, 101, total - 2, total - 1):
            window = chunk[i * HOP: i * HOP + FRAME].numpy()
            assert np.array_equal(window, _crepe_window(audio, i)), f"frame {i} window mismatch"

    def test_count_matches_real_torchcrepe_output(self):
        """The band-energy frame count equals the count of a real
        torchcrepe.predict on the same 16 kHz input (the exact question: do the
        pitch frames and the RMS energy pass have the same length?)."""
        torchcrepe = pytest.importorskip("torchcrepe")
        import torch

        audio = _burst_audio(total=32000)  # 2 s: burst at [1.000, 1.040) s
        wav = torch.from_numpy(audio).unsqueeze(0)
        pitch, _ = torchcrepe.predict(
            wav,
            sample_rate=CREPE_SR,
            hop_length=HOP,
            fmin=65.41,
            fmax=1046.5,
            model="full",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
            batch_size=2048,
            device="cpu",
            pad=True,
        )

        energies = _compute_band_energy(audio, CREPE_SR, HOP)
        assert len(energies) == pitch.shape[1] == _crepe_frame_count(len(audio))


class TestAnalyzePitchIntegration:
    def test_analysis_arrays_same_length_and_burst_aligned(self):
        pytest.importorskip("torchcrepe")
        from cli.transcribe import analyze_pitch

        audio = _burst_audio(total=32000)
        times, freqs, confs, energies = analyze_pitch(audio, CREPE_SR, hop_ms=10)

        n = _crepe_frame_count(len(audio))
        assert len(times) == len(freqs) == len(confs) == len(energies) == n
        assert float(times[0]) == 0.0
        # Analyzed at 16 kHz with 10 ms hop: frame i is centered on i * 0.010 s
        i = 100
        assert float(times[i]) == pytest.approx(1.0, abs=1e-4)
        # Normalized energies: zero outside the burst, peaked at its center
        assert (energies[:97] == 0).all()
        assert (energies[108:] == 0).all()
        assert (energies[97:108] > 0).all()
        assert 101 <= int(np.argmax(energies)) <= 103
        assert float(energies[int(np.argmax(energies))]) == pytest.approx(1.0)
