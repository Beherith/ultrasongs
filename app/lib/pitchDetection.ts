/**
 * YIN pitch detection algorithm.
 * Returns frequency in Hz, or -1 if no reliable pitch is found.
 * Expects mono float32 PCM buffer at the given sample rate.
 */
export function detectPitch(buffer: Float32Array, sampleRate: number): number {
  const W = buffer.length;
  const half = Math.floor(W / 2);

  // Step 1: difference function
  const d = new Float32Array(half);
  for (let tau = 1; tau < half; tau++) {
    let s = 0;
    for (let i = 0; i < half; i++) {
      const delta = buffer[i] - buffer[i + tau];
      s += delta * delta;
    }
    d[tau] = s;
  }

  // Step 2: cumulative mean normalised difference
  const cmnd = new Float32Array(half);
  cmnd[0] = 1;
  let sum = 0;
  for (let tau = 1; tau < half; tau++) {
    sum += d[tau];
    cmnd[tau] = sum === 0 ? 0 : (d[tau] * tau) / sum;
  }

  // Step 3: absolute threshold — find first dip below 0.15
  const threshold = 0.15;
  let tau = 2;
  while (tau < half - 1) {
    if (cmnd[tau] < threshold) {
      // Parabolic interpolation for sub-sample precision
      const prev = cmnd[tau - 1];
      const curr = cmnd[tau];
      const next = cmnd[tau + 1];
      const betterTau = tau + (next - prev) / (2 * (2 * curr - prev - next));
      return sampleRate / betterTau;
    }
    tau++;
  }

  return -1;
}

/** Convert Hz to MIDI note number (middle C = 60). */
export function hzToMidi(hz: number): number {
  if (hz <= 0) return -1;
  return Math.round(12 * Math.log2(hz / 440) + 69);
}
