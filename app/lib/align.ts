import { splitWord } from "./syllabify";
import type { WordTimestamp, Pause } from "../api/transcribe/route";
import * as fs from "node:fs";
import * as path from "node:path";

export interface AlignedSyllable {
  syllable: string;
  start: number;
  end: number;
  midi: number;
  isLineBreak?: boolean;
}

interface PitchFrame {
  time: number;
  midi: number;
  confidence: number;
}

interface WordWithPitch extends WordTimestamp {
  pitchFrames?: PitchFrame[];
}

// ── File-based logger ─────────────────────────────────────────────────────

let logFd: number | null = null;
let logPath: string = "";

function openLog(songId: string) {
  const tmpDir = path.resolve("./tmp");
  fs.mkdirSync(tmpDir, { recursive: true });
  const safeId = songId ? songId.replace(/[^a-zA-Z0-9_\-\s]/g, "").trim().replace(/\s+/g, "_") : "alignment";
  logPath = path.join(tmpDir, `${safeId}_align.log`);
  fs.writeFileSync(logPath, "");
  logFd = fs.openSync(logPath, "a");
}

function closeLog() {
  if (logFd !== null) {
    try { fs.closeSync(logFd); } catch { /* ignore */ }
    logFd = null;
  }
}

function log(tag: string, msg: string) {
  const line = `[align:${tag}] ${msg}`;
  console.log(line);
  if (logFd !== null) {
    try { fs.writeSync(logFd, line + "\n"); } catch { /* ignore */ }
  }
}

// ── Debug types ────────────────────────────────────────────────────────────

interface DebugBacktrackStep {
  i: number;
  j: number;
  matrix: "M" | "X" | "Y";
  score: number;
  lyricChar: string;
  whisperChar: string;
  whisperWordIdx: number;
  whisperWord: string;
}

interface DebugWordMatch {
  lyricWord: string;
  lyricNorm: string;
  lyricCharRange: [number, number];
  matchedWhisperWordIdxs: number[];
  matchedWhisperWords: string[];
  start: number | null;
  end: number | null;
  midi: number | null;
  source: "sw_aligned" | "interpolated_before" | "interpolated_between" | "interpolated_after" | "no_pitch_data";
  charAlignments: Array<{ lyricChar: string; whisperChar: string; whisperWordIdx: number; score: number }>;
}

interface DebugLine {
  lineIdx: number;
  lyricLine: string;
  words: DebugWordMatch[];
  syllables: Array<{ syllable: string; start: number; end: number; midi: number; pitchFrameCount: number }>;
}

interface DebugRoot {
  songId: string;
  language: string;
  lyricCharCount: number;
  whisperCharCount: number;
  whisperWordCount: number;
  whisperTimeRange: [number, number];
  swMaxScore: number;
  swMaxPos: [number, number];
  swBacktrackLength: number;
  backtrack: DebugBacktrackStep[];
  pauses: Pause[];
  lines: DebugLine[];
  summary: {
    totalLyricWords: number;
    alignedWords: number;
    interpolatedWords: number;
    totalSyllables: number;
    lineBreaks: number;
  };
}

const debug: DebugRoot = {
  songId: "",
  language: "",
  lyricCharCount: 0,
  whisperCharCount: 0,
  whisperWordCount: 0,
  whisperTimeRange: [0, 0],
  swMaxScore: 0,
  swMaxPos: [0, 0],
  swBacktrackLength: 0,
  backtrack: [],
  pauses: [],
  lines: [],
  summary: {
    totalLyricWords: 0,
    alignedWords: 0,
    interpolatedWords: 0,
    totalSyllables: 0,
    lineBreaks: 0,
  },
};

// ── Character normalization ────────────────────────────────────────────────

function normalizeChar(c: string): string {
  return c.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

// ── Phonetic character matching ────────────────────────────────────────────

function phoneticScore(x: string, y: string): number {
  if (x === y) return 1;
  const groups: string[][] = [
    ["a", "e", "i"],
    ["o", "u"],
    ["s", "z", "c"],
    ["t", "d"],
    ["p", "b"],
    ["k", "g"],
    ["f", "v", "w"],
    ["m", "n"],
    ["l", "r"],
    ["y", "i"],
    ["h", "j"],
    ["b", "v"],
  ];
  for (const g of groups) {
    const ix = g.indexOf(x);
    const iy = g.indexOf(y);
    if (ix >= 0 && iy >= 0) {
      return 0.6 - 0.1 * Math.abs(ix - iy);
    }
  }
  const cross: [string, string][] = [
    ["a", "o"], ["a", "u"], ["e", "i"], ["e", "o"], ["i", "y"],
    ["s", "sh"], ["z", "zh"], ["f", "ph"], ["c", "k"], ["q", "k"],
    ["w", "u"], ["r", "l"], ["b", "p"], ["d", "t"], ["g", "k"],
  ];
  for (const [a, b] of cross) {
    if ((x === a && y === b) || (x === b && y === a)) return 0.5;
  }
  return -0.3;
}

// ── Smith-Waterman with affine gap penalties ──────────────────────────────

const MATCH_SCORE = 4;
const GAP_OPEN = 4;
const GAP_EXTEND = 0.5;

interface SWCell {
  score: number;
  trace: 0 | 1 | 2 | 3;
}

function smithWaterman(
  lyricChars: string[],
  whisperChars: string[],
): {
  maxScore: number;
  maxI: number;
  maxJ: number;
  backtrack: Array<{ i: number; j: number; matrix: "M" | "X" | "Y"; score: number }>;
} {
  const L = lyricChars.length;
  const W = whisperChars.length;

  const M: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));
  const X: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));
  const Y: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));

  const traceM: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));
  const traceX: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));
  const traceY: number[][] = Array.from({ length: L + 1 }, () => Array.from({ length: W + 1 }, () => 0));

  let maxScore = 0;
  let maxI = 0;
  let maxJ = 0;

  for (let i = 1; i <= L; i++) {
    for (let j = 1; j <= W; j++) {
      const s = phoneticScore(lyricChars[i - 1], whisperChars[j - 1]) * MATCH_SCORE;

      M[i][j] = Math.max(0, s + M[i - 1][j - 1], s - GAP_OPEN + Math.max(X[i - 1][j - 1], Y[i - 1][j - 1]));
      if (M[i][j] === 0) traceM[i][j] = 0;
      else if (M[i][j] === s + M[i - 1][j - 1]) traceM[i][j] = 1;
      else traceM[i][j] = 2;

      X[i][j] = Math.max(-GAP_OPEN + M[i - 1][j], -GAP_EXTEND + X[i - 1][j]);
      traceX[i][j] = X[i][j] === -GAP_OPEN + M[i - 1][j] ? 3 : 4;

      Y[i][j] = Math.max(-GAP_OPEN + M[i][j - 1], -GAP_EXTEND + Y[i][j - 1]);
      traceY[i][j] = Y[i][j] === -GAP_OPEN + M[i][j - 1] ? 5 : 6;

      const best = Math.max(M[i][j], X[i][j], Y[i][j]);
      if (best > maxScore) {
        maxScore = best;
        maxI = i;
        maxJ = j;
      }
    }
  }

  const backtrack: Array<{ i: number; j: number; matrix: "M" | "X" | "Y"; score: number }> = [];
  let ci = maxI, cj = maxJ;

  let cmat: "M" | "X" | "Y";
  let cscore: number;
  if (M[ci][cj] >= X[ci][cj] && M[ci][cj] >= Y[ci][cj]) { cmat = "M"; cscore = M[ci][cj]; }
  else if (X[ci][cj] >= Y[ci][cj]) { cmat = "X"; cscore = X[ci][cj]; }
  else { cmat = "Y"; cscore = Y[ci][cj]; }

  while (cscore > 0 && (ci > 0 || cj > 0)) {
    backtrack.push({ i: ci, j: cj, matrix: cmat, score: cscore });

    if (cmat === "M") {
      const t = traceM[ci][cj];
      if (t === 1) { ci--; cj--; }
      else { ci--; cj--; }
    } else if (cmat === "X") {
      const t = traceX[ci][cj];
      if (t === 3) { ci--; }
      else { ci--; }
    } else {
      const t = traceY[ci][cj];
      if (t === 5) { cj--; }
      else { cj--; }
    }

    if (ci < 0) ci = 0;
    if (cj < 0) cj = 0;

    if (M[ci][cj] >= X[ci][cj] && M[ci][cj] >= Y[ci][cj]) { cmat = "M"; cscore = M[ci][cj]; }
    else if (X[ci][cj] >= Y[ci][cj]) { cmat = "X"; cscore = X[ci][cj]; }
    else { cmat = "Y"; cscore = Y[ci][cj]; }

    if (cscore <= 0) break;
  }

  backtrack.reverse();
  return { maxScore, maxI, maxJ, backtrack };
}

// ── MIDI extraction ────────────────────────────────────────────────────────

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

function midiForRange(word: WordTimestamp, start: number, end: number): { midi: number; frameCount: number } {
  const frames = (word as WordWithPitch).pitchFrames;
  if (!frames?.length) return { midi: word.midi, frameCount: 0 };

  for (const threshold of [0.5, 0.3, 0.1]) {
    const values = frames
      .filter((p) => p.time >= start && p.time <= end && p.confidence > threshold)
      .map((p) => p.midi)
      .filter((m) => Number.isFinite(m) && m > 0);
    const value = median(values);
    if (value !== null) return { midi: value, frameCount: values.length };
  }

  return { midi: word.midi, frameCount: 0 };
}

// ── Main alignment ────────────────────────────────────────────────────────

export function alignLyrics(
  lyrics: string,
  whisperWords: WordTimestamp[],
  lang: string,
  pauses: Pause[] = [],
  _songId?: string
): AlignedSyllable[] {
  const startTime = Date.now();
  openLog(_songId || "unknown");

  const rawLines = lyrics.split("\n").map((l) => l.trim()).filter(Boolean);
  const whisperNorm = whisperWords.map((w) => w.word.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]/g, ""));

  log("init", `═══════════════════════════════════════════════════════════`);
  log("init", `Alignment started for song "${_songId || "unknown"}"`);
  log("init", `Language: ${lang}`);
  log("init", `Lyric lines: ${rawLines.length}`);
  log("init", `Whisper words: ${whisperWords.length}`);
  log("init", `═══════════════════════════════════════════════════════════`);

  // ── Build lyric character sequence (preserving line/word structure) ──

  interface LyricChar {
    orig: string;
    norm: string;
    wordIdx: number;
    lineIdx: number;
  }

  const lyricWords: Array<{ word: string; lineIdx: number }> = [];
  for (let li = 0; li < rawLines.length; li++) {
    for (const w of rawLines[li].split(/\s+/).filter(Boolean)) {
      lyricWords.push({ word: w, lineIdx: li });
    }
  }

  const lyricChars: LyricChar[] = [];
  for (let wi = 0; wi < lyricWords.length; wi++) {
    for (const ch of lyricWords[wi].word) {
      lyricChars.push({ orig: ch, norm: normalizeChar(ch), wordIdx: wi, lineIdx: lyricWords[wi].lineIdx });
    }
    if (wi < lyricWords.length - 1) {
      lyricChars.push({ orig: " ", norm: " ", wordIdx: wi, lineIdx: lyricWords[wi].lineIdx });
    }
  }

  // ── Build whisper character sequence (tracking word boundaries) ──

  interface WhisperChar {
    orig: string;
    norm: string;
    wordIdx: number;
  }

  const whisperChars: WhisperChar[] = [];
  for (let wi = 0; wi < whisperWords.length; wi++) {
    const cleaned = whisperWords[wi].word.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    for (const ch of cleaned) {
      whisperChars.push({ orig: ch, norm: normalizeChar(ch), wordIdx: wi });
    }
    if (wi < whisperWords.length - 1) {
      whisperChars.push({ orig: " ", norm: " ", wordIdx: wi });
    }
  }

  log("chars", `Lyric chars: ${lyricChars.length} (${lyricWords.length} words)`);
  log("chars", `Whisper chars: ${whisperChars.length} (${whisperWords.length} words)`);
  log("chars", `Lyric text: "${lyricChars.map(c => c.orig).join("")}"`);
  log("chars", `Whisper text: "${whisperChars.map(c => c.orig).join("")}"`);

  // ── Smith-Waterman alignment ──

  log("sw", `Running Smith-Waterman: ${lyricChars.length} × ${whisperChars.length}`);
  const swStart = Date.now();
  const { maxScore, maxI, maxJ, backtrack } = smithWaterman(
    lyricChars.map(c => c.norm),
    whisperChars.map(c => c.norm)
  );
  log("sw", `SW complete in ${Date.now() - swStart}ms: maxScore=${maxScore.toFixed(2)} at (${maxI}, ${maxJ}), backtrack=${backtrack.length} steps`);

  // ── Log full backtrack to debug ──

  const debugBacktrack: DebugBacktrackStep[] = [];
  for (const step of backtrack) {
    const lc = step.i > 0 && step.i <= lyricChars.length ? lyricChars[step.i - 1] : null;
    const wc = step.j > 0 && step.j <= whisperChars.length ? whisperChars[step.j - 1] : null;
    debugBacktrack.push({
      i: step.i,
      j: step.j,
      matrix: step.matrix,
      score: step.score,
      lyricChar: lc?.orig ?? "-",
      whisperChar: wc?.orig ?? "-",
      whisperWordIdx: wc?.wordIdx ?? -1,
      whisperWord: wc ? whisperWords[wc.wordIdx]?.word ?? "?" : "-",
    });
  }

  log("bt", `Backtrack (${debugBacktrack.length} steps):`);
  for (const step of debugBacktrack) {
    log("bt", `  [${step.matrix}] (${step.i},${step.j}) score=${step.score.toFixed(2)} '${step.lyricChar}'↔'${step.whisperChar}' W[${step.whisperWordIdx}] "${step.whisperWord}"`);
  }

  // ── Extract word-level matches from backtrack ──

  const lyricWordMatched = new Array(lyricWords.length).fill(false);
  const lyricWordWhisperIdxs: number[][] = lyricWords.map(() => []);
  const lyricWordCharAlignments: Array<Array<{ lyricChar: string; whisperChar: string; whisperWordIdx: number; score: number }>> = lyricWords.map(() => []);

  for (const step of backtrack) {
    if (step.matrix === "M" && step.i > 0 && step.j > 0) {
      const lc = lyricChars[step.i - 1];
      const wc = whisperChars[step.j - 1];
      if (lc && wc && lc.norm !== " " && wc.norm !== " ") {
        lyricWordMatched[lc.wordIdx] = true;
        if (!lyricWordWhisperIdxs[lc.wordIdx].includes(wc.wordIdx)) {
          lyricWordWhisperIdxs[lc.wordIdx].push(wc.wordIdx);
        }
        lyricWordCharAlignments[lc.wordIdx].push({
          lyricChar: lc.orig,
          whisperChar: wc.orig,
          whisperWordIdx: wc.wordIdx,
          score: step.score,
        });
      }
    }
  }

  // ── Compute timestamps for matched lyric words ──

  interface WordResult {
    word: string;
    lineIdx: number;
    start: number;
    end: number;
    midi: number;
    source: "sw_aligned" | "interpolated_before" | "interpolated_between" | "interpolated_after";
    whisperIdxs: number[];
    charAlignments: Array<{ lyricChar: string; whisperChar: string; whisperWordIdx: number; score: number }>;
    pitchFrames?: PitchFrame[];
  }

  const wordResults: WordResult[] = lyricWords.map((lw, wi) => ({
    word: lw.word,
    lineIdx: lw.lineIdx,
    start: 0,
    end: 0,
    midi: 60,
    source: "sw_aligned" as const,
    whisperIdxs: lyricWordWhisperIdxs[wi],
    charAlignments: lyricWordCharAlignments[wi],
  }));

  for (let wi = 0; wi < wordResults.length; wi++) {
    if (!lyricWordMatched[wi]) continue;
    const idxs = wordResults[wi].whisperIdxs.slice().sort((a, b) => a - b);
    const starts = idxs.map((idx) => whisperWords[idx].start);
    const ends = idxs.map((idx) => whisperWords[idx].end);
    wordResults[wi].start = Math.min(...starts);
    wordResults[wi].end = Math.max(...ends);

    const allFrames: PitchFrame[] = [];
    for (const idx of idxs) {
      const frames = (whisperWords[idx] as WordWithPitch).pitchFrames;
      if (frames) allFrames.push(...frames);
    }
    if (allFrames.length > 0) {
      wordResults[wi].pitchFrames = allFrames;
      const mr = midiForRange({ word: "", start: wordResults[wi].start, end: wordResults[wi].end, midi: 60, pitchFrames: allFrames } as WordTimestamp, wordResults[wi].start, wordResults[wi].end);
      wordResults[wi].midi = mr.midi;
    } else {
      wordResults[wi].midi = idxs.length > 0 ? whisperWords[idxs[0]].midi : 60;
    }
  }

  // ── Interpolate unmatched words ──

  const matchedIndices: number[] = [];
  for (let wi = 0; wi < wordResults.length; wi++) {
    if (lyricWordMatched[wi]) matchedIndices.push(wi);
  }

  if (matchedIndices.length === 0) {
    log("interp", `No matched words — all words will have default timestamps`);
    for (const wr of wordResults) {
      wr.source = "interpolated_before";
    }
  } else {
    const first = matchedIndices[0];
    const last = matchedIndices[matchedIndices.length - 1];

    // Before first matched word
    for (let wi = 0; wi < first; wi++) {
      wordResults[wi].source = "interpolated_before";
    }
    if (first > 0) {
      const anchorStart = wordResults[first].start;
      const slot = Math.max(0.1, anchorStart / first);
      for (let wi = 0; wi < first; wi++) {
        wordResults[wi].start = Math.max(0, anchorStart - (first - wi) * slot);
        wordResults[wi].end = Math.max(0, anchorStart - (first - wi - 1) * slot);
        wordResults[wi].midi = wordResults[first].midi;
      }
    }

    // Between matched words
    for (let ai = 0; ai < matchedIndices.length - 1; ai++) {
      const a = matchedIndices[ai];
      const b = matchedIndices[ai + 1];
      if (b - a <= 1) continue;

      const tStart = wordResults[a].end;
      const tEnd = wordResults[b].start;
      const duration = Math.max(0, tEnd - tStart);
      const midiA = wordResults[a].midi;
      const midiB = wordResults[b].midi;
      const gaps = b - a;

      for (let k = 1; k < gaps; k++) {
        const wi = a + k;
        wordResults[wi].source = "interpolated_between";
        const frac = k / gaps;
        wordResults[wi].start = tStart + frac * duration;
        wordResults[wi].end = tStart + ((k + 1) / gaps) * duration;
        wordResults[wi].midi = Math.round(midiA + frac * (midiB - midiA));
      }
    }

    // After last matched word
    for (let wi = last + 1; wi < wordResults.length; wi++) {
      wordResults[wi].source = "interpolated_after";
    }
    if (last < wordResults.length - 1) {
      const avgDur = matchedIndices.reduce((sum, idx) => sum + (wordResults[idx].end - wordResults[idx].start), 0) / matchedIndices.length;
      const fallback = Math.max(0.2, avgDur);
      for (let wi = last + 1; wi < wordResults.length; wi++) {
        const offset = (wi - last) * fallback;
        wordResults[wi].start = wordResults[last].end + offset;
        wordResults[wi].end = wordResults[last].end + offset + fallback;
        wordResults[wi].midi = wordResults[last].midi;
      }
    }
  }

  log("words", `Word alignment summary:`);
  for (let wi = 0; wi < wordResults.length; wi++) {
    const wr = wordResults[wi];
    const matchedCount = lyricWordMatched[wi] ? wr.whisperIdxs.length : 0;
    log("words", `  [${wi}] "${wr.word}" ${wr.source} ${matchedCount > 0 ? `→ W[${wr.whisperIdxs.slice().sort().map((idx) => `${idx}("${whisperWords[idx]?.word ?? "?"}")`).join(", ")}]` : "unmatched"} ${wr.start.toFixed(3)}s-${wr.end.toFixed(3)}s midi=${wr.midi}`);
  }

  // ── Syllabification + output ──

  const output: AlignedSyllable[] = [];
  let totalSyllables = 0;
  let totalLineBreaks = 0;
  let currentLineIdx = -1;

  const debugLines: DebugLine[] = [];
  let currentDebugLine: DebugLine | null = null;

  for (let wi = 0; wi < wordResults.length; wi++) {
    const wr = wordResults[wi];

    if (wr.lineIdx !== currentLineIdx) {
      if (currentDebugLine) debugLines.push(currentDebugLine);
      currentLineIdx = wr.lineIdx;
      currentDebugLine = {
        lineIdx: wr.lineIdx,
        lyricLine: rawLines[wr.lineIdx] ?? "",
        words: [],
        syllables: [],
      };
    }

    currentDebugLine!.words.push({
      lyricWord: wr.word,
      lyricNorm: wr.word.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]/g, ""),
      lyricCharRange: [
        lyricChars.findIndex((c, idx) => {
          const wordStart = lyricChars.slice(0, idx).filter(c2 => c2.wordIdx === wi).length === 0;
          return c.wordIdx === wi && wordStart;
        }) >= 0 ? lyricChars.findIndex((c) => c.wordIdx === wi) : 0,
        lyricChars.filter((c) => c.wordIdx <= wi).length - 1,
      ],
      matchedWhisperWordIdxs: wr.whisperIdxs.slice().sort(),
      matchedWhisperWords: wr.whisperIdxs.slice().sort().map((idx) => whisperWords[idx]?.word ?? ""),
      start: wr.start,
      end: wr.end,
      midi: wr.midi,
      source: wr.source,
      charAlignments: wr.charAlignments,
    });

    const syllables = splitWord(wr.word, lang);
    const sylDuration = Math.max(0.01, (wr.end - wr.start) / syllables.length);

    for (let si = 0; si < syllables.length; si++) {
      const sylStart = wr.start + si * sylDuration;
      const sylEnd = wr.start + (si + 1) * sylDuration;

      let midi = wr.midi;
      if (wr.pitchFrames && wr.pitchFrames.length > 0) {
        const mr = midiForRange(
          { word: wr.word, start: sylStart, end: sylEnd, midi: wr.midi, pitchFrames: wr.pitchFrames } as WordTimestamp,
          sylStart,
          sylEnd
        );
        midi = mr.midi;
        if (currentDebugLine) {
          currentDebugLine.syllables.push({
            syllable: syllables[si],
            start: sylStart,
            end: sylEnd,
            midi,
            pitchFrameCount: mr.frameCount,
          });
        }
      } else if (currentDebugLine) {
        currentDebugLine.syllables.push({
          syllable: syllables[si],
          start: sylStart,
          end: sylEnd,
          midi,
          pitchFrameCount: 0,
        });
      }

      output.push({ syllable: syllables[si], start: sylStart, end: sylEnd, midi });
      totalSyllables++;
    }
  }

  if (currentDebugLine) debugLines.push(currentDebugLine);

  // Insert line breaks
  const finalOutput: AlignedSyllable[] = [];
  let lineBreakPos = 0;
  for (let li = 0; li < rawLines.length; li++) {
    const lineWords = rawLines[li].split(/\s+/).filter(Boolean);
    const lineSylCount = lineWords.reduce((sum, w) => {
      const syms = splitWord(w, lang);
      return sum + syms.length;
    }, 0);

    finalOutput.push(...output.slice(lineBreakPos, lineBreakPos + lineSylCount));
    lineBreakPos += lineSylCount;

    if (li < rawLines.length - 1) {
      const nextSyl = output[lineBreakPos];
      if (nextSyl) {
        finalOutput.push({ syllable: "", start: nextSyl.start, end: nextSyl.start, midi: 0, isLineBreak: true });
        totalLineBreaks++;
      }
    }
  }

  // ── Build debug output ──

  const alignedCount = wordResults.filter((wr) => wr.source === "sw_aligned").length;
  const interpCount = wordResults.filter((wr) => wr.source !== "sw_aligned").length;

  debug.songId = _songId || "unknown";
  debug.language = lang;
  debug.lyricCharCount = lyricChars.length;
  debug.whisperCharCount = whisperChars.length;
  debug.whisperWordCount = whisperWords.length;
  debug.whisperTimeRange = [whisperWords[0]?.start ?? 0, whisperWords.at(-1)?.end ?? 0];
  debug.swMaxScore = maxScore;
  debug.swMaxPos = [maxI, maxJ];
  debug.swBacktrackLength = backtrack.length;
  debug.backtrack = debugBacktrack;
  debug.pauses = pauses;
  debug.lines = debugLines;
  debug.summary = {
    totalLyricWords: lyricWords.length,
    alignedWords: alignedCount,
    interpolatedWords: interpCount,
    totalSyllables,
    lineBreaks: totalLineBreaks,
  };

  log("summary", `═══════════════════════════════════════════════════════════`);
  log("summary", `Alignment complete in ${Date.now() - startTime}ms`);
  log("summary", `  Lyric words:    ${lyricWords.length}`);
  log("summary", `  Aligned:        ${alignedCount} (${((alignedCount / lyricWords.length) * 100).toFixed(1)}%)`);
  log("summary", `  Interpolated:   ${interpCount}`);
  log("summary", `  Syllables:      ${totalSyllables}`);
  log("summary", `  Line breaks:    ${totalLineBreaks}`);
  log("summary", `  SW maxScore:    ${maxScore.toFixed(2)} at (${maxI}, ${maxJ})`);
  log("summary", `  SW backtrack:   ${backtrack.length} steps`);
  log("summary", `═══════════════════════════════════════════════════════════`);

  const tmpDir = path.resolve("./tmp");
  fs.mkdirSync(tmpDir, { recursive: true });
  const safeId = _songId ? _songId.replace(/[^a-zA-Z0-9_\-\s]/g, "").trim().replace(/\s+/g, "_") : "alignment";

  log("debug", `Writing debug data to ${path.join(tmpDir, `${safeId}_align_debug.json`)}`);
  log("debug", `Log file written to ${logPath}`);
  closeLog();

  fs.writeFileSync(path.join(tmpDir, `${safeId}_align_debug.json`), JSON.stringify(debug, null, 2));

  return finalOutput;
}
