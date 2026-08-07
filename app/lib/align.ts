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

const MAX_FORWARD_SEARCH_SEC = 90;
const MAX_IN_LINE_GAP_SEC = 4;

// ── File-based logger ─────────────────────────────────────────────────────

let logFd: number | null = null;
let logPath: string = "";

function openLog(songId: string) {
  const tmpDir = path.resolve("./tmp");
  fs.mkdirSync(tmpDir, { recursive: true });
  const safeId = songId ? songId.replace(/[^a-zA-Z0-9_\-\s]/g, "").trim().replace(/\s+/g, "_") : "alignment";
  logPath = path.join(tmpDir, `${safeId}_align.log`);
  // Truncate then open for writing
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

// ── Debug collection ──────────────────────────────────────────────────────

interface DebugWordMatch {
  lyricWord: string;
  lyricNorm: string;
  matchedWord: string | null;
  matchedNorm: string | null;
  matchedWhisperIdx: number | null;
  textScore: number | null;
  combinedScore: number | null;
  maxTextScore: number;
  timeJump: number | null;
  start: number | null;
  end: number | null;
  midi: number | null;
  reason: string;
}

interface DebugLine {
  lineIdx: number;
  lyricLine: string;
  words: DebugWordMatch[];
  searchStart: number;
  lastMatchTimeBefore: number;
  lastMatchTimeAfter: number;
  clusters: Array<{ positions: number[]; whisperIdxs: number[]; spanSec: number }>;
  clusterResolution: string | null;
  validation: string | null;
  interpolated: Array<{ word: string; start: number; end: number; midi: number; source: string }>;
  syllables: Array<{ syllable: string; start: number; end: number; midi: number; pitchFrameCount: number }>;
}

interface DebugRoot {
  songId: string;
  language: string;
  lyricLineCount: number;
  lyricWordCount: number;
  whisperWordCount: number;
  whisperTimeRange: [number, number];
  pauses: Pause[];
  lines: DebugLine[];
  summary: {
    totalLyricWords: number;
    matchedWords: number;
    unmatchedWords: number;
    interpolatedBefore: number;
    interpolatedBetween: number;
    interpolatedAfter: number;
    totalSyllables: number;
    lineBreaks: number;
  };
}

const debug: DebugRoot = {
  songId: "",
  language: "",
  lyricLineCount: 0,
  lyricWordCount: 0,
  whisperWordCount: 0,
  whisperTimeRange: [0, 0],
  pauses: [],
  lines: [],
  summary: {
    totalLyricWords: 0,
    matchedWords: 0,
    unmatchedWords: 0,
    interpolatedBefore: 0,
    interpolatedBetween: 0,
    interpolatedAfter: 0,
    totalSyllables: 0,
    lineBreaks: 0,
  },
};

// ── Phonetic matching ─────────────────────────────────────────────────────

function phoneticCost(x: string, y: string): number {
  if (x === y) return 0;
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
    ["d", "th"],
  ];
  for (const g of groups) {
    const ix = g.indexOf(x);
    const iy = g.indexOf(y);
    if (ix >= 0 && iy >= 0) {
      return 0.3 + 0.15 * Math.abs(ix - iy);
    }
  }
  const cross: [string, string][] = [
    ["a", "o"], ["a", "u"], ["e", "i"], ["e", "o"], ["i", "y"],
    ["s", "sh"], ["z", "zh"], ["f", "ph"], ["c", "k"], ["q", "k"],
    ["w", "u"], ["r", "l"], ["b", "p"], ["d", "t"], ["g", "k"],
  ];
  for (const [a, b] of cross) {
    if ((x === a && y === b) || (x === b && y === a)) return 0.4;
  }
  return 1;
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const subCost = phoneticCost(a[i - 1], b[j - 1]);
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + subCost
      );
    }
  }

  return dp[m][n];
}

function normalize(w: string): string {
  return w
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]/g, "");
}

function wordScore(lNorm: string, wNorm: string): number {
  if (wNorm === lNorm) return 0;
  if (lNorm.length >= 3 && wNorm.length > lNorm.length && wNorm.endsWith(lNorm)) return 0.03;
  if (lNorm.length >= 3 && wNorm.startsWith(lNorm)) return 0.06;

  const dist = levenshtein(lNorm, wNorm);
  return dist / Math.max(lNorm.length, wNorm.length, 1);
}

function maxTextScore(lNorm: string): number {
  if (lNorm.length <= 2) return 0.05;
  if (lNorm.length === 3) return 0.35;
  if (lNorm.length <= 5) return 0.55;
  return 0.55;
}

// ── Line matching ─────────────────────────────────────────────────────────

function matchLine(
  lineIdx: number,
  lineWords: string[],
  whisperWords: WordTimestamp[],
  whisperNorm: string[],
  searchStart: number,
  lastMatchTime: number
): {
  matched: Array<WordTimestamp | null>;
  searchStart: number;
  lastMatchTime: number;
  debugWords: DebugWordMatch[];
  clusters: Array<{ positions: number[]; whisperIdxs: number[]; spanSec: number }>;
  clusterResolution: string | null;
  validation: string | null;
} {
  log("match:line", `Line ${lineIdx}: "${lineWords.join(" ")}" (${lineWords.length} words), searchStart=${searchStart}, lastMatchTime=${lastMatchTime.toFixed(2)}s`);

  const matched: Array<WordTimestamp | null> = [];
  const debugWords: DebugWordMatch[] = [];
  let ss = searchStart;
  let lmt = lastMatchTime;
  const matchedIndices: Array<number | null> = [];

  for (let wi = 0; wi < lineWords.length; wi++) {
    const lw = lineWords[wi];
    const lNorm = normalize(lw);
    const mts = maxTextScore(lNorm);

    log("match:word", `  Line ${lineIdx} word ${wi}: "${lw}" → norm "${lNorm}", maxTextScore=${mts.toFixed(2)}`);

    if (!lNorm) {
      log("match:word", `    → SKIPPED (empty after normalization)`);
      matched.push(null);
      matchedIndices.push(null);
      debugWords.push({
        lyricWord: lw,
        lyricNorm: lNorm,
        matchedWord: null,
        matchedNorm: null,
        matchedWhisperIdx: null,
        textScore: null,
        combinedScore: null,
        maxTextScore: mts,
        timeJump: null,
        start: null,
        end: null,
        midi: null,
        reason: "empty_after_normalize",
      });
      continue;
    }

    let bestIdx = -1;
    let bestScore = Infinity;
    let bestTextScore = Infinity;
    let bestJump = 0;
    let candidatesConsidered = 0;
    let candidatesPassedText = 0;

    const searchEnd = lmt >= 0 ? lmt + MAX_FORWARD_SEARCH_SEC : MAX_FORWARD_SEARCH_SEC;
    log("match:word", `    Search window: Whisper[${ss}..${Math.min(ss + 200, whisperWords.length - 1)}], time limit ${searchEnd.toFixed(2)}s`);

    for (let i = ss; i < whisperWords.length; i++) {
      const wStart = whisperWords[i].start;

      if (lmt >= 0 && wStart > lmt + MAX_FORWARD_SEARCH_SEC) {
        log("match:word", `    → Whisper[${i}] "${whisperWords[i].word}" at ${wStart.toFixed(2)}s EXCEEDS forward search (${lmt.toFixed(2)} + ${MAX_FORWARD_SEARCH_SEC}s = ${searchEnd.toFixed(2)}s), stopping search`);
        break;
      }
      if (lmt < 0 && wStart > MAX_FORWARD_SEARCH_SEC) {
        log("match:word", `    → Whisper[${i}] "${whisperWords[i].word}" at ${wStart.toFixed(2)}s EXCEEDS initial search window (${MAX_FORWARD_SEARCH_SEC}s), stopping search`);
        break;
      }

      candidatesConsidered++;
      const textScore = wordScore(lNorm, whisperNorm[i]);

      if (textScore > mts) {
        continue;
      }
      candidatesPassedText++;

      const jump = lmt >= 0
        ? Math.max(0, wStart - lmt - 20)
        : Math.max(0, wStart - 20);
      const score = textScore + jump * (lmt >= 0 ? 0.015 : 0.05);

      if (i - ss < 5 || score < bestScore) {
        log("match:word", `      Whisper[${i}] "${whisperWords[i].word}" (${whisperNorm[i]}): textScore=${textScore.toFixed(3)}, jump=${jump.toFixed(2)}s, combined=${score.toFixed(3)} ${score < bestScore ? "← NEW BEST" : ""}`);
      }

      if (score < bestScore) {
        bestScore = score;
        bestTextScore = textScore;
        bestIdx = i;
        bestJump = jump;
      }
      if (bestScore === 0) {
        log("match:word", `      → Perfect match, breaking early`);
        break;
      }
    }

    log("match:word", `    Searched ${candidatesConsidered} candidates, ${candidatesPassedText} passed text threshold`);

    if (bestIdx >= 0 && bestTextScore <= mts) {
      const ww = whisperWords[bestIdx];
      ss = bestIdx + 1;
      lmt = ww.start;
      matched.push(ww);
      matchedIndices.push(bestIdx);

      log("match:word", `    → MATCHED: Whisper[${bestIdx}] "${ww.word}" (${whisperNorm[bestIdx]}) at ${ww.start.toFixed(2)}s-${ww.end.toFixed(2)}s, textScore=${bestTextScore.toFixed(3)}, combined=${bestScore.toFixed(3)}, jump=${bestJump.toFixed(2)}s, midi=${ww.midi}`);

      debugWords.push({
        lyricWord: lw,
        lyricNorm: lNorm,
        matchedWord: ww.word,
        matchedNorm: whisperNorm[bestIdx],
        matchedWhisperIdx: bestIdx,
        textScore: bestTextScore,
        combinedScore: bestScore,
        maxTextScore: mts,
        timeJump: bestJump,
        start: ww.start,
        end: ww.end,
        midi: ww.midi,
        reason: "matched",
      });
    } else {
      matched.push(null);
      matchedIndices.push(null);

      log("match:word", `    → NO MATCH (bestIdx=${bestIdx}, bestTextScore=${bestTextScore === Infinity ? "∞" : bestTextScore.toFixed(3)}, maxTextScore=${mts.toFixed(2)})`);

      debugWords.push({
        lyricWord: lw,
        lyricNorm: lNorm,
        matchedWord: null,
        matchedNorm: null,
        matchedWhisperIdx: null,
        textScore: bestTextScore === Infinity ? null : bestTextScore,
        combinedScore: bestScore === Infinity ? null : bestScore,
        maxTextScore: mts,
        timeJump: null,
        start: null,
        end: null,
        midi: null,
        reason: bestIdx < 0 ? "no_candidates_in_window" : "text_score_exceeds_threshold",
      });
    }
  }

  // ── Cluster resolution ─────────────────────────────────────────────────

  const clusters: Array<{ positions: number[]; whisperIdxs: number[]; spanSec: number }> = [];
  const clusterGroups: Array<Array<number>> = [];

  for (let pos = 0; pos < matched.length; pos++) {
    const idx = matchedIndices[pos];
    if (idx === null) continue;
    const prevCluster = clusterGroups[clusterGroups.length - 1];
    const prevPos = prevCluster?.[prevCluster.length - 1];
    const prevIdx = prevPos === undefined ? null : matchedIndices[prevPos];
    const startsNewCluster =
      prevIdx !== null &&
      whisperWords[idx].start - whisperWords[prevIdx].start > MAX_IN_LINE_GAP_SEC;

    if (!prevCluster || startsNewCluster) clusterGroups.push([pos]);
    else prevCluster.push(pos);
  }

  for (const group of clusterGroups) {
    const idxs = group.map((p) => matchedIndices[p]!).filter((i): i is number => i !== null);
    const times = idxs.map((i) => whisperWords[i].start);
    const span = times.length > 1 ? Math.max(...times) - Math.min(...times) : 0;
    clusters.push({
      positions: group,
      whisperIdxs: idxs,
      spanSec: span,
    });
  }

  let clusterResolution: string | null = null;

  if (clusters.length > 1) {
    log("match:cluster", `  Line ${lineIdx}: ${clusters.length} clusters detected:`);
    clusters.forEach((c, ci) => {
      log("match:cluster", `    Cluster ${ci}: positions=${JSON.stringify(c.positions)}, whisperIdxs=${JSON.stringify(c.whisperIdxs)}, span=${c.spanSec.toFixed(2)}s`);
    });

    const keep = clusters.reduce((best, cluster) =>
      cluster.positions.length > best.positions.length ? cluster : best
    );
    clusterResolution = `multi_cluster: kept cluster with ${keep.positions.length} matches, discarded ${clusters.length - 1} other cluster(s)`;
    log("match:cluster", `    → Keeping cluster with ${keep.positions.length} matches`);

    const keepSet = new Set(keep.positions);
    for (let pos = 0; pos < matched.length; pos++) {
      if (!keepSet.has(pos)) {
        const discardedWord = lineWords[pos];
        const discardedIdx = matchedIndices[pos];
        log("match:cluster", `    → Discarding position ${pos} "${discardedWord}" (was Whisper[${discardedIdx}])`);
        debugWords[pos].reason = "discarded_by_cluster_resolution";
        matched[pos] = null;
        matchedIndices[pos] = null;
      }
    }
    const lastKeptPos = keep.positions[keep.positions.length - 1];
    const lastKeptIdx = matchedIndices[lastKeptPos];
    if (lastKeptIdx !== null) {
      ss = lastKeptIdx + 1;
      lmt = whisperWords[lastKeptIdx].start;
    }
  } else if (clusters.length === 1) {
    clusterResolution = `single_cluster: ${clusters[0].positions.length} matches, span=${clusters[0].spanSec.toFixed(2)}s`;
    log("match:cluster", `  Line ${lineIdx}: single cluster, ${clusters[0].positions.length} matches, span=${clusters[0].spanSec.toFixed(2)}s`);
  } else {
    clusterResolution = "no_clusters: no matches found";
    log("match:cluster", `  Line ${lineIdx}: no clusters (no matches)`);
  }

  // ── Line validation ────────────────────────────────────────────────────

  const keptIndices = matchedIndices.filter((idx): idx is number => idx !== null);
  const keptTimes = keptIndices.map((idx) => whisperWords[idx].start);
  const keptSignificantMatches = matched.filter((m) => m && normalize(m.word).length > 2).length;
  const lineSpan = keptTimes.length > 1 ? Math.max(...keptTimes) - Math.min(...keptTimes) : 0;
  const maxLineSpan = Math.max(12, lineWords.length * 3);

  let validation: string | null = null;

  if (lineWords.length >= 4 && keptSignificantMatches === 0) {
    validation = `rejected: ${lineWords.length} words but 0 significant matches`;
    log("match:validate", `  Line ${lineIdx}: REJECTED — ${lineWords.length} words, ${keptSignificantMatches} significant matches (need ≥1)`);
    const nullDebug = lineWords.map((lw) => ({
      lyricWord: lw,
      lyricNorm: normalize(lw),
      matchedWord: null as string | null,
      matchedNorm: null as string | null,
      matchedWhisperIdx: null as number | null,
      textScore: null as number | null,
      combinedScore: null as number | null,
      maxTextScore: maxTextScore(normalize(lw)),
      timeJump: null as number | null,
      start: null as number | null,
      end: null as number | null,
      midi: null as number | null,
      reason: "line_rejected_no_significant_matches",
    }));
    return {
      matched: lineWords.map(() => null),
      searchStart,
      lastMatchTime,
      debugWords: nullDebug,
      clusters,
      clusterResolution,
      validation,
    };
  }

  if (lineWords.length >= 4 && lineSpan > maxLineSpan) {
    validation = `rejected: span ${lineSpan.toFixed(2)}s exceeds max ${maxLineSpan}s`;
    log("match:validate", `  Line ${lineIdx}: REJECTED — span ${lineSpan.toFixed(2)}s > maxLineSpan ${maxLineSpan}s`);
    const nullDebug = lineWords.map((lw) => ({
      lyricWord: lw,
      lyricNorm: normalize(lw),
      matchedWord: null as string | null,
      matchedNorm: null as string | null,
      matchedWhisperIdx: null as number | null,
      textScore: null as number | null,
      combinedScore: null as number | null,
      maxTextScore: maxTextScore(normalize(lw)),
      timeJump: null as number | null,
      start: null as number | null,
      end: null as number | null,
      midi: null as number | null,
      reason: "line_rejected_span_exceeds_max",
    }));
    return {
      matched: lineWords.map(() => null),
      searchStart,
      lastMatchTime,
      debugWords: nullDebug,
      clusters,
      clusterResolution,
      validation,
    };
  }

  if (lineWords.length >= 4) {
    validation = `passed: ${keptSignificantMatches} significant matches, span ${lineSpan.toFixed(2)}s ≤ ${maxLineSpan}s`;
    log("match:validate", `  Line ${lineIdx}: PASSED — ${keptSignificantMatches} significant matches, span ${lineSpan.toFixed(2)}s ≤ ${maxLineSpan}s`);
  } else {
    validation = `skipped: line has ${lineWords.length} words (<4, no validation)`;
    log("match:validate", `  Line ${lineIdx}: VALIDATION SKIPPED — only ${lineWords.length} words (<4)`);
  }

  const matchCount = matched.filter((m) => m !== null).length;
  log("match:line", `  Line ${lineIdx} result: ${matchCount}/${lineWords.length} matched, searchStart→${ss}, lastMatchTime→${lmt.toFixed(2)}s`);

  return { matched, searchStart: ss, lastMatchTime: lmt, debugWords, clusters, clusterResolution, validation };
}

// ── Interpolation ─────────────────────────────────────────────────────────

function interpolateMissing(
  matched: Array<WordTimestamp | null>,
  lyricWords: string[]
): { result: WordTimestamp[]; debug: Array<{ word: string; start: number; end: number; midi: number; source: string }> } {
  log("interpolate", `Starting interpolation: ${matched.length} words, ${matched.filter((m) => m !== null).length} matched, ${matched.filter((m) => m === null).length} unmatched`);

  const result: Array<WordTimestamp> = matched.map((m, i) =>
    m ?? { word: lyricWords[i], start: -1, end: -1, midi: 60 }
  );

  const debugInterp: Array<{ word: string; start: number; end: number; midi: number; source: string }> = [];
  const anchors = result.map((r, i) => (r.start >= 0 ? i : -1)).filter((i) => i >= 0);

  log("interpolate", `Anchors at indices: [${anchors.join(", ")}]`);

  if (anchors.length === 0) {
    log("interpolate", `  No anchors — all words unmatched, returning defaults`);
    for (let i = 0; i < result.length; i++) {
      debugInterp.push({ word: lyricWords[i], start: -1, end: -1, midi: 60, source: "no_anchors_default" });
    }
    return { result, debug: debugInterp };
  }

  const first = anchors[0];
  const last = anchors[anchors.length - 1];

  // Before first anchor
  if (first > 0) {
    const firstAnchorStart = result[first].start;
    const start = Math.max(0, firstAnchorStart - first);
    const slot = (firstAnchorStart - start) / first;
    log("interpolate", `  Before first anchor [${first}]: ${first} words, start=${start.toFixed(3)}s, slot=${slot.toFixed(3)}s, anchorStart=${firstAnchorStart.toFixed(3)}s`);
    for (let i = 0; i < first; i++) {
      result[i].start = start + i * slot;
      result[i].end = start + (i + 1) * slot;
      result[i].midi = result[first].midi;
      log("interpolate", `    Word ${i} "${lyricWords[i]}": ${result[i].start.toFixed(3)}s-${result[i].end.toFixed(3)}s, midi=${result[i].midi}`);
      debugInterp.push({ word: lyricWords[i], start: result[i].start, end: result[i].end, midi: result[i].midi, source: `interpolated_before_anchor_${first}` });
    }
  } else {
    for (let i = 0; i < first; i++) {
      debugInterp.push({ word: lyricWords[i], start: result[i].start, end: result[i].end, midi: result[i].midi, source: "anchor" });
    }
  }

  // Between anchors
  for (let ai = 0; ai < anchors.length - 1; ai++) {
    const a = anchors[ai];
    const b = anchors[ai + 1];
    const gap = b - a;

    if (a < first) {
      debugInterp.push({ word: lyricWords[a], start: result[a].start, end: result[a].end, midi: result[a].midi, source: "anchor" });
      continue;
    }

    if (gap <= 1) {
      debugInterp.push({ word: lyricWords[a], start: result[a].start, end: result[a].end, midi: result[a].midi, source: "anchor" });
      continue;
    }

    const tStart = result[a].end;
    const tEnd = result[b].start;
    const duration = tEnd - tStart;
    const midiA = result[a].midi;
    const midiB = result[b].midi;

    log("interpolate", `  Between anchors [${a}] and [${b}]: ${gap - 1} words, tStart=${tStart.toFixed(3)}s, tEnd=${tEnd.toFixed(3)}s, duration=${duration.toFixed(3)}s, midi ${midiA}→${midiB}`);

    debugInterp.push({ word: lyricWords[a], start: result[a].start, end: result[a].end, midi: result[a].midi, source: "anchor" });

    for (let k = 1; k < gap; k++) {
      const frac = k / gap;
      result[a + k].start = tStart + frac * duration;
      result[a + k].end = tStart + ((k + 1) / gap) * duration;
      result[a + k].midi = Math.round(midiA + frac * (midiB - midiA));
      log("interpolate", `    Word ${a + k} "${lyricWords[a + k]}": frac=${frac.toFixed(3)}, ${result[a + k].start.toFixed(3)}s-${result[a + k].end.toFixed(3)}s, midi=${result[a + k].midi}`);
      debugInterp.push({ word: lyricWords[a + k], start: result[a + k].start, end: result[a + k].end, midi: result[a + k].midi, source: `interpolated_between_${a}_${b}` });
    }
  }

  // After last anchor
  log("interpolate", `  After last anchor [${last}]: ${result.length - 1 - last} words, fallback=0.3s/word, anchorEnd=${result[last].end.toFixed(3)}s`);
  debugInterp.push({ word: lyricWords[last], start: result[last].start, end: result[last].end, midi: result[last].midi, source: "anchor" });

  const fallbackWordSec = 0.3;
  for (let i = last + 1; i < result.length; i++) {
    const offset = (i - last) * fallbackWordSec;
    result[i].start = result[last].end + offset;
    result[i].end = result[last].end + offset + fallbackWordSec;
    result[i].midi = result[last].midi;
    log("interpolate", `    Word ${i} "${lyricWords[i]}": ${result[i].start.toFixed(3)}s-${result[i].end.toFixed(3)}s, midi=${result[i].midi}`);
    debugInterp.push({ word: lyricWords[i], start: result[i].start, end: result[i].end, midi: result[i].midi, source: `interpolated_after_anchor_${last}` });
  }

  return { result, debug: debugInterp };
}

// ── MIDI extraction ───────────────────────────────────────────────────────

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

function midiForRange(word: WordTimestamp, start: number, end: number): { midi: number; frameCount: number; threshold: number | null } {
  const frames = (word as WordWithPitch).pitchFrames;
  if (!frames?.length) {
    log("midi", `    No pitch frames available for word "${word.word}", fallback midi=${word.midi}`);
    return { midi: word.midi, frameCount: 0, threshold: null };
  }

  for (const threshold of [0.5, 0.3, 0.1]) {
    const values = frames
      .filter((p) => p.time >= start && p.time <= end && p.confidence > threshold)
      .map((p) => p.midi)
      .filter((m) => Number.isFinite(m) && m > 0);
    const value = median(values);
    if (value !== null) {
      log("midi", `    Pitch frames: threshold=${threshold}, ${values.length} frames in [${start.toFixed(3)}, ${end.toFixed(3)}], median midi=${value}`);
      return { midi: value, frameCount: values.length, threshold };
    }
  }

  log("midi", `    No confident pitch frames in [${start.toFixed(3)}, ${end.toFixed(3)}], fallback midi=${word.midi}`);
  return { midi: word.midi, frameCount: 0, threshold: null };
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
  const lines = lyrics.split("\n").map((l) => l.trim()).filter(Boolean);
  const whisperNorm = whisperWords.map((w) => normalize(w.word));

  log("init", `═══════════════════════════════════════════════════════════`);
  log("init", `Log file: ${logPath}`);
  log("init", `Alignment started for song "${_songId || "unknown"}"`);
  log("init", `Language: ${lang}`);
  log("init", `Lyric lines: ${lines.length}`);
  log("init", `Whisper words: ${whisperWords.length}`);
  log("init", `Whisper time range: ${whisperWords[0]?.start?.toFixed(2) ?? "N/A"}s — ${whisperWords.at(-1)?.end?.toFixed(2) ?? "N/A"}s`);
  log("init", `Pauses: ${pauses.length} regions`);
  if (pauses.length > 0) {
    log("init", `Pause regions: ${pauses.map((p) => `${p.start.toFixed(2)}s-${p.end.toFixed(2)}s`).join(", ")}`);
  }
  log("init", `═══════════════════════════════════════════════════════════`);

  // Whisper word summary
  log("whisper", `All ${whisperWords.length} Whisper words:`);
  whisperWords.forEach((w, i) => {
    log("whisper", `  [${i}] "${w.word}" (${normalize(w.word)}) ${w.start.toFixed(2)}s-${w.end.toFixed(2)}s midi=${w.midi} pitchFrames=${((w as WordWithPitch).pitchFrames?.length ?? 0)}`);
  });

  debug.songId = _songId || "unknown";
  debug.language = lang;
  debug.lyricLineCount = lines.length;
  debug.whisperWordCount = whisperWords.length;
  debug.whisperTimeRange = [whisperWords[0]?.start ?? 0, whisperWords.at(-1)?.end ?? 0];
  debug.pauses = pauses;

  const allMatched: Array<WordTimestamp | null> = [];
  const allLyricWords: string[] = [];
  const allDebugWords: DebugWordMatch[] = [];
  let searchStart = 0;
  let lastMatchTime = -1;

  // ── Phase 1: Match each line ──────────────────────────────────────────

  log("phase", `── Phase 1: Line matching ────────────────────────────────`);

  for (let li = 0; li < lines.length; li++) {
    const lineWords = lines[li].split(/\s+/).filter(Boolean);
    allLyricWords.push(...lineWords);

    log("line", `▶ Line ${li}: "${lines[li]}" (${lineWords.length} words)`);

    const res = matchLine(li, lineWords, whisperWords, whisperNorm, searchStart, lastMatchTime);
    allMatched.push(...res.matched);
    allDebugWords.push(...res.debugWords);

    const lineDebug: DebugLine = {
      lineIdx: li,
      lyricLine: lines[li],
      words: res.debugWords,
      searchStart,
      lastMatchTimeBefore: lastMatchTime,
      lastMatchTimeAfter: res.lastMatchTime,
      clusters: res.clusters,
      clusterResolution: res.clusterResolution,
      validation: res.validation,
      interpolated: [],
      syllables: [],
    };

    searchStart = res.searchStart;
    lastMatchTime = res.lastMatchTime;

    const lineMatchCount = res.matched.filter((m) => m !== null).length;
    log("line", `◀ Line ${li}: ${lineMatchCount}/${lineWords.length} matched, clusterRes="${res.clusterResolution}", validation="${res.validation}"`);

    debug.lines.push(lineDebug);
  }

  const matchedCount = allMatched.filter((m) => m !== null).length;
  log("phase", `Phase 1 complete: ${matchedCount}/${allMatched.length} total words matched`);

  // ── Phase 2: Interpolate missing words ────────────────────────────────

  log("phase", `── Phase 2: Gap interpolation ────────────────────────────`);
  const { result: interpolated, debug: interpDebug } = interpolateMissing(allMatched, allLyricWords);

  const stillNegative = interpolated.filter((w) => w.start < 0).length;
  log("phase", `Phase 2 complete: ${stillNegative} words still have negative timestamps (no anchors at all)`);

  // Update debug with interpolation data
  for (const line of debug.lines) {
    const lineWordCount = line.words.length;
    let wordOffset = debug.lines.indexOf(line) > 0
      ? debug.lines.slice(0, debug.lines.indexOf(line)).reduce((sum, l) => sum + l.words.length, 0)
      : 0;
    for (let wi = 0; wi < lineWordCount; wi++) {
      const globalIdx = wordOffset + wi;
      if (interpDebug[globalIdx]) {
        line.interpolated.push(interpDebug[globalIdx]);
      }
    }
  }

  // ── Phase 3: Syllabification + MIDI extraction ────────────────────────

  log("phase", `── Phase 3: Syllabification + MIDI extraction ────────────`);
  const output: AlignedSyllable[] = [];
  let wordIdx = 0;
  let totalSyllables = 0;
  let totalLineBreaks = 0;

  for (let li = 0; li < lines.length; li++) {
    const lineWords = lines[li].split(/\s+/).filter(Boolean);
    const lineDebug = debug.lines[li];

    for (const word of lineWords) {
      const ts = interpolated[wordIdx];
      const syllables = splitWord(word, lang);
      const sylDuration = Math.max(0, ts.end - ts.start) / syllables.length;

      log("syl", `  Word "${word}" → ${syllables.length} syllable(s): [${syllables.join(") | (")}], duration=${(ts.end - ts.start).toFixed(3)}s, sylDuration=${sylDuration.toFixed(3)}s`);

      syllables.forEach((syl, si) => {
        const start = ts.start + si * sylDuration;
        const end = ts.start + (si + 1) * sylDuration;

        let midiResult: { midi: number; frameCount: number; threshold: number | null };
        if (ts.start >= 0 && (ts as WordWithPitch).pitchFrames?.length) {
          midiResult = midiForRange(ts, start, end);
        } else {
          midiResult = { midi: ts.midi, frameCount: 0, threshold: null };
          log("midi", `    Word "${word}" syl "${syl}": no pitch data, using word midi=${ts.midi}`);
        }

        output.push({
          syllable: syl,
          start,
          end,
          midi: midiResult.midi,
        });

        if (lineDebug) {
          lineDebug.syllables.push({
            syllable: syl,
            start,
            end,
            midi: midiResult.midi,
            pitchFrameCount: midiResult.frameCount,
          });
        }

        totalSyllables++;
      });

      wordIdx++;
    }

    if (li < lines.length - 1) {
      const nextLineStart = interpolated[wordIdx]?.start ?? output[output.length - 1]?.end ?? 0;
      log("linebreak", `  Line break after line ${li}: placed at ${nextLineStart.toFixed(3)}s (next line first word start)`);
      output.push({ syllable: "", start: nextLineStart, end: nextLineStart, midi: 0, isLineBreak: true });
      totalLineBreaks++;
    }
  }

  log("phase", `Phase 3 complete: ${totalSyllables} syllables, ${totalLineBreaks} line breaks`);

  // ── Summary ───────────────────────────────────────────────────────────

  const unmatchedWords = allMatched.filter((m) => m === null).length;
  const beforeCount = interpDebug.filter((d) => d.source.startsWith("interpolated_before")).length;
  const betweenCount = interpDebug.filter((d) => d.source.startsWith("interpolated_between")).length;
  const afterCount = interpDebug.filter((d) => d.source.startsWith("interpolated_after")).length;

  debug.summary = {
    totalLyricWords: allLyricWords.length,
    matchedWords: matchedCount,
    unmatchedWords,
    interpolatedBefore: beforeCount,
    interpolatedBetween: betweenCount,
    interpolatedAfter: afterCount,
    totalSyllables,
    lineBreaks: totalLineBreaks,
  };

  log("summary", `═══════════════════════════════════════════════════════════`);
  log("summary", `Alignment complete in ${Date.now() - startTime}ms`);
  log("summary", `  Lyric words:    ${allLyricWords.length}`);
  log("summary", `  Matched:        ${matchedCount} (${((matchedCount / allLyricWords.length) * 100).toFixed(1)}%)`);
  log("summary", `  Unmatched:      ${unmatchedWords}`);
  log("summary", `    Before first: ${beforeCount}`);
  log("summary", `    Between:      ${betweenCount}`);
  log("summary", `    After last:   ${afterCount}`);
  log("summary", `  Syllables:      ${totalSyllables}`);
  log("summary", `  Line breaks:    ${totalLineBreaks}`);
  log("summary", `═══════════════════════════════════════════════════════════`);

  // ── Write debug JSON ──────────────────────────────────────────────────

  const tmpDir = path.resolve("./tmp");
  fs.mkdirSync(tmpDir, { recursive: true });
  const safeId = _songId ? _songId.replace(/[^a-zA-Z0-9_\-\s]/g, "").trim().replace(/\s+/g, "_") : "alignment";

  log("debug", `Writing debug data to ${path.join(tmpDir, `${safeId}_align_debug.json`)}`);
  log("debug", `Log file written to ${logPath}`);
  closeLog();

  fs.writeFileSync(path.join(tmpDir, `${safeId}_align_debug.json`), JSON.stringify(debug, null, 2));

  return output;
}
