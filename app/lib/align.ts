import { splitWord } from "./syllabify";
import type { WordTimestamp, Pause } from "../api/transcribe/route";

export interface AlignedSyllable {
  syllable: string;
  start: number; // seconds
  end: number;   // seconds
  midi: number;
  isLineBreak?: boolean;
}

/** Levenshtein distance between two strings (case-insensitive, stripped). */
function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

function normalize(w: string): string {
  return w
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "") // strip accents
    .replace(/[^a-z0-9]/g, "");
}

/**
 * Text similarity score between a lyric word and a Whisper word.
 * 0 = perfect match, approaching 1 = bad.
 *
 * Handles the "numeratorTwo" → "two" case: if the lyric word is a suffix
 * of the Whisper token (Whisper sometimes concatenates adjacent words),
 * treat it as a near-perfect match instead of computing full edit distance.
 */
function wordScore(lNorm: string, wNorm: string): number {
  if (wNorm === lNorm) return 0;
  // Suffix match: "numeratortwo" ends with "two" → near-perfect
  if (lNorm.length >= 3 && wNorm.length > lNorm.length && wNorm.endsWith(lNorm)) return 0.05;
  // Prefix match: "they're" starts with "they" → good
  if (lNorm.length >= 3 && wNorm.startsWith(lNorm)) return 0.1;
  const dist = levenshtein(lNorm, wNorm);
  return dist / Math.max(lNorm.length, wNorm.length, 1);
}

/**
 * Matches lyric words to Whisper words, starting search from `searchStart`.
 * Returns [matched[], newSearchStart, newLastMatchTime].
 *
 * Scoring = text similarity + soft time-jump penalty.
 * Penalty: each second beyond 20s from the last match costs 0.015 score points,
 * making a 57-second jump cost +0.55 — enough to prefer a nearby suffix match.
 */
function matchLine(
  lineWords: string[],
  whisperWords: WordTimestamp[],
  whisperNorm: string[],
  searchStart: number,
  lastMatchTime: number
): { matched: Array<WordTimestamp | null>; searchStart: number; lastMatchTime: number } {
  const matched: Array<WordTimestamp | null> = [];
  let ss = searchStart;
  let lmt = lastMatchTime;

  for (const lw of lineWords) {
    const lNorm = normalize(lw);
    if (!lNorm) { matched.push(null); continue; }

    let bestIdx = -1;
    let bestScore = Infinity;

    for (let i = ss; i < whisperWords.length; i++) {
      const ts = wordScore(lNorm, whisperNorm[i]);

      // Soft penalty for large forward jumps (beyond 20 s from last matched word)
      let timePenalty = 0;
      if (lmt >= 0) {
        const jump = Math.max(0, whisperWords[i].start - lmt - 20);
        timePenalty = jump * 0.015;
      }

      const score = ts + timePenalty;
      if (score < bestScore) { bestScore = score; bestIdx = i; }
      if (bestScore === 0) break;
    }

    // Short words (≤ 2 chars: "I", "a", "to") need a stricter threshold —
    // they appear everywhere and cause false positives at wrong timestamps.
    const threshold = lNorm.length <= 2 ? 0.25 : 0.65;

    if (bestIdx >= 0 && bestScore < threshold) {
      ss = bestIdx + 1;
      lmt = whisperWords[bestIdx].start;
      matched.push(whisperWords[bestIdx]);
    } else {
      matched.push(null);
    }
  }

  return { matched, searchStart: ss, lastMatchTime: lmt };
}

/**
 * For unmatched words, interpolate timing linearly between surrounding anchors.
 * Also averages MIDI from neighbors.
 */
function interpolateMissing(
  matched: Array<WordTimestamp | null>,
  lyricWords: string[]
): WordTimestamp[] {
  const result: Array<WordTimestamp> = matched.map((m, i) =>
    m ?? { word: lyricWords[i], start: -1, end: -1, midi: 60 }
  );

  const anchors = result
    .map((r, i) => (r.start >= 0 ? i : -1))
    .filter((i) => i >= 0);

  if (anchors.length === 0) return result;

  // Fill before first anchor
  for (let i = 0; i < anchors[0]; i++) {
    result[i].start = result[anchors[0]].start;
    result[i].end = result[anchors[0]].start;
    result[i].midi = result[anchors[0]].midi;
  }

  // Fill after last anchor
  const last = anchors[anchors.length - 1];
  const FALLBACK_WORD_SEC = 0.3;
  for (let i = last + 1; i < result.length; i++) {
    const offset = (i - last) * FALLBACK_WORD_SEC;
    result[i].start = result[last].end + offset;
    result[i].end = result[last].end + offset + FALLBACK_WORD_SEC;
    result[i].midi = result[last].midi;
  }

  // Interpolate between consecutive anchors
  for (let ai = 0; ai < anchors.length - 1; ai++) {
    const a = anchors[ai];
    const b = anchors[ai + 1];
    const gap = b - a;
    if (gap <= 1) continue;
    const tStart = result[a].end;
    const tEnd = result[b].start;
    const duration = tEnd - tStart;
    const midiA = result[a].midi;
    const midiB = result[b].midi;
    for (let k = 1; k < gap; k++) {
      const frac = k / gap;
      result[a + k].start = tStart + frac * duration;
      result[a + k].end = tStart + ((k + 1) / gap) * duration;
      result[a + k].midi = Math.round(midiA + frac * (midiB - midiA));
    }
  }

  return result;
}

/**
 * Aligns Whisper word timestamps to user-provided lyrics.
 *
 * Processes line-by-line. After each line, uses the pause list to advance
 * the Whisper search pointer past the next silence region — this anchors
 * each lyric paragraph to the correct vocal phrase and prevents the matcher
 * from pulling words from a completely different section of the song.
 */
export function alignLyrics(
  lyrics: string,
  whisperWords: WordTimestamp[],
  lang: string,
  pauses: Pause[] = []
): AlignedSyllable[] {
  const lines = lyrics.split("\n").map((l) => l.trim()).filter(Boolean);
  const whisperNorm = whisperWords.map((w) => normalize(w.word));
  const sortedPauses = [...pauses].sort((a, b) => a.start - b.start);

  const allMatched: Array<WordTimestamp | null> = [];
  const allLyricWords: string[] = [];

  let searchStart = 0;
  let lastMatchTime = -1;

  for (let li = 0; li < lines.length; li++) {
    const lineWords = lines[li].split(/\s+/).filter(Boolean);
    allLyricWords.push(...lineWords);

    const res = matchLine(lineWords, whisperWords, whisperNorm, searchStart, lastMatchTime);
    allMatched.push(...res.matched);
    searchStart = res.searchStart;
    lastMatchTime = res.lastMatchTime;

    // After each lyric line, look for the next pause after the last matched time.
    // Advance searchStart past that pause so the next line's words aren't pulled
    // from within a silence region (where Whisper may hallucinate filler words).
    if (lastMatchTime >= 0 && li < lines.length - 1) {
      const nextPause = sortedPauses.find(
        (p) => p.start >= lastMatchTime - 0.3 && p.end > lastMatchTime
      );
      if (nextPause) {
        // Skip Whisper words that fall inside this silence region
        const afterPause = whisperWords.findIndex((w) => w.start >= nextPause.end);
        if (afterPause > searchStart) {
          searchStart = afterPause;
          // Nudge lastMatchTime to the pause end so time-penalty stays accurate
          lastMatchTime = nextPause.end;
        }
      }
    }
  }

  const interpolated = interpolateMissing(allMatched, allLyricWords);

  // Reconstruct per-line structure with line-break markers
  const output: AlignedSyllable[] = [];
  let wordIdx = 0;

  for (let li = 0; li < lines.length; li++) {
    const lineWords = lines[li].split(/\s+/).filter(Boolean);

    for (const word of lineWords) {
      const ts = interpolated[wordIdx++];
      const syllables = splitWord(word, lang);
      const sylDuration = (ts.end - ts.start) / syllables.length;

      syllables.forEach((syl, si) => {
        output.push({
          syllable: syl,
          start: ts.start + si * sylDuration,
          end: ts.start + (si + 1) * sylDuration,
          midi: ts.midi,
        });
      });
    }

    if (li < lines.length - 1) {
      const nextLineStart = interpolated[wordIdx]?.start ?? output[output.length - 1]?.end ?? 0;
      output.push({ syllable: "", start: nextLineStart, end: nextLineStart, midi: 0, isLineBreak: true });
    }
  }

  return output;
}
