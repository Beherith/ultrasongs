# DTW Alignment Implementation Plan

## Problem

The current greedy word-by-word matcher fails when Whisper's word boundaries or
transcriptions diverge from the lyrics:

- **2→1 compression**: "Raise your" → "Racial" (two lyric words, one Whisper word)
- **1→2 expansion**: one lyric word split across two Whisper words
- **Cascade failure**: one wrong match shifts `searchStart`/`lastMatchTime`, derailing
  every subsequent word in the line and potentially following lines

Anchor matching + scoring fixes mitigate this but don't solve the root cause:
greedy per-word decisions can't see the global optimum.

## Goal

Replace the greedy word-by-word matcher (`matchLine`) with a DTW-based segment
aligner that finds the globally optimal lyric→Whisper alignment path within each
anchor-bounded segment.

## Architecture

### Current flow (Phases 1–3)

```
alignLyrics()
  └─ Phase 1: for each line → matchLine() [greedy word-by-word]
       └─ cluster resolution
       └─ line validation
  └─ Phase 2: interpolateMissing() [gap filling]
  └─ Phase 3: syllabification + MIDI extraction
```

### New flow

```
alignLyrics()
  └─ Phase 1a: Anchor detection (unchanged, already implemented)
       └─ Build anchorMap: lyricPos → whisperIdx for exact matches
  └─ Phase 1b: DTW segment alignment [NEW]
       └─ Split line into segments between consecutive anchors
       └─ For each segment → dtwAlign(lyricWords[], whisperWords[])
       └─ Assemble matches from anchor hits + DTW segment results
  └─ Phase 1c: Line validation (unchanged)
  └─ Phase 2: interpolateMissing() (unchanged)
  └─ Phase 3: syllabification + MIDI (unchanged)
```

## DTW Design

### 1. Cost Matrix

Build an `M × N` matrix where `M = lyric segment length`, `N = Whisper segment
length`. Each cell `(i, j)` holds the cost of aligning lyric word `i` to Whisper
word `j`:

```typescript
function cellCost(lyricNorm: string, whisperNorm: string): number {
  // Reuse existing wordScore() — already has phonetic Levenshtein
  return wordScore(lyricNorm, whisperNorm);
}
```

**Delete cost** (Whisper word with no lyric match, e.g., filler words, "um"):
```typescript
const DELETE_COST = 0.7;  // penalized but allowed
```

**Insert cost** (lyric word with no Whisper match):
```typescript
const INSERT_COST = 0.8;  // slightly higher — we prefer matching everything
```

**Merge cost** (multiple lyric words → one Whisper word):
This emerges naturally from DTW — if cells (i, j) and (i+1, j) are both on the
optimal path, lyric words i and i+1 both map to Whisper word j.

### 2. Step Patterns

Allow three step types (standard DTW):

```
(i-1, j-1) → (i, j)    : 1-to-1 match (diagonal)
(i-1, j)   → (i, j)    : insert — lyric word i has no Whisper match
(i, j-1)   → (i, j)    : delete — Whisper word j has no lyric match
```

This naturally handles 2→1 merges: lyric words i and i+1 both map to Whisper j
when the path goes (i-1,j-1) → (i,j) → (i+1,j).

### 3. Cumulative Cost Matrix

```typescript
function dtwAlign(
  lyricWords: string[],       // segment of lyric words
  whisperWords: WordTimestamp[], // segment of Whisper words
  whisperNorm: string[],
): DTWResult {
  const M = lyricWords.length;
  const N = whisperWords.length;

  // Normalize lyric words
  const lyricNorm = lyricWords.map(normalize);

  // Initialize cumulative cost matrix
  const cost: number[][] = Array.from({ length: M + 1 }, () =>
    Array(N + 1).fill(Infinity)
  );
  cost[0][0] = 0;

  // Boundary conditions
  for (let i = 1; i <= M; i++) {
    cost[i][0] = cost[i - 1][0] + INSERT_COST;
  }
  for (let j = 1; j <= N; j++) {
    cost[0][j] = cost[0][j - 1] + DELETE_COST;
  }

  // Fill matrix
  for (let i = 1; i <= M; i++) {
    for (let j = 1; j <= N; j++) {
      const matchCost = cellCost(lyricNorm[i - 1], whisperNorm[j - 1]);
      cost[i][j] = Math.min(
        cost[i - 1][j - 1] + matchCost,    // diagonal: match
        cost[i - 1][j] + INSERT_COST,       // up: insert (lyric unmatched)
        cost[i][j - 1] + DELETE_COST        // left: delete (whisper extra)
      );
    }
  }

  // Backtrack to find optimal path
  const path: Array<{ lyricIdx: number; whisperIdx: number }> = [];
  let i = M, j = N;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0) {
      const matchCost = cellCost(lyricNorm[i - 1], whisperNorm[j - 1]);
      const diag = cost[i - 1][j - 1] + matchCost;
      const up = cost[i - 1][j] + INSERT_COST;
      const left = cost[i][j - 1] + DELETE_COST;
      const min = Math.min(diag, up, left);

      if (Math.abs(min - diag) < 0.001) {
        path.push({ lyricIdx: i - 1, whisperIdx: j - 1 });
        i--; j--;
      } else if (Math.abs(min - up) < 0.001) {
        path.push({ lyricIdx: i - 1, whisperIdx: -1 });
        i--;
      } else {
        path.push({ lyricIdx: -1, whisperIdx: j - 1 });
        j--;
      }
    } else if (i > 0) {
      path.push({ lyricIdx: i - 1, whisperIdx: -1 });
      i--;
    } else {
      path.push({ lyricIdx: -1, whisperIdx: j - 1 });
      j--;
    }
  }
  path.reverse();

  return { path, totalCost: cost[M][N] };
}
```

### 4. Temporal Constraints

DTW must respect temporal ordering. Add constraints during matrix fill:

```typescript
// Whisper word j cannot start before the previous matched whisper word ends
function isTemporallyValid(lyricIdx: number, whisperIdx: number, prevWhisperIdx: number): boolean {
  if (prevWhisperIdx < 0) return true;
  return whisperWords[whisperIdx].start >= whisperWords[prevWhisperIdx].start - 0.5;
}

// Whisper word j cannot be too far in the future from lyric word i's expected position
function isWithinTimeWindow(lyricIdx: number, whisperIdx: number, segmentStart: number): boolean {
  const maxGap = 5; // seconds
  return whisperWords[whisperIdx].start - segmentStart <= maxGap + lyricIdx * 1.5;
}
```

Set invalid cells to `Infinity` so they're never chosen.

### 5. Post-Path Processing

After backtracking, convert the DTW path into the `matched: Array<WordTimestamp | null>`
format expected by downstream code:

```typescript
function pathToMatches(
  path: Array<{ lyricIdx: number; whisperIdx: number }>,
  lyricWords: string[],
  whisperWords: WordTimestamp[],
): Array<WordTimestamp | null> {
  const matched: Array<WordTimestamp | null> = new Array(lyricWords.length).fill(null);

  for (const step of path) {
    if (step.lyricIdx >= 0 && step.whisperIdx >= 0) {
      matched[step.lyricIdx] = whisperWords[step.whisperIdx];
    }
  }

  return matched;
}
```

## Integration with Existing Code

### Segment Construction

The `matchLine` function becomes a coordinator:

```typescript
function matchLine(/* same signature */) {
  // 1. Anchor detection (already implemented)
  const anchorMap = detectAnchors(lineWords, whisperWords, whisperNorm, ss, lmt);

  // 2. Build segments between anchors
  const segments = buildSegments(lineWords.length, anchorMap, whisperWords);
  // e.g., [{ lyricStart: 0, lyricEnd: 1, whisperStart: 11, whisperEnd: 12 }]
  //       [{ lyricStart: 4, lyricEnd: 5, whisperStart: 14, whisperEnd: 15 }]

  // 3. Run DTW on each segment
  const allMatched: Array<WordTimestamp | null> = new Array(lineWords.length).fill(null);

  // Place anchor matches
  for (const [pos, idx] of anchorMap) {
    allMatched[pos] = whisperWords[idx];
  }

  // Run DTW on gaps
  for (const seg of segments) {
    const lyricSeg = lineWords.slice(seg.lyricStart, seg.lyricEnd + 1);
    const whisperSeg = whisperWords.slice(seg.whisperStart, seg.whisperEnd + 1);
    const normSeg = whisperNorm.slice(seg.whisperStart, seg.whisperEnd + 1);

    const { path, totalCost } = dtwAlign(lyricSeg, whisperSeg, normSeg);
    const segMatches = pathToMatches(path, lyricSeg, whisperSeg);

    for (let i = 0; i < segMatches.length; i++) {
      allMatched[seg.lyricStart + i] = segMatches[i];
    }
  }

  // 4. Validation (unchanged)
  // ...
}
```

### Segment Building Logic

```typescript
interface Segment {
  lyricStart: number;
  lyricEnd: number;
  whisperStart: number;
  whisperEnd: number;
}

function buildSegments(
  lineWordCount: number,
  anchorMap: Map<number, number>,
  whisperWords: WordTimestamp[],
): Segment[] {
  const segments: Segment[] = [];
  const sortedAnchors = [...anchorMap.entries()].sort((a, b) => a[0] - b[0]);

  if (sortedAnchors.length === 0) {
    // No anchors — DTW the entire line (bounded by search window)
    const whisperEnd = findSearchEnd(whisperWords, lmt);
    segments.push({
      lyricStart: 0,
      lyricEnd: lineWordCount - 1,
      whisperStart: ss,
      whisperEnd,
    });
    return segments;
  }

  // Gap before first anchor
  if (sortedAnchors[0][0] > 0) {
    segments.push({
      lyricStart: 0,
      lyricEnd: sortedAnchors[0][0] - 1,
      whisperStart: ss,
      whisperEnd: sortedAnchors[0][1] - 1,
    });
  }

  // Gaps between consecutive anchors
  for (let i = 0; i < sortedAnchors.length - 1; i++) {
    const [lyricA, whisperA] = sortedAnchors[i];
    const [lyricB, whisperB] = sortedAnchors[i + 1];

    if (lyricB > lyricA + 1) {
      // There are unmatched lyric words between these anchors
      segments.push({
        lyricStart: lyricA + 1,
        lyricEnd: lyricB - 1,
        whisperStart: whisperA + 1,
        whisperEnd: whisperB - 1,
      });
    }
  }

  // Gap after last anchor
  const [lastLyric, lastWhisper] = sortedAnchors[sortedAnchors.length - 1];
  if (lastLyric < lineWordCount - 1) {
    const whisperEnd = findSearchEnd(whisperWords, whisperWords[lastWhisper].start);
    segments.push({
      lyricStart: lastLyric + 1,
      lyricEnd: lineWordCount - 1,
      whisperStart: lastWhisper + 1,
      whisperEnd,
    });
  }

  return segments;
}
```

### Example: "Raise your pick and raise your voice!"

```
Anchors: pick→[12], and→[13], voice→[15]

Segments:
  [0]: lyric "Raise your" (pos 0-1) ↔ whisper [11..11] = ["Racial"]
  [1]: lyric "raise your" (pos 4-5) ↔ whisper [14..14] = ["racial"]

DTW on segment [0]:
  Cost matrix (2×1):
         Racial
  Raise   0.50
  your    0.55

  Optimal path: (Raise→Racial), (your→INSERT)
  Result: Raise→Whisper[11], your→null

DTW on segment [1]:
  Cost matrix (2×1):
         racial
  raise   0.50
  your    0.55

  Optimal path: (raise→racial), (your→INSERT)
  Result: raise→Whisper[14], your→null

Final matches:
  Raise → Whisper[11] "Racial"    ✓
  your  → null                     (interpolated later)
  pick  → Whisper[12] "pick"      ✓ (anchor)
  and   → Whisper[13] "and"       ✓ (anchor)
  raise → Whisper[14] "racial"    ✓
  your  → null                     (interpolated later)
  voice → Whisper[15] "voice"     ✓ (anchor)
```

## Implementation Steps

### Step 1: Add DTW core function (new file or section)

- `dtwAlign()` — cost matrix + backtrack, returns path
- `pathToMatches()` — converts DTW path to `Array<WordTimestamp | null>`
- Constants: `DELETE_COST`, `INSERT_COST`

**File**: Add as a new section in `align.ts` before `// ── Line matching ──`

### Step 2: Add segment building

- `buildSegments()` — splits line into anchor-bounded segments
- `findSearchEnd()` — finds the last Whisper word within the search window

**File**: Same section as DTW core

### Step 3: Refactor `matchLine()` to use DTW

- Keep anchor detection (already implemented)
- Replace the greedy `for (let wi = 0; ...)` loop with:
  1. `buildSegments()` call
  2. DTW on each segment
  3. Anchor placement
  4. Assembly into `matched` array
- Keep cluster resolution and validation (may become less critical but still useful as safety net)

### Step 4: Debug integration

- Extend `DebugWordMatch` with `dtwSegment?: string` and `dtwCost?: number`
- Log DTW cost matrices for each segment
- Log optimal path for debugging

### Step 5: Testing

- Run against "Diggy Diggy Hole" — verify line 2 matches correctly
- Run against a song with clean Whisper output — verify no regression
- Run against a song with heavy Whisper errors — verify robustness

## Constants to Tune

| Constant | Initial Value | Purpose |
|----------|--------------|---------|
| `DELETE_COST` | 0.7 | Cost of skipping a Whisper word |
| `INSERT_COST` | 0.8 | Cost of unmatched lyric word |
| `MAX_SEGMENT_SIZE` | 20 | Max Whisper words per DTW segment (O(M×N) memory) |
| `MERGE_PENALTY` | 0.1 | Extra cost when multiple lyric words map to one Whisper word |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Large segments → O(M×N) memory/time | Cap segment size at 20 words; split if needed |
| DTW chooses wrong path for ambiguous text | Anchors constrain segments; temporal constraints prune invalid cells |
| Regression on clean songs | Anchors provide exact matches; DTW on 0-length gaps is trivial |
| Debug complexity | Log cost matrix summary (dimensions, total cost, path length) |

## What Stays the Same

- **Phase 2 (interpolation)**: unchanged — still fills gaps for unmatched words
- **Phase 3 (syllabification + MIDI)**: unchanged — consumes `WordTimestamp[]`
- **`wordScore()` / `normalize()` / `levenshtein()`**: reused as DTW cell cost
- **File logger**: unchanged
- **Debug JSON output**: extended with DTW fields, same structure
- **`alignLyrics()` signature**: unchanged — drop-in replacement
