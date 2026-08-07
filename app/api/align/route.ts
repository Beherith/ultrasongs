import { NextRequest, NextResponse } from "next/server";
import { existsSync } from "fs";
import { writeFile } from "fs/promises";
import { alignLyrics } from "@/app/lib/align";
import { detectBpm } from "@/app/lib/ultrastar";
import path from "path";
import type { WordTimestamp, Pause } from "@/app/api/transcribe/route";
import type { EditorNote } from "@/app/lib/editorNote";

interface AlignRequest {
  mp3: string;
  words: WordTimestamp[];
  language: string;
  lyrics: string;
  pauses?: Pause[];
}

export async function POST(request: NextRequest) {
  try {
    const { mp3, words, language, lyrics, pauses }: AlignRequest = await request.json();

    if (!mp3 || !words?.length || !lyrics) {
      return NextResponse.json({ message: "Missing required fields" }, { status: 400 });
    }
    if (!existsSync(mp3)) {
      return NextResponse.json({ message: "Audio file not found" }, { status: 404 });
    }

    const songId = path.basename(mp3, ".mp3");
    const [aligned, { bpm }] = await Promise.all([
      Promise.resolve(alignLyrics(lyrics, words, language, pauses, songId)),
      detectBpm(mp3),
    ]);

    const duration = words.at(-1)?.end ?? 0;

    const notes: EditorNote[] = aligned
      .filter((s) => !s.isLineBreak)
      .map((s, i) => ({
        id: `n${i}`,
        syllable: s.syllable,
        startSec: s.start,
        durationSec: Math.max(0.1, s.end - s.start),
        pitch: s.midi,
        type: ":" as const,
      }));

    // GAP = first note start minus a small lead-in (ms)
    const firstStart = notes[0]?.startSec ?? 0;
    const gap = Math.max(0, firstStart * 1000 - 500);

    const result = { notes, bpm, gap, duration };

    // Save alignment result as JSON
    const jsonPath = path.join(path.dirname(mp3), path.basename(mp3, ".mp3")) + "_align_notes.json";
    await writeFile(jsonPath, JSON.stringify(result, null, 2));
    console.log(`[align] Saved alignment: ${jsonPath}`);

    return NextResponse.json(result);
  } catch (err) {
    console.error("[align]", err);
    return NextResponse.json(
      { message: err instanceof Error ? err.message : "Alignment failed" },
      { status: 500 }
    );
  }
}
