import { NextRequest, NextResponse } from "next/server";
import { readdir, readFile, writeFile, mkdir, copyFile } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import crypto from "crypto";

const DRAFTS_DIR = path.resolve(process.env.DRAFTS_DIR ?? "./drafts");

async function ensureDraftsDir() {
  if (!existsSync(DRAFTS_DIR)) await mkdir(DRAFTS_DIR, { recursive: true });
}

export async function GET() {
  try {
    await ensureDraftsDir();
    const entries = await readdir(DRAFTS_DIR, { withFileTypes: true });
    const drafts = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const jsonPath = path.join(DRAFTS_DIR, entry.name, "draft.json");
      if (!existsSync(jsonPath)) continue;
      try {
        const { id, title, artist, savedAt, videoFilename } = JSON.parse(
          await readFile(jsonPath, "utf-8")
        );
        drafts.push({ id, title, artist, savedAt, hasVideo: !!videoFilename });
      } catch { /* skip corrupt draft */ }
    }

    drafts.sort((a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime());
    return NextResponse.json(drafts);
  } catch (err) {
    return NextResponse.json({ message: err instanceof Error ? err.message : "Failed" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    await ensureDraftsDir();
    const body = await request.json();
    const {
      draftId, title, artist, lyrics, bpm, gap, notes, words, language, pauses,
      mp3, videoPath, vocalsPath, accompanimentPath,
    } = body;

    const id: string = draftId ?? crypto.randomUUID();
    const dir = path.join(DRAFTS_DIR, id);
    await mkdir(dir, { recursive: true });

    const audioFilename = "audio.mp3";
    if (mp3 && existsSync(mp3)) await copyFile(mp3, path.join(dir, audioFilename));

    let videoFilename: string | undefined;
    if (videoPath && existsSync(videoPath)) {
      videoFilename = "video" + path.extname(videoPath).toLowerCase();
      await copyFile(videoPath, path.join(dir, videoFilename));
    }

    let vocalsFilename: string | undefined;
    if (vocalsPath && existsSync(vocalsPath)) {
      vocalsFilename = "vocals.mp3";
      await copyFile(vocalsPath, path.join(dir, vocalsFilename));
    }

    let accompanimentFilename: string | undefined;
    if (accompanimentPath && existsSync(accompanimentPath)) {
      accompanimentFilename = "accompaniment.mp3";
      await copyFile(accompanimentPath, path.join(dir, accompanimentFilename));
    }

    const draft = {
      id, title, artist, lyrics: lyrics ?? "", savedAt: new Date().toISOString(),
      bpm, gap, notes, words, language, pauses: pauses ?? [],
      audioFilename, videoFilename, vocalsFilename, accompanimentFilename,
    };

    await writeFile(path.join(dir, "draft.json"), JSON.stringify(draft));
    return NextResponse.json({ id });
  } catch (err) {
    return NextResponse.json({ message: err instanceof Error ? err.message : "Failed" }, { status: 500 });
  }
}
