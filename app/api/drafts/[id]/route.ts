import { NextRequest, NextResponse } from "next/server";
import { readFile, rm } from "fs/promises";
import { existsSync } from "fs";
import path from "path";

const DRAFTS_DIR = path.resolve(process.env.DRAFTS_DIR ?? "./drafts");

export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await ctx.params;
    const dir = path.join(DRAFTS_DIR, id);
    const jsonPath = path.join(dir, "draft.json");
    if (!existsSync(jsonPath)) {
      return NextResponse.json({ message: "Draft not found" }, { status: 404 });
    }
    const data = JSON.parse(await readFile(jsonPath, "utf-8"));
    return NextResponse.json({
      ...data,
      mp3: path.join(dir, data.audioFilename),
      videoPath: data.videoFilename ? path.join(dir, data.videoFilename) : undefined,
      vocalsPath: data.vocalsFilename ? path.join(dir, data.vocalsFilename) : undefined,
      accompanimentPath: data.accompanimentFilename ? path.join(dir, data.accompanimentFilename) : undefined,
    });
  } catch (err) {
    return NextResponse.json({ message: err instanceof Error ? err.message : "Failed" }, { status: 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  ctx: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await ctx.params;
    const dir = path.join(DRAFTS_DIR, id);
    if (!existsSync(dir)) {
      return NextResponse.json({ message: "Draft not found" }, { status: 404 });
    }
    await rm(dir, { recursive: true, force: true });
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ message: err instanceof Error ? err.message : "Failed" }, { status: 500 });
  }
}
