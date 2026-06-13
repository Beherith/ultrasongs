import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { existsSync } from "fs";
import path from "path";

const TMP_DIR = path.resolve(process.env.TMP_DIR ?? "./tmp");
const DRAFTS_DIR = path.resolve(process.env.DRAFTS_DIR ?? "./drafts");

function isAllowedPath(resolved: string) {
  const r = resolved.toLowerCase();
  return r.startsWith(TMP_DIR.toLowerCase()) || r.startsWith(DRAFTS_DIR.toLowerCase());
}

export async function GET(request: NextRequest) {
  const mp3 = request.nextUrl.searchParams.get("mp3");

  if (!mp3) {
    return NextResponse.json({ message: "Missing mp3 param" }, { status: 400 });
  }

  const resolved = path.resolve(mp3);
  if (!isAllowedPath(resolved)) {
    return NextResponse.json({ message: "Invalid path" }, { status: 403 });
  }

  if (!existsSync(resolved)) {
    return NextResponse.json({ message: "File not found" }, { status: 404 });
  }

  const buf = await readFile(resolved);
  return new NextResponse(buf, {
    headers: {
      "Content-Type": "audio/mpeg",
      "Content-Length": buf.byteLength.toString(),
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    },
  });
}
