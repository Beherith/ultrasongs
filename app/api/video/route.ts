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
  const videoParam = request.nextUrl.searchParams.get("path");

  if (!videoParam) {
    return NextResponse.json({ message: "Missing path param" }, { status: 400 });
  }

  const resolved = path.resolve(videoParam);
  if (!isAllowedPath(resolved)) {
    return NextResponse.json({ message: "Invalid path" }, { status: 403 });
  }

  if (!existsSync(resolved)) {
    return NextResponse.json({ message: "File not found" }, { status: 404 });
  }

  const ext = path.extname(resolved).toLowerCase();
  const contentType = ext === ".webm" ? "video/webm" : ext === ".mov" ? "video/quicktime" : "video/mp4";

  const buf = await readFile(resolved);
  return new NextResponse(buf, {
    headers: {
      "Content-Type": contentType,
      "Content-Length": buf.byteLength.toString(),
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    },
  });
}
