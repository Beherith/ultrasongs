#!/usr/bin/env node
/**
 * End-to-end pipeline test script.
 *
 * Usage:
 *   npx tsx scripts/test-pipeline.ts <mp3_file> <lyrics_file> [title] [artist]
 *
 * Example:
 *   npx tsx scripts/test-pipeline.ts "Diggy Diggy Hole.mp3" diggy_lyrics.txt "Diggy Diggy Hole" "Siouxsie and the Banshees"
 *
 * Requires:
 *   - Next.js dev server running (pnpm dev)
 *   - Python transcription service running on port 8001
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve } from "path";
import { execSync } from "child_process";

const BASE_URL = process.env.NEXT_DEV_URL ?? "http://localhost:3000";

function getArgs() {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    console.error("Usage: npx tsx scripts/test-pipeline.ts <mp3_file> <lyrics_file> [title] [artist]");
    console.error("Example: npx tsx scripts/test-pipeline.ts song.mp3 lyrics.txt \"Song Title\" \"Artist\"");
    process.exit(1);
  }
  return {
    mp3Path: resolve(args[0]),
    lyricsPath: resolve(args[1]),
    title: args[2] ?? "Test Song",
    artist: args[3] ?? "Test Artist",
  };
}

async function uploadMp3(mp3Path: string): Promise<string> {
  console.log(`[1/3] Uploading MP3: ${mp3Path}`);

  const boundary = `boundary_${Date.now()}`;
  const mp3Data = readFileSync(mp3Path);
  const mp3Name = mp3Path.split("/").pop()?.split("\\").pop() ?? "song.mp3";

  const bodyParts = [
    `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${mp3Name}"\r\nContent-Type: audio/mpeg\r\n\r\n`,
    mp3Data,
    `\r\n--${boundary}--\r\n`,
  ];

  const body = Buffer.concat([
    Buffer.from(bodyParts[0]),
    mp3Data,
    Buffer.from(bodyParts[2]),
  ]);

  const res = await fetch(`${BASE_URL}/api/upload`, {
    method: "POST",
    headers: {
      "Content-Type": `multipart/form-data; boundary=${boundary}`,
    },
    body,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  const json = await res.json();
  console.log(`  → MP3 saved to: ${json.mp3}`);
  return json.mp3;
}

async function transcribe(mp3Path: string, lyrics: string): Promise<{
  words: Array<{ word: string; start: number; end: number; midi: number }>;
  language: string;
  vocalsPath?: string;
  accompanimentPath?: string;
  pauses?: Array<{ start: number; end: number }>;
}> {
  console.log(`[2/3] Transcribing with lyrics (${lyrics.split("\n").filter(Boolean).length} lines)…`);
  console.log("  (this may take a while — GPU processing)");

  const res = await fetch(`${BASE_URL}/api/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mp3: mp3Path, lyrics }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Transcribe failed (${res.status}): ${text}`);
  }

  // Read SSE stream — buffer chunks so lines aren't split at read boundaries
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let result: any = null;
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      // Process any remaining buffered text
      if (buffer.trim()) {
        const remainingLines = buffer.split("\n");
        for (const line of remainingLines) {
          if (line.startsWith("data: ")) {
            try {
              const parsed = JSON.parse(line.slice(6));
              if (parsed.stage) console.log(`  → ${parsed.stage}`);
              else if (parsed.error) throw new Error(`Transcription error: ${parsed.error}`);
              else if (parsed.done) result = parsed;
            } catch (e) {
              if (!(e instanceof SyntaxError)) throw e;
            }
          }
        }
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    // Split on double-newline (SSE event boundary), keep partial trailing text
    const parts = buffer.split("\n\n");
    buffer = parts[parts.length - 1]; // keep incomplete tail
    const completeEvents = parts.slice(0, -1);

    for (const event of completeEvents) {
      const lines = event.split("\n");
      let dataPayload = "";
      for (const line of lines) {
        if (line.startsWith(": ")) continue; // keepalive comment
        if (line.startsWith("data: ")) {
          dataPayload += line.slice(6);
        }
      }
      if (!dataPayload) continue;

      try {
        const parsed = JSON.parse(dataPayload);
        if (parsed.stage) {
          console.log(`  → ${parsed.stage}`);
        } else if (parsed.error) {
          throw new Error(`Transcription error: ${parsed.error}`);
        } else if (parsed.done) {
          result = parsed;
        }
      } catch (e) {
        if (!(e instanceof SyntaxError)) throw e;
      }
    }
  }

  if (!result) throw new Error("Transcription completed but no result received");

  console.log(`  → ${result.words.length} words detected, language: ${result.language}`);
  if (result.pauses?.length) console.log(`  → ${result.pauses.length} pause regions`);
  return result;
}

async function generate(
  mp3Path: string,
  transcribeResult: {
    words: Array<{ word: string; start: number; end: number; midi: number }>;
    language: string;
    vocalsPath?: string;
    accompanimentPath?: string;
    pauses?: Array<{ start: number; end: number }>;
  },
  lyrics: string,
  title: string,
  artist: string
): Promise<string> {
  console.log(`[3/3] Generating Ultrastar package: "${title}" by ${artist}`);

  const res = await fetch(`${BASE_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mp3: mp3Path,
      words: transcribeResult.words,
      language: transcribeResult.language,
      lyrics,
      title,
      artist,
      pauses: transcribeResult.pauses,
      vocalsPath: transcribeResult.vocalsPath,
      accompanimentPath: transcribeResult.accompanimentPath,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Generate failed (${res.status}): ${text}`);
  }

  const zipName = `${title.replace(/[^a-zA-Z0-9\s]/g, "").trim()}.zip`;
  const outputPath = resolve(zipName);
  const buffer = Buffer.from(await res.arrayBuffer());
  writeFileSync(outputPath, buffer);

  console.log(`  → Saved: ${outputPath} (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`);
  return outputPath;
}

async function main() {
  const { mp3Path, lyricsPath, title, artist } = getArgs();

  if (!existsSync(mp3Path)) {
    console.error(`Error: MP3 file not found: ${mp3Path}`);
    process.exit(1);
  }
  if (!existsSync(lyricsPath)) {
    console.error(`Error: Lyrics file not found: ${lyricsPath}`);
    process.exit(1);
  }

  const lyrics = readFileSync(lyricsPath, "utf-8");

  console.log(`\n  Pipeline test: "${title}" by ${artist}`);
  console.log(`  MP3:    ${mp3Path}`);
  console.log(`  Lyrics: ${lyricsPath}`);
  console.log(`  Server: ${BASE_URL}\n`);

  try {
    const mp3ServerPath = await uploadMp3(mp3Path);
    const transcribeResult = await transcribe(mp3ServerPath, lyrics);
    const outputPath = await generate(mp3ServerPath, transcribeResult, lyrics, title, artist);

    console.log(`\n✅ Pipeline complete! Output: ${outputPath}\n`);
  } catch (err) {
    console.error(`\n❌ Pipeline failed:`, err);
    process.exit(1);
  }
}

main();
