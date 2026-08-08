#!/usr/bin/env node
/**
 * End-to-end pipeline test script.
 *
 * Named arguments:
 *   --mp3 <path>        Path to MP3 file
 *   --lyrics <path>     Path to lyrics (.txt) file
 *   --title <string>    Song title
 *   --artist <string>   Artist name
 *   --url <url>         Next.js server URL (default: http://localhost:3000)
 *   --stage <stage>     Stage to run: upload | transcribe | align | generate | all (default: all)
 *   --from-json <path>  Skip earlier stages, load input from this JSON file
 *   --to-json <path>    Override output JSON path (default: auto-generated)
 *
 * Examples:
 *   # Full pipeline
 *   npx tsx scripts/test-pipeline.ts --mp3 "Diggy Diggy Hole.mp3" --lyrics diggy_lyrics.txt --title "Diggy Diggy Hole" --artist "Siouxsie and the Banshees"
 *
 *   # Upload only
 *   npx tsx scripts/test-pipeline.ts --stage upload --mp3 "Diggy Diggy Hole.mp3"
 *
 *   # Transcribe only (uses upload output)
 *   npx tsx scripts/test-pipeline.ts --stage transcribe --from-json upload_output.json --lyrics diggy_lyrics.txt
 *
 *   # Align only (tests alignment logic in isolation)
 *   npx tsx scripts/test-pipeline.ts --stage align --from-json transcribe_output.json --lyrics diggy_lyrics.txt
 *
 *   # Generate only (uses transcribe output)
 *   npx tsx scripts/test-pipeline.ts --stage generate --from-json transcribe_output.json --lyrics diggy_lyrics.txt --title "Diggy Diggy Hole" --artist "Siouxsie and the Banshees"
 *
 * Requires:
 *   - Next.js dev server running (pnpm dev)
 *   - Python transcription service running on port 8001 (for transcribe stage)
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve, basename, dirname, extname } from "path";

const BASE_URL = process.env.NEXT_DEV_URL ?? "http://localhost:3000";

type Stage = "upload" | "transcribe" | "align" | "generate" | "all";

interface UploadResult {
  mp3: string;
}

interface TranscribeInput {
  mp3: string;
  lyrics: string;
}

interface TranscribeResult {
  words: Array<{ word: string; start: number; end: number; midi: number }>;
  language: string;
  vocalsPath?: string;
  accompanimentPath?: string;
  pauses?: Array<{ start: number; end: number }>;
}

interface AlignInput {
  mp3: string;
  words: Array<{ word: string; start: number; end: number; midi: number }>;
  language: string;
  lyrics: string;
  pauses?: Array<{ start: number; end: number }>;
}

interface AlignResult {
  notes: Array<{
    id: string;
    syllable: string;
    startSec: number;
    durationSec: number;
    pitch: number;
    type: string;
  }>;
  bpm: number;
  gap: number;
  duration: number;
}

interface GenerateInput {
  mp3: string;
  words: Array<{ word: string; start: number; end: number; midi: number }>;
  language: string;
  lyrics: string;
  title: string;
  artist: string;
  pauses?: Array<{ start: number; end: number }>;
  vocalsPath?: string;
  accompanimentPath?: string;
}

// ── Argument parsing ────────────────────────────────────────────────────────

function parseArgs(): {
  stage: Stage;
  mp3Path?: string;
  lyricsPath?: string;
  title?: string;
  artist?: string;
  fromJson?: string;
  toJson?: string;
  url: string;
} {
  const args = process.argv.slice(2);
  const parsed: Record<string, string | undefined> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--") && i + 1 < args.length) {
      parsed[args[i].slice(2)] = args[i + 1];
      i++;
    }
  }

  const stage = (parsed.stage as Stage) ?? "all";
  if (!["upload", "transcribe", "align", "generate", "all"].includes(stage)) {
    console.error(`Invalid stage: ${stage}. Must be: upload, transcribe, align, generate, all`);
    process.exit(1);
  }

  return {
    stage,
    mp3Path: parsed.mp3 ? resolve(parsed.mp3) : undefined,
    lyricsPath: parsed.lyrics ? resolve(parsed.lyrics) : undefined,
    title: parsed.title,
    artist: parsed.artist,
    fromJson: parsed["from-json"] ? resolve(parsed["from-json"]) : undefined,
    toJson: parsed["to-json"],
    url: parsed.url ?? BASE_URL,
  };
}

// ── Stages ──────────────────────────────────────────────────────────────────

async function runUpload(url: string, mp3Path: string): Promise<UploadResult> {
  console.log(`[upload] Uploading MP3: ${mp3Path}`);

  const boundary = `boundary_${Date.now()}`;
  const mp3Data = readFileSync(mp3Path);
  const mp3Name = basename(mp3Path);

  const body = Buffer.concat([
    Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${mp3Name}"\r\nContent-Type: audio/mpeg\r\n\r\n`),
    mp3Data,
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);

  const res = await fetch(`${url}/api/upload`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body,
  });

  if (!res.ok) throw new Error(`Upload failed (${res.status}): ${await res.text()}`);

  const result: UploadResult = await res.json();
  console.log(`  → MP3 saved to: ${result.mp3}`);
  return result;
}

async function runTranscribe(url: string, input: TranscribeInput): Promise<TranscribeResult> {
  console.log(`[transcribe] Transcribing with lyrics (${input.lyrics.split("\n").filter(Boolean).length} lines)…`);
  console.log("  (this may take a while — GPU processing)");

  const res = await fetch(`${url}/api/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mp3: input.mp3, lyrics: input.lyrics }),
  });

  if (!res.ok) throw new Error(`Transcribe failed (${res.status}): ${await res.text()}`);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let result: TranscribeResult | null = null;
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) {
        for (const line of buffer.split("\n")) {
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
    const parts = buffer.split("\n\n");
    buffer = parts[parts.length - 1];

    for (const event of parts.slice(0, -1)) {
      let dataPayload = "";
      for (const line of event.split("\n")) {
        if (line.startsWith(": ")) continue;
        if (line.startsWith("data: ")) dataPayload += line.slice(6);
      }
      if (!dataPayload) continue;

      try {
        const parsed = JSON.parse(dataPayload);
        if (parsed.stage) console.log(`  → ${parsed.stage}`);
        else if (parsed.error) throw new Error(`Transcription error: ${parsed.error}`);
        else if (parsed.done) result = parsed;
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

async function runAlign(url: string, input: AlignInput): Promise<AlignResult> {
  console.log(`[align] Aligning lyrics (${input.lyrics.split("\n").filter(Boolean).length} lines) to ${input.words.length} words…`);

  const res = await fetch(`${url}/api/align`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) throw new Error(`Align failed (${res.status}): ${await res.text()}`);

  const result: AlignResult = await res.json();
  console.log(`  → ${result.notes.length} aligned notes, BPM: ${result.bpm}, gap: ${result.gap}ms`);
  console.log(`  → Duration: ${result.duration.toFixed(1)}s`);

  // Print alignment summary
  const syllables = result.notes.map((n) => n.syllable).join(" ");
  const lines = input.lyrics.split("\n").filter(Boolean);
  console.log(`  → Lyrics: ${lines.length} lines`);
  console.log(`  → First note: "${result.notes[0]?.syllable}" at ${result.notes[0]?.startSec.toFixed(3)}s`);
  if (result.notes.length > 1) {
    const last = result.notes[result.notes.length - 1];
    console.log(`  → Last note:  "${last.syllable}" at ${last.startSec.toFixed(3)}s`);
  }

  return result;
}

async function runGenerate(url: string, input: GenerateInput): Promise<string> {
  console.log(`[generate] Generating Ultrastar package: "${input.title}" by ${input.artist}`);

  const res = await fetch(`${url}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) throw new Error(`Generate failed (${res.status}): ${await res.text()}`);

  const zipName = `${input.title.replace(/[^a-zA-Z0-9\s]/g, "").trim()}.zip`;
  const outputPath = resolve(zipName);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(outputPath, buf);
  console.log(`  → Saved: ${outputPath} (${(buf.length / 1024 / 1024).toFixed(2)} MB)`);
  return outputPath;
}

// ── JSON save/load helpers ──────────────────────────────────────────────────

function saveJson(data: unknown, defaultPath: string, overridePath?: string): string {
  const path = overridePath ?? defaultPath;
  writeFileSync(path, JSON.stringify(data, null, 2));
  console.log(`  → Saved JSON: ${path}`);
  return path;
}

function loadJson<T>(path: string): T {
  if (!existsSync(path)) {
    console.error(`Error: JSON file not found: ${path}`);
    process.exit(1);
  }
  return JSON.parse(readFileSync(path, "utf-8")) as T;
}

function defaultToJsonPath(stage: Stage, sourcePath: string): string {
  const base = sourcePath.replace(/\.[^.]+$/, "");
  return `${base}_${stage}.json`;
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();
  const url = args.url;

  console.log(`\n  Pipeline test — stage: ${args.stage}, server: ${url}\n`);

  try {
    if (args.stage === "all" || args.stage === "upload") {
      // ── UPLOAD ──────────────────────────────────────────────────────────
      if (!args.mp3Path) {
        console.error("Error: --mp3 is required for upload stage");
        process.exit(1);
      }
      if (!existsSync(args.mp3Path)) {
        console.error(`Error: MP3 file not found: ${args.mp3Path}`);
        process.exit(1);
      }

      const uploadResult = await runUpload(url, args.mp3Path);
      const uploadJson = saveJson(uploadResult, defaultToJsonPath("upload", args.mp3Path), args.toJson);

      if (args.stage === "upload") {
        console.log(`\n✅ Upload complete.\n`);
        return;
      }

      // Pass to transcribe
      if (args.stage === "all") {
        if (!args.lyricsPath) {
          console.error("Error: --lyrics is required for transcribe stage");
          process.exit(1);
        }
        if (!existsSync(args.lyricsPath)) {
          console.error(`Error: Lyrics file not found: ${args.lyricsPath}`);
          process.exit(1);
        }

        const transcribeInput: TranscribeInput = {
          mp3: uploadResult.mp3,
          lyrics: readFileSync(args.lyricsPath, "utf-8"),
        };

        const transcribeResult = await runTranscribe(url, transcribeInput);
        const transcribeJson = saveJson(transcribeResult, defaultToJsonPath("transcribe", args.mp3Path));

        if (!args.title || !args.artist) {
          console.error("Error: --title and --artist are required for generate stage");
          process.exit(1);
        }

        const generateInput: GenerateInput = {
          mp3: uploadResult.mp3,
          words: transcribeResult.words,
          language: transcribeResult.language,
          lyrics: transcribeInput.lyrics,
          title: args.title,
          artist: args.artist,
          pauses: transcribeResult.pauses,
          vocalsPath: transcribeResult.vocalsPath,
          accompanimentPath: transcribeResult.accompanimentPath,
        };

        const outputPath = await runGenerate(url, generateInput);
        console.log(`\n✅ Pipeline complete! Output: ${outputPath}\n`);
      }
    }

    if (args.stage === "transcribe") {
      // ── TRANSCRIBE ──────────────────────────────────────────────────────
      let uploadResult: UploadResult;

      if (args.fromJson) {
        uploadResult = loadJson<UploadResult>(args.fromJson);
      } else if (args.mp3Path) {
        if (!existsSync(args.mp3Path)) {
          console.error(`Error: MP3 file not found: ${args.mp3Path}`);
          process.exit(1);
        }
        uploadResult = await runUpload(url, args.mp3Path);
      } else {
        console.error("Error: --from-json or --mp3 is required for transcribe stage");
        process.exit(1);
      }

      if (!args.lyricsPath) {
        console.error("Error: --lyrics is required for transcribe stage");
        process.exit(1);
      }
      if (!existsSync(args.lyricsPath)) {
        console.error(`Error: Lyrics file not found: ${args.lyricsPath}`);
        process.exit(1);
      }

      const transcribeInput: TranscribeInput = {
        mp3: uploadResult.mp3,
        lyrics: readFileSync(args.lyricsPath, "utf-8"),
      };

      const transcribeResult = await runTranscribe(url, transcribeInput);
      const sourcePath = args.fromJson ?? args.lyricsPath;
      saveJson(transcribeResult, defaultToJsonPath("transcribe", sourcePath), args.toJson);

      console.log(`\n✅ Transcribe complete.\n`);
    }

    if (args.stage === "align") {
      // ── ALIGN ───────────────────────────────────────────────────────────
      if (!args.fromJson) {
        console.error("Error: --from-json is required for align stage (provide transcribe output JSON)");
        process.exit(1);
      }

      const transcribeResult = loadJson<TranscribeResult>(args.fromJson);

      if (!args.lyricsPath) {
        console.error("Error: --lyrics is required for align stage");
        process.exit(1);
      }
      if (!existsSync(args.lyricsPath)) {
        console.error(`Error: Lyrics file not found: ${args.lyricsPath}`);
        process.exit(1);
      }

      // Derive mp3 path
      let mp3Path = "";
      if (args.mp3Path) {
        mp3Path = args.mp3Path;
      } else if (transcribeResult.vocalsPath) {
        mp3Path = transcribeResult.vocalsPath.replace(/_vocals\.mp3$/, ".mp3");
      }
      if (!mp3Path) {
        console.error("Error: Cannot determine MP3 path. Use --mp3 or provide transcribe JSON with vocalsPath.");
        process.exit(1);
      }

      const alignInput: AlignInput = {
        mp3: mp3Path,
        words: transcribeResult.words,
        language: transcribeResult.language,
        lyrics: readFileSync(args.lyricsPath, "utf-8"),
        pauses: transcribeResult.pauses,
      };

      const alignResult = await runAlign(url, alignInput);
      const sourcePath = args.fromJson ?? args.lyricsPath;
      saveJson(alignResult, defaultToJsonPath("align", sourcePath), args.toJson);

      console.log(`\n✅ Align complete.\n`);
    }

    if (args.stage === "generate") {
      // ── GENERATE ────────────────────────────────────────────────────────
      if (!args.fromJson) {
        console.error("Error: --from-json is required for generate stage (provide transcribe output JSON)");
        process.exit(1);
      }

      const transcribeResult = loadJson<TranscribeResult>(args.fromJson);

      if (!args.lyricsPath) {
        console.error("Error: --lyrics is required for generate stage");
        process.exit(1);
      }
      if (!existsSync(args.lyricsPath)) {
        console.error(`Error: Lyrics file not found: ${args.lyricsPath}`);
        process.exit(1);
      }
      if (!args.title || !args.artist) {
        console.error("Error: --title and --artist are required for generate stage");
        process.exit(1);
      }

      // Derive mp3 path: vocalsPath is always <mp3_base>_vocals.mp3
      let mp3Path = "";
      if (args.mp3Path) {
        mp3Path = args.mp3Path;
      } else if (transcribeResult.vocalsPath) {
        mp3Path = transcribeResult.vocalsPath.replace(/_vocals\.mp3$/, ".mp3");
      }
      if (!mp3Path) {
        console.error("Error: Cannot determine MP3 path. Use --mp3 or provide transcribe JSON with vocalsPath.");
        process.exit(1);
      }

      const generateInput: GenerateInput = {
        mp3: mp3Path,
        words: transcribeResult.words,
        language: transcribeResult.language,
        lyrics: readFileSync(args.lyricsPath, "utf-8"),
        title: args.title,
        artist: args.artist,
        pauses: transcribeResult.pauses,
        vocalsPath: transcribeResult.vocalsPath,
        accompanimentPath: transcribeResult.accompanimentPath,
      };

      const outputPath = await runGenerate(url, generateInput);
      console.log(`\n✅ Generate complete! Output: ${outputPath}\n`);
    }
  } catch (err) {
    console.error(`\n❌ Pipeline failed:`, err);
    process.exit(1);
  }
}

main();
