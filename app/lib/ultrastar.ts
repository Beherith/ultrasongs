import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import ffmpeg from "fluent-ffmpeg";
import MusicTempo from "music-tempo";
import { statSync } from "fs";

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

export interface UltrastarNote {
  type: ":" | "*" | "-";
  startBeat: number;
  duration: number;
  pitch: number;
  syllable: string;
}

export interface UltrastarMeta {
  title: string;
  artist: string;
  mp3: string;
  bpm: number;
  gap: number;
  video?: string;
}

/**
 * Converts milliseconds to Ultrastar beats.
 * beat = ((ms - GAP) / 1000) * (BPM / 60) * 4
 */
export function msToBeats(ms: number, bpm: number, gap: number): number {
  return Math.round(((ms - gap) / 1000) * (bpm / 60) * 4);
}

export function buildUltrastarTxt(notes: UltrastarNote[], meta: UltrastarMeta): string {
  const headerLines = [
    `#TITLE:${meta.title}`,
    `#ARTIST:${meta.artist}`,
    `#MP3:${meta.mp3}`,
    ...(meta.video ? [`#VIDEO:${meta.video}`] : []),
    `#BPM:${meta.bpm.toFixed(2)}`,
    `#GAP:${Math.round(meta.gap)}`,
    "",
  ];
  const header = headerLines.join("\n");

  const body = notes
    .map((n) =>
      n.type === "-"
        ? `- ${n.startBeat}`
        : `${n.type} ${n.startBeat} ${n.duration} ${n.pitch} ${n.syllable}`
    )
    .join("\n");

  return header + body + "\nE\n";
}

/** Extracts float32 mono PCM from an mp3 via FFmpeg, returns samples at the given sampleRate. */
function extractPcm(mp3Path: string, sampleRate = 44100): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    const cmd = ffmpeg(mp3Path)
      .noVideo()
      .audioChannels(1)
      .audioFrequency(sampleRate)
      .audioCodec("pcm_f32le")
      .format("f32le")
      .on("error", (err) => {
        const stderr = Buffer.concat(stderrChunks).toString().trim();
        const msg = stderr ? `${err.message}\nFFmpeg stderr:\n${stderr}` : err.message;
        reject(Object.assign(new Error(msg), { cause: err }));
      })
      .on("stderr", (line: string) => stderrChunks.push(Buffer.from(line + "\n")));

    cmd.pipe()
      .on("data", (chunk: Buffer) => chunks.push(chunk))
      .on("end", () => {
        const buf = Buffer.concat(chunks);
        const floats = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
        resolve(floats);
      })
      .on("error", (err) => {
        const stderr = Buffer.concat(stderrChunks).toString().trim();
        const msg = stderr ? `${err.message}\nFFmpeg stderr:\n${stderr}` : err.message;
        reject(Object.assign(new Error(msg), { cause: err }));
      });
  });
}

/**
 * Detects BPM from an mp3 file.
 * Returns { bpm, beats } where beats are timestamps in seconds.
 */
export async function detectBpm(mp3Path: string): Promise<{ bpm: number; beats: number[] }> {
  try {
    const stats = statSync(mp3Path);
    console.log(`[detectBpm] File: ${mp3Path}, size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);

    console.log("[detectBpm] Extracting PCM via FFmpeg...");
    const samples = await extractPcm(mp3Path);
    console.log(`[detectBpm] PCM extracted: ${samples.length} samples (~${(samples.length / 44100).toFixed(1)}s)`);

    console.log("[detectBpm] Running MusicTempo analysis...");
    const mt = new MusicTempo(samples);
    console.log("[detectBpm] MusicTempo constructor succeeded, accessing tempo...");
    const bpm = parseFloat(String(mt.tempo));
    console.log(`[detectBpm] Detected BPM: ${bpm}, beats: ${mt.beats.length}`);
    return { bpm, beats: mt.beats };
  } catch (err) {
    console.error("[detectBpm] Tempo extraction failed, using fallback 120 BPM");
    if (err instanceof Error) {
      console.error(`[detectBpm]   name:    ${err.name}`);
      console.error(`[detectBpm]   message: ${err.message}`);
      console.error(`[detectBpm]   stack:   ${err.stack ?? "(no stack)"}`);
      console.error(`[detectBpm]   cause:   ${err.cause ?? "(none)"}`);
    } else {
      console.error(`[detectBpm]   type:    ${typeof err}`);
      console.error(`[detectBpm]   value:   ${JSON.stringify(err)}`);
    }
    return { bpm: 120, beats: [] };
  }
}

export interface ParsedUltrastar {
  title: string;
  artist: string;
  bpm: number;
  gap: number;
  mp3Filename: string;
  videoFilename?: string;
  notes: UltrastarNote[];
}

/** Parses an Ultrastar .txt file into structured data. */
export function parseUltrastarTxt(content: string): ParsedUltrastar {
  const lines = content.split(/\r?\n/);
  const meta: Record<string, string> = {};
  const notes: UltrastarNote[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#")) {
      const colon = trimmed.indexOf(":");
      if (colon > 0) {
        meta[trimmed.slice(1, colon).toUpperCase()] = trimmed.slice(colon + 1).trim();
      }
    } else if (trimmed.startsWith(": ") || trimmed.startsWith("* ")) {
      const parts = trimmed.split(" ");
      if (parts.length >= 4) {
        notes.push({
          type: parts[0] as ":" | "*",
          startBeat: parseInt(parts[1]),
          duration: parseInt(parts[2]),
          pitch: parseInt(parts[3]),
          syllable: parts.slice(4).join(" "),
        });
      }
    } else if (trimmed.startsWith("- ")) {
      notes.push({ type: "-", startBeat: parseInt(trimmed.slice(2)), duration: 0, pitch: 0, syllable: "" });
    }
  }

  return {
    title: meta.TITLE ?? "",
    artist: meta.ARTIST ?? "",
    bpm: parseFloat((meta.BPM ?? "120").replace(",", ".")),
    gap: parseInt(meta.GAP ?? "0"),
    mp3Filename: meta.MP3 ?? "",
    videoFilename: meta.VIDEO,
    notes,
  };
}
