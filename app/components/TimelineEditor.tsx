"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { EditorNote } from "@/app/lib/editorNote";
import { detectPitch, hzToMidi } from "@/app/lib/pitchDetection";

// ─── Layout constants ────────────────────────────────────────────────────────
const KEYS_W = 68;
const PX_PITCH = 14;
const MIN_PITCH = 36;    // C2
const MAX_PITCH = 84;    // C6
const ROLL_H = (MAX_PITCH - MIN_PITCH + 1) * PX_PITCH;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);

function noteName(p: number) { return `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`; }
function isBlack(p: number) { return BLACK.has(p % 12); }
function pitchToY(p: number) { return (MAX_PITCH - p) * PX_PITCH; }

interface Props {
  initialNotes: EditorNote[];
  bpm: number;
  gap: number;
  audioUrl: string;
  videoUrl?: string;
  title: string;
  artist: string;
  onExport: (notes: EditorNote[], bpm: number, gap: number) => void;
  onClose?: () => void;
}

export default function TimelineEditor({
  initialNotes, bpm, gap, audioUrl, videoUrl, title, artist, onExport, onClose,
}: Props) {
  const [notes, setNotes] = useState<EditorNote[]>(initialNotes);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pxPerSec, setPxPerSec] = useState(120);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadSec, setPlayheadSec] = useState(0);
  const [micEnabled, setMicEnabled] = useState(false);
  const [micMidi, setMicMidi] = useState<number | null>(null);
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);
  const [micDeviceId, setMicDeviceId] = useState<string>("");
  const [micLabel, setMicLabel] = useState<string>("");
  const [micTrace, setMicTrace] = useState<Array<{ timeSec: number; midi: number }>>([]);

  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef(0);
  const micRafRef = useRef(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const previewCtxRef = useRef<AudioContext | null>(null);
  const syllableInputRef = useRef<HTMLInputElement>(null);
  const pxPerSecRef = useRef(pxPerSec);
  useEffect(() => { pxPerSecRef.current = pxPerSec; }, [pxPerSec]);

  // Ref so the mic tick can read isPlaying without stale closure
  const isPlayingRef = useRef(false);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);

  // Pending mic points flushed to state every ~6 frames to avoid per-frame re-renders
  const micPendingRef = useRef<Array<{ timeSec: number; midi: number }>>([]);
  const micFrameRef = useRef(0);

  const duration = notes.reduce((m, n) => Math.max(m, n.startSec + n.durationSec), 10);
  const totalW = KEYS_W + (duration + 4) * pxPerSec;
  const selectedNote = notes.find((n) => n.id === selectedId) ?? null;

  // ── Playback ──────────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (isPlaying) {
      a.pause();
      videoRef.current?.pause();
      cancelAnimationFrame(rafRef.current);
      setIsPlaying(false);
    } else {
      if (videoRef.current) videoRef.current.currentTime = a.currentTime;
      a.play();
      videoRef.current?.play().catch(() => {});
      // Clear previous mic trace so each play session starts fresh
      setMicTrace([]);
      micPendingRef.current = [];
      setIsPlaying(true);
      const tick = () => {
        setPlayheadSec(audioRef.current?.currentTime ?? 0);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [isPlaying]);

  const seekTo = useCallback((sec: number) => {
    const clamped = Math.max(0, sec);
    setPlayheadSec(clamped);
    if (audioRef.current) audioRef.current.currentTime = clamped;
    if (videoRef.current) videoRef.current.currentTime = clamped;
  }, []);

  const playNotePreview = useCallback((midi: number, durationSec: number) => {
    if (!previewCtxRef.current || previewCtxRef.current.state === "closed") {
      previewCtxRef.current = new AudioContext();
    }
    const ctx = previewCtxRef.current;
    const hz = 440 * Math.pow(2, (midi - 69) / 12);
    const dur = Math.min(Math.max(durationSec, 0.15), 1.5);
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.value = hz;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.35, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
    osc.start(now);
    osc.stop(now + dur);
  }, []);

  // Auto-scroll playhead into view
  useEffect(() => {
    if (!isPlaying || !scrollRef.current) return;
    const x = playheadSec * pxPerSec;
    const container = scrollRef.current;
    const visible = container.scrollLeft + container.clientWidth - KEYS_W;
    if (x > visible - 40 || x < container.scrollLeft) {
      container.scrollLeft = Math.max(0, x - 80);
    }
  }, [isPlaying, playheadSec, pxPerSec]);

  // ── Microphone ────────────────────────────────────────────────────────────
  const startMic = useCallback(async (deviceId?: string) => {
    cancelAnimationFrame(micRafRef.current);
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    micStreamRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current = null;

    const constraints: MediaStreamConstraints = {
      audio: deviceId ? { deviceId: { exact: deviceId } } : true,
      video: false,
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    micStreamRef.current = stream;

    const track = stream.getAudioTracks()[0];
    setMicLabel(track.label || "Micrófono");

    // Enumerate after getUserMedia so labels are populated (browser policy)
    const all = await navigator.mediaDevices.enumerateDevices();
    setMicDevices(all.filter((d) => d.kind === "audioinput"));

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyserRef.current = analyser;
    ctx.createMediaStreamSource(stream).connect(analyser);

    const buf = new Float32Array(analyser.fftSize);
    micFrameRef.current = 0;
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      const hz = detectPitch(buf, ctx.sampleRate);
      const midi = hz > 0 ? hzToMidi(hz) : null;
      const valid = midi !== null && midi >= MIN_PITCH && midi <= MAX_PITCH ? midi : null;
      setMicMidi(valid);

      // Record pitch when playing (~10 points/sec, every 6 frames at 60fps)
      if (isPlayingRef.current && valid !== null) {
        micPendingRef.current.push({ timeSec: audioRef.current?.currentTime ?? 0, midi: valid });
      }
      micFrameRef.current++;
      if (micFrameRef.current % 6 === 0 && micPendingRef.current.length > 0) {
        const batch = micPendingRef.current.splice(0);
        setMicTrace((prev) => [...prev, ...batch]);
      }

      micRafRef.current = requestAnimationFrame(tick);
    };
    micRafRef.current = requestAnimationFrame(tick);
  }, []);

  const toggleMic = useCallback(async () => {
    if (micEnabled) {
      cancelAnimationFrame(micRafRef.current);
      micStreamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close();
      micStreamRef.current = null;
      analyserRef.current = null;
      audioCtxRef.current = null;
      setMicMidi(null);
      setMicLabel("");
      setMicEnabled(false);
      return;
    }
    try {
      await startMic(micDeviceId || undefined);
      setMicEnabled(true);
    } catch {
      setMicEnabled(false);
    }
  }, [micEnabled, micDeviceId, startMic]);

  // ── Apply mic trace to note pitches ──────────────────────────────────────
  const applyMicTrace = useCallback(() => {
    if (micTrace.length === 0) return;
    setNotes((prev) =>
      prev.map((note) => {
        const pts = micTrace.filter(
          (p) => p.timeSec >= note.startSec && p.timeSec <= note.startSec + note.durationSec
        );
        if (pts.length === 0) return note;
        const sorted = pts.map((p) => p.midi).sort((a, b) => a - b);
        return { ...note, pitch: sorted[Math.floor(sorted.length / 2)] };
      })
    );
  }, [micTrace]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      cancelAnimationFrame(micRafRef.current);
      micStreamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close();
      previewCtxRef.current?.close();
    };
  }, []);

  // ── Drag note (move / resize) ─────────────────────────────────────────────
  const startDrag = useCallback(
    (e: React.MouseEvent, note: EditorNote, kind: "move" | "resize") => {
      e.preventDefault();
      e.stopPropagation();
      setSelectedId(note.id);
      const startX = e.clientX;
      const startY = e.clientY;
      const { startSec, durationSec, pitch } = note;
      const onMove = (ev: MouseEvent) => {
        const dx = ev.clientX - startX;
        const dy = ev.clientY - startY;
        const pps = pxPerSecRef.current;
        setNotes((prev) =>
          prev.map((n) => {
            if (n.id !== note.id) return n;
            if (kind === "resize") return { ...n, durationSec: Math.max(0.05, durationSec + dx / pps) };
            return {
              ...n,
              startSec: Math.max(0, startSec + dx / pps),
              pitch: Math.max(MIN_PITCH, Math.min(MAX_PITCH, pitch - Math.round(dy / PX_PITCH))),
            };
          })
        );
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    []
  );

  // ── Drag playhead ─────────────────────────────────────────────────────────
  const startPlayheadDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (isPlaying) {
        audioRef.current?.pause();
        videoRef.current?.pause();
        cancelAnimationFrame(rafRef.current);
        setIsPlaying(false);
      }
      const container = scrollRef.current;
      const onMove = (ev: MouseEvent) => {
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const x = ev.clientX - rect.left + container.scrollLeft - KEYS_W;
        seekTo(Math.max(0, x / pxPerSecRef.current));
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [isPlaying, seekTo]
  );

  // ── Seek on grid single-click ─────────────────────────────────────────────
  const handleGridClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.detail > 1) return; // skip double-click
      const rect = e.currentTarget.getBoundingClientRect();
      const sec = (e.clientX - rect.left) / pxPerSec;
      seekTo(sec);
    },
    [pxPerSec, seekTo]
  );

  // ── Add note on grid double-click ─────────────────────────────────────────
  const handleGridDblClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const sec = Math.max(0, (e.clientX - rect.left) / pxPerSecRef.current);
      const yRel = e.clientY - rect.top;
      const pitch = Math.max(MIN_PITCH, Math.min(MAX_PITCH, Math.round(MAX_PITCH - yRel / PX_PITCH)));
      const id = crypto.randomUUID();
      const newNote: EditorNote = { id, startSec: sec, durationSec: 0.3, pitch, syllable: "+", type: ":" };
      setNotes((prev) => [...prev, newNote].sort((a, b) => a.startSec - b.startSec));
      setSelectedId(id);
      // Focus syllable input so user can type the syllable immediately
      setTimeout(() => { syllableInputRef.current?.focus(); syllableInputRef.current?.select(); }, 0);
    },
    []
  );

  // ── Split selected note ───────────────────────────────────────────────────
  const splitSelected = useCallback(() => {
    if (!selectedId) return;
    setNotes((prev) => {
      const idx = prev.findIndex((n) => n.id === selectedId);
      if (idx === -1) return prev;
      const n = prev[idx];
      const half = n.durationSec / 2;
      const cut = Math.ceil(n.syllable.length / 2);
      const a: EditorNote = { ...n, durationSec: half, syllable: n.syllable.slice(0, cut) };
      const b: EditorNote = {
        ...n, id: crypto.randomUUID(), startSec: n.startSec + half,
        durationSec: half, syllable: n.syllable.slice(cut),
      };
      const result = [...prev];
      result.splice(idx, 1, a, b);
      return result;
    });
  }, [selectedId]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return; // don't steal input events
      if (e.code === "Space") { e.preventDefault(); togglePlay(); }
      if ((e.code === "Delete" || e.code === "Backspace") && selectedId) {
        setNotes((prev) => prev.filter((n) => n.id !== selectedId));
        setSelectedId(null);
      }
      if (e.code === "KeyS" && !e.ctrlKey && !e.metaKey && selectedId) {
        e.preventDefault();
        splitSelected();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlay, selectedId, splitSelected]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const playheadX = KEYS_W + playheadSec * pxPerSec;
  const micY = micMidi !== null ? pitchToY(micMidi) + PX_PITCH / 2 : null;

  return (
    <div className="flex flex-col rounded-xl border border-zinc-200 bg-white overflow-hidden dark:border-zinc-700 dark:bg-zinc-900">
      {/* ── Header ── */}
      <div className="flex items-center justify-between gap-2 border-b border-zinc-200 px-3 py-2 dark:border-zinc-700">
        <span className="truncate text-sm font-semibold text-zinc-700 dark:text-zinc-200">
          {title} — {artist}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          <label className="flex items-center gap-1 text-xs text-zinc-400">
            <span>Zoom</span>
            <input
              type="range" min={40} max={400} step={10} value={pxPerSec}
              onChange={(e) => setPxPerSec(Number(e.target.value))}
              className="w-20 accent-blue-500"
            />
          </label>
          <div className="flex items-center gap-1">
            <button
              onClick={toggleMic}
              title={micLabel || undefined}
              className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
                micEnabled
                  ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-400"
                  : "bg-zinc-100 text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
              }`}
            >
              🎤 {micEnabled ? (micLabel ? micLabel.split("(")[0].trim() : "Mic on") : "Mic"}
            </button>
            {micDevices.length > 1 && (
              <select
                value={micDeviceId}
                onChange={async (e) => {
                  const id = e.target.value;
                  setMicDeviceId(id);
                  if (micEnabled) {
                    try { await startMic(id); } catch { /* ignore */ }
                  }
                }}
                className="rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
              >
                {micDevices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Micrófono ${d.deviceId.slice(0, 6)}`}
                  </option>
                ))}
              </select>
            )}
          </div>
          <button
            onClick={() => onExport(notes, bpm, gap)}
            className="rounded bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700"
          >
            Export
          </button>
          {onClose && (
            <button onClick={onClose} className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200">✕</button>
          )}
        </div>
      </div>

      {/* ── Transport ── */}
      <div className="flex flex-wrap items-center gap-3 border-b border-zinc-100 px-3 py-1.5 text-xs dark:border-zinc-800">
        {/* Play / time */}
        <button
          onClick={togglePlay}
          className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-white hover:bg-blue-700"
          title="Space"
        >
          {isPlaying ? (
            <svg viewBox="0 0 10 10" className="h-3 w-3" fill="currentColor">
              <rect x="1" y="1" width="3" height="8" /><rect x="6" y="1" width="3" height="8" />
            </svg>
          ) : (
            <svg viewBox="0 0 10 10" className="h-3 w-3" fill="currentColor">
              <polygon points="2,1 9,5 2,9" />
            </svg>
          )}
        </button>
        <span className="font-mono text-zinc-500">{playheadSec.toFixed(2)} s</span>

        {/* Selected note controls */}
        {selectedNote && (
          <>
            <div className="h-4 w-px bg-zinc-200 dark:bg-zinc-700" />
            <input
              ref={syllableInputRef}
              value={selectedNote.syllable}
              onChange={(e) =>
                setNotes((prev) =>
                  prev.map((n) => n.id === selectedId ? { ...n, syllable: e.target.value } : n)
                )
              }
              placeholder="syllable"
              className="w-24 rounded border border-zinc-300 bg-white px-1.5 py-0.5 text-xs text-zinc-800 focus:outline-none focus:ring-1 focus:ring-blue-400 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
            />
            <button
              onClick={splitSelected}
              title="Split note in half (S)"
              className="rounded border border-zinc-300 px-2 py-0.5 text-xs text-zinc-600 hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-400 dark:hover:bg-zinc-800"
            >
              Split (S)
            </button>
            <button
              onClick={() => { setNotes((prev) => prev.filter((n) => n.id !== selectedId)); setSelectedId(null); }}
              title="Delete note (Del)"
              className="rounded border border-red-300 px-2 py-0.5 text-xs text-red-600 hover:bg-red-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950"
            >
              Delete
            </button>
            <select
              value={selectedNote.type}
              onChange={(e) =>
                setNotes((prev) =>
                  prev.map((n) => n.id === selectedId ? { ...n, type: e.target.value as ":" | "*" } : n)
                )
              }
              className="rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-200"
            >
              <option value=":">Normal</option>
              <option value="*">Golden</option>
            </select>
          </>
        )}

        {micEnabled && (
          <span className="text-green-600 dark:text-green-400">
            🎤 {micMidi !== null ? noteName(micMidi) : "—"}
            {isPlaying && <span className="ml-1 animate-pulse text-green-500">●</span>}
          </span>
        )}
        {micTrace.length > 0 && (
          <>
            <span className="text-xs text-zinc-400">{micTrace.length} pts</span>
            <button
              onClick={applyMicTrace}
              className="rounded bg-green-600 px-2 py-0.5 text-xs font-semibold text-white hover:bg-green-700"
            >
              Aplicar mic
            </button>
            <button
              onClick={() => { setMicTrace([]); micPendingRef.current = []; }}
              className="rounded border border-zinc-300 px-2 py-0.5 text-xs text-zinc-500 hover:bg-zinc-100 dark:border-zinc-600 dark:hover:bg-zinc-800"
            >
              Limpiar
            </button>
          </>
        )}
        <span className="ml-auto text-zinc-400">
          Drag note · right edge=resize · dbl-click empty=add · S=split · Del=remove
        </span>
      </div>

      {/* ── Video preview ── */}
      {videoUrl && (
        <div className="border-b border-zinc-200 bg-black dark:border-zinc-700">
          <video ref={videoRef} src={videoUrl} muted className="mx-auto block max-h-48 w-full object-contain" />
        </div>
      )}

      {/* ── Piano roll ── */}
      <div ref={scrollRef} className="overflow-auto" style={{ maxHeight: 480 }}>
        <div style={{ width: totalW, height: ROLL_H, position: "relative" }}>

          {/* Piano keys */}
          <div style={{ position: "sticky", left: 0, top: 0, width: KEYS_W, height: ROLL_H, zIndex: 20, flexShrink: 0 }}>
            {Array.from({ length: MAX_PITCH - MIN_PITCH + 1 }, (_, i) => {
              const p = MAX_PITCH - i;
              const black = isBlack(p);
              const highlighted = micMidi === p;
              return (
                <div
                  key={p}
                  style={{ position: "absolute", top: i * PX_PITCH, left: 0, width: KEYS_W - 2, height: PX_PITCH - 1 }}
                  className={`flex items-center border-b text-[9px] select-none transition-colors ${
                    highlighted
                      ? "bg-green-400 text-white border-green-500"
                      : black
                      ? "bg-zinc-700 text-zinc-400 border-zinc-600"
                      : "bg-white text-zinc-400 border-zinc-200 dark:bg-zinc-800 dark:border-zinc-700"
                  }`}
                >
                  {p % 12 === 0 && <span className="ml-1">{noteName(p)}</span>}
                </div>
              );
            })}
          </div>

          {/* Grid rows + beat lines */}
          <div
            style={{ position: "absolute", left: KEYS_W, top: 0, right: 0, height: ROLL_H, zIndex: 1 }}
            onClick={handleGridClick}
            onDoubleClick={handleGridDblClick}
          >
            {Array.from({ length: MAX_PITCH - MIN_PITCH + 1 }, (_, i) => {
              const p = MAX_PITCH - i;
              return (
                <div
                  key={p}
                  style={{ position: "absolute", top: i * PX_PITCH, left: 0, right: 0, height: PX_PITCH }}
                  className={isBlack(p) ? "bg-zinc-100 dark:bg-zinc-800/60" : "bg-white dark:bg-zinc-900"}
                />
              );
            })}
            {Array.from({ length: Math.ceil(duration + 4) }, (_, beat) => (
              <div
                key={beat}
                style={{
                  position: "absolute",
                  left: (beat * 60) / bpm * pxPerSec,
                  top: 0, width: 1, height: ROLL_H,
                  background: beat % 4 === 0 ? "rgba(0,0,0,0.12)" : "rgba(0,0,0,0.05)",
                  zIndex: 2, pointerEvents: "none",
                }}
              />
            ))}
          </div>

          {/* Mic trace dots */}
          {micTrace.map((pt, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: KEYS_W + pt.timeSec * pxPerSec - 2,
                top: pitchToY(pt.midi) + PX_PITCH / 2 - 2,
                width: 4, height: 4,
                borderRadius: "50%",
                background: "rgba(34,197,94,0.65)",
                zIndex: 15,
                pointerEvents: "none",
              }}
            />
          ))}

          {/* Notes */}
          {notes.map((note) => {
            const x = KEYS_W + note.startSec * pxPerSec;
            const y = pitchToY(note.pitch);
            const w = note.durationSec * pxPerSec;
            const sel = note.id === selectedId;
            return (
              <div
                key={note.id}
                style={{
                  position: "absolute", left: x + 1, top: y + 2,
                  width: Math.max(w - 2, 8), height: PX_PITCH - 4,
                  zIndex: sel ? 30 : 10, cursor: "grab", userSelect: "none",
                }}
                className={`flex items-center rounded text-[10px] font-bold text-white px-1 overflow-hidden whitespace-nowrap transition-shadow ${
                  sel ? "ring-2 ring-white shadow-lg" : ""
                } ${note.type === "*" ? "bg-amber-500 hover:bg-amber-400" : "bg-blue-500 hover:bg-blue-400"}`}
                onMouseDown={(e) => startDrag(e, note, "move")}
                onClick={(e) => { e.stopPropagation(); playNotePreview(note.pitch, note.durationSec); }}
                onDoubleClick={(e) => e.stopPropagation()}
              >
                {note.syllable}
                {/* Resize handle */}
                <div
                  style={{ position: "absolute", right: 0, top: 0, width: 6, height: "100%", cursor: "ew-resize" }}
                  onMouseDown={(e) => startDrag(e, note, "resize")}
                />
              </div>
            );
          })}

          {/* Playhead — draggable */}
          <div
            style={{
              position: "absolute", left: playheadX - 4, top: 0,
              width: 10, height: ROLL_H,
              zIndex: 40, cursor: "col-resize",
            }}
            onMouseDown={startPlayheadDrag}
          >
            {/* Visual line centred in the hit area */}
            <div style={{ position: "absolute", left: 4, top: 0, width: 2, height: "100%", background: "#ef4444" }} />
          </div>

          {/* Mic guide line */}
          {micEnabled && micY !== null && (
            <div style={{
              position: "absolute", left: KEYS_W, top: micY - 1, right: 0,
              height: 2, background: "rgba(34,197,94,0.75)", zIndex: 35, pointerEvents: "none",
            }} />
          )}
        </div>
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        onEnded={() => {
          videoRef.current?.pause();
          setIsPlaying(false);
          cancelAnimationFrame(rafRef.current);
        }}
      />
    </div>
  );
}
