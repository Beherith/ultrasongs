"use client";

type Phase = "idle" | "uploading" | "transcribing" | "generating" | "done" | "error";

interface Props {
  phase: Phase;
  onClick: () => void;
  disabled?: boolean;
}

const LABELS: Record<Phase, string> = {
  idle: "Generate Ultrastar file",
  uploading: "Uploading…",
  transcribing: "Transcribing audio…",
  generating: "Building song…",
  done: "Generate again",
  error: "Try again",
};

export default function GenerateButton({ phase, onClick, disabled }: Props) {
  const busy = phase === "uploading" || phase === "transcribing" || phase === "generating";

  return (
    <button
      onClick={onClick}
      disabled={disabled || busy}
      className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-500 dark:disabled:bg-zinc-700 dark:disabled:text-zinc-400"
    >
      {busy && (
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
      )}
      {LABELS[phase]}
    </button>
  );
}
