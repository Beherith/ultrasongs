"use client";

interface Props {
  title: string;
  artist: string;
  lyrics: string;
  onTitle: (v: string) => void;
  onArtist: (v: string) => void;
  onLyrics: (v: string) => void;
  disabled?: boolean;
}

export default function LyricsInput({ title, artist, lyrics, onTitle, onArtist, onLyrics, disabled }: Props) {
  const fieldCls =
    "w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-zinc-50 disabled:text-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:placeholder-zinc-500 dark:disabled:bg-zinc-800";

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => onTitle(e.target.value)}
            placeholder="Song title"
            disabled={disabled}
            className={fieldCls}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Artist</label>
          <input
            type="text"
            value={artist}
            onChange={(e) => onArtist(e.target.value)}
            placeholder="Artist name"
            disabled={disabled}
            className={fieldCls}
          />
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
          Lyrics <span className="text-zinc-400">(one line per verse line)</span>
        </label>
        <textarea
          value={lyrics}
          onChange={(e) => onLyrics(e.target.value)}
          placeholder={"Hello darkness my old friend\nI've come to talk with you again"}
          disabled={disabled}
          rows={10}
          className={`${fieldCls} resize-y font-mono leading-relaxed`}
        />
      </div>
    </div>
  );
}
