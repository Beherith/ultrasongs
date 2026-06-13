export interface EditorNote {
  id: string;
  syllable: string;
  startSec: number;
  durationSec: number;
  pitch: number; // MIDI 0-127
  type: ":" | "*"; // normal | golden
}
