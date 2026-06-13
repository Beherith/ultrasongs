declare module "music-tempo" {
  class MusicTempo {
    constructor(audioData: Float32Array | number[]);
    tempo: string | number;
    beats: number[];
  }
  export = MusicTempo;
}
