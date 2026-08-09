# Migration fixtures

Fast unit tests build their own small in-memory fixtures. Large media and model
artifacts are intentionally not copied into this directory.

The optional parity test currently recognizes these workspace-root files:

- `notes.txt` - known-good reference Ultrastar song
- `Diggy Diggy Hole_transcribe.json` - frozen legacy transcription and pitch data
- `diggy_lyrics.txt` - lyrics supplied to the alignment stage

Run the downstream parity check with:

```powershell
python -m pytest tests/parity -m parity -q
```

The test skips when any local fixture is missing. It does not run Demucs,
torchcrepe, or Whisper; it validates the deterministic path from frozen
transcription through alignment, generation, similarity scoring, and report
creation.
