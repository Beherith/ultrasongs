import json

with open('tmp/whisperx_pitch.json', encoding='utf-8') as f:
    wp = json.load(f)

# Check: do pitch frames extend beyond whisper word ends?
extensions = []
for w in wp['words'][:30]:
    word_end = w['end']
    max_pitch_time = max((f['time'] for f in w['pitchFrames'] if f['midi'] > 0), default=0)
    ext = max_pitch_time - word_end
    extensions.append(ext)
    if ext > 0.01 or w['word'] in ('Brothers', 'of', 'mine'):
        last_frames = [f for f in w['pitchFrames'] if f['time'] >= word_end - 0.05]
        print(f"'{w['word']}' end={word_end:.3f}s  max_pitch={max_pitch_time:.3f}s  ext={ext:+.3f}s")
        for f in last_frames:
            print(f"  frame t={f['time']:.3f} midi={f['midi']} conf={f['confidence']:.2f}")

print()
print(f"Words with pitch extending beyond whisper end: {sum(1 for e in extensions if e > 0.01)} / {len(extensions)}")

# Check: what's the gap between consecutive words' pitch?
print()
for i in range(min(10, len(wp['words']) - 1)):
    w1 = wp['words'][i]
    w2 = wp['words'][i+1]
    w1_max_pitch = max((f['time'] for f in w1['pitchFrames'] if f['midi'] > 0), default=w1['end'])
    w2_min_pitch = min((f['time'] for f in w2['pitchFrames'] if f['midi'] > 0), default=w2['start'])
    gap = w2_min_pitch - w1_max_pitch
    print(f"'{w1['word']}'->'{w2['word']}': pitch gap = {gap*1000:+.1f}ms (w1 max={w1_max_pitch:.3f}, w2 min={w2_min_pitch:.3f})")
