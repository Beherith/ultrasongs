import json, math

with open('tmp/align_debug.json', encoding='utf-8') as f:
    debug = json.load(f)

bpm = 123.05
gap = 32120

def beats_to_ms(beat):
    return (beat / 4 / bpm * 60 * 1000) + gap

notes = []
with open('output/tit11.txt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            parts = line.split()
            start_beat = int(parts[1])
            duration = int(parts[2])
            syllable = parts[4]
            notes.append({
                'syllable': syllable,
                'start_beat': start_beat,
                'duration': duration,
                'start_ms': beats_to_ms(start_beat),
                'end_ms': beats_to_ms(start_beat + duration),
            })

syls = [s for s in debug['final_output'] if not s['is_line_break']]

print(f'Total notes in .txt: {len(notes)}')
print(f'Total syllables in debug: {len(syls)}')
print()

drifts = []
for i in range(min(len(notes), len(syls))):
    s = syls[i]
    n = notes[i]
    drift_s = n['start_ms'] - s['start'] * 1000
    drift_e = n['end_ms'] - s['end'] * 1000
    drifts.append((i, s['syllable'], s['start'], s['end'], n['start_ms']/1000, n['end_ms']/1000, drift_s, drift_e, s.get('pitch_end', 0)))

print('First 15:')
for d in drifts[:15]:
    pe = f" pitch_end={d[8]:.3f}" if d[8] > 0 else ""
    print(f"  [{d[0]:>3}] '{d[1]}' align={d[2]:.3f}-{d[3]:.3f}s  txt={d[4]:.3f}-{d[5]:.3f}s  drift={d[6]:+.1f}/{d[7]:+.1f}ms{pe}")

print()
print('Around verse 5 (indices 60-75):')
for d in drifts[60:76]:
    pe = f" pitch_end={d[8]:.3f}" if d[8] > 0 else ""
    print(f"  [{d[0]:>3}] '{d[1]}' align={d[2]:.3f}-{d[3]:.3f}s  txt={d[4]:.3f}-{d[5]:.3f}s  drift={d[6]:+.1f}/{d[7]:+.1f}ms{pe}")

print()
print('Around verse 15 (indices 180-195):')
for d in drifts[180:196]:
    pe = f" pitch_end={d[8]:.3f}" if d[8] > 0 else ""
    print(f"  [{d[0]:>3}] '{d[1]}' align={d[2]:.3f}-{d[3]:.3f}s  txt={d[4]:.3f}-{d[5]:.3f}s  drift={d[6]:+.1f}/{d[7]:+.1f}ms{pe}")

print()
print('Last 15:')
for d in drifts[-15:]:
    pe = f" pitch_end={d[8]:.3f}" if d[8] > 0 else ""
    print(f"  [{d[0]:>3}] '{d[1]}' align={d[2]:.3f}-{d[3]:.3f}s  txt={d[4]:.3f}-{d[5]:.3f}s  drift={d[6]:+.1f}/{d[7]:+.1f}ms{pe}")

print()
avg_start = sum(d[6] for d in drifts) / len(drifts)
avg_end = sum(d[7] for d in drifts) / len(drifts)
max_start = max(abs(d[6]) for d in drifts)
max_end = max(abs(d[7]) for d in drifts)
print(f'Average drift start: {avg_start:+.1f} ms')
print(f'Average drift end: {avg_end:+.1f} ms')
print(f'Max abs drift start: {max_start:.1f} ms')
print(f'Max abs drift end: {max_end:.1f} ms')

# Count pitch_end extensions
pe_count = sum(1 for d in drifts if d[8] > 0)
pe_avg_extend = sum(d[8] - d[3] for d in drifts if d[8] > 0) / max(pe_count, 1)
print(f'Syllables with pitch_end extension: {pe_count} ({100*pe_count/len(drifts):.1f}%)')
print(f'Average pitch_end extension: {pe_avg_extend*1000:.1f} ms')
