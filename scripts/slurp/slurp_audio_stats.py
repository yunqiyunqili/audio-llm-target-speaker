"""Header-level stats of extracted SLURP close-talk (-headset) recordings per official split.
Reads only FLAC headers (soundfile.info). Writes per-recording CSV (file, split, sr, ch, frames,
seconds, bytes) and a summary JSON. A file referenced by two records is listed once per record.
Usage (CPU node): python -I scripts/slurp/slurp_audio_stats.py ANNOT_DIR AUDIO_DIR OUT_DIR
"""
import collections
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf


def describe(x):
    x = np.asarray(x, float)
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()), total_hours=float(x.sum() / 3600),
                **{k: float(v) for k, v in zip(['min', 'p1', 'p5', 'p25', 'median', 'p75', 'p95', 'p99', 'max'], q)})


def main(annot, audio, out):
    os.makedirs(out, exist_ok=True)
    rows, summary, cache = [], {}, {}
    for split in ('train', 'devel', 'test'):
        recs = []
        for line in open(f'{annot}/{split}.jsonl'):
            r = json.loads(line)
            for x in r['recordings']:
                if x['file'].endswith('-headset.flac'):
                    recs.append((r['slurp_id'], x['file']))
        srs, chs, secs, subtypes, missing = collections.Counter(), collections.Counter(), [], collections.Counter(), 0
        for sid, f in recs:
            p = os.path.join(audio, f)
            if f not in cache:
                if not os.path.isfile(p):
                    cache[f] = None
                else:
                    i = sf.info(p)
                    cache[f] = dict(sr=i.samplerate, ch=i.channels, frames=i.frames, subtype=i.subtype,
                                    bytes=os.path.getsize(p))
            i = cache[f]
            if i is None:
                missing += 1; continue
            srs[i['sr']] += 1; chs[i['ch']] += 1; subtypes[i['subtype']] += 1
            secs.append(i['frames'] / i['sr'])
            rows.append(dict(split=split, slurp_id=sid, file=f, seconds=i['frames'] / i['sr'], **i))
        hist = np.histogram(secs, bins=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 1e9])[0].tolist()
        summary[split] = dict(headset_entries=len(recs), missing_files=missing, sample_rates=dict(srs),
                              channels=dict(chs), subtypes=dict(subtypes), seconds=describe(secs),
                              seconds_hist=dict(zip(['<1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-8', '8-10',
                                                     '10-15', '15-20', '>=20'], hist)))
        print(split, json.dumps(summary[split]), flush=True)
    regular = sum(os.path.getsize(os.path.join(audio, f)) for f in os.listdir(audio))
    summary['extracted_regular_file_bytes'] = regular
    summary['extracted_files'] = len(os.listdir(audio))
    with open(f'{out}/headset_durations.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(f'{out}/audio_stats.json', 'w'), indent=2)
    print('BYTES', regular, 'FILES', summary['extracted_files'])


if __name__ == '__main__':
    main(*sys.argv[1:4])
