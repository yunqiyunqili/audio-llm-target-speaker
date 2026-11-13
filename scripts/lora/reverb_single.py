"""Single-talker full-reverb audio for LoRA step 1 (docs/lora_slu_plan_v2.md §3.1, §5 C1-R). CPU only.
Chain per recording (same steps as mix_spec_v1 + the C5 16 kHz chain; no interferer, no noise):
  1. s = resample_poly(slurp_16k, 1, 2) (8 kHz).
  2. RIR: SMS-WSJ example + source position (h_0 / h_1, as SMS-WSJ uses for its two speakers), 6 ch, 8 kHz.
     rir_start = min over the 6 channels of the first sample with |h| > 0.1 * max|h| (mix_slurp_v1.rir_start).
  3. y8 = extract_piece(fftconvolve(s, h[mic 0]), -rir_start, len(s)) (mix_slurp_v1 alignment; tail cut at len(s)).
  4. y16 = level(resample_poly(y8, 2, 1)[:2 * len(s)]) (qwen_omni_c4.level: active-frame RMS -30 dBFS; = C5 chain).
  5. Written as 16-bit PCM wav, mic 0 only.
RIR sets: devel -> cv_dev93 (C1-R evaluation); train -> train_si284 (training); test -> test_eval92 (SLURP test C1-R only,
  added 2026-11-12, docs/slurp_test_freeze_v1.md §3; same assignment rule, devel/train unchanged).
Assignment (deterministic, whole split, independent of --part): recordings sorted by file; rng = random.Random(2026);
  examples = rng.sample(sorted example ids (int), n_recordings) (all distinct); then pos = rng.randrange(2) per recording.
--part K --nparts N: process recordings [K*n//N, (K+1)*n//N) of the sorted list. --dry-run: compute everything, write nothing
  except OUT/dryrun_part{K}.json (timings, peaks, clipping, durations).
Usage: python -I scripts/lora/reverb_single.py --split {devel,train,test} --out DIR [--part K --nparts N] [--dry-run]
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / 'scripts/qwen')); sys.path.insert(0, str(PROJ / 'scripts/sms_wsj'))
from mix_slurp_v1 import extract_piece, rir_start  # noqa: E402
from qwen_omni_c4 import level  # noqa: E402

SLURP = Path('<SHARED_ROOT>/data/slurp/audio/slurp_real')
RIRS = Path('<SEPARATION_DATA_ROOT>/data/sms_wsj/rirs')
RIRSET = dict(devel='cv_dev93', train='train_si284', test='test_eval92')
MAN = PROJ / 'data/slurp/manifests/split_v2_seed2026'
SEED = 2026


def assignment(split):
    rows = sorted(csv.DictReader(open(MAN / f'{split}_recordings.csv', newline='')), key=lambda r: r['file'])
    assert len({r['file'] for r in rows}) == len(rows)
    sc = json.load(open(RIRS / 'scenarios.json'))['datasets'][RIRSET[split]]
    ids = sorted(sc, key=int); rng = random.Random(SEED)
    ex = rng.sample(ids, len(rows)); pos = [rng.randrange(2) for _ in rows]
    return [dict(file=r['file'], slurp_id=int(r['slurp_id']), rir_set=RIRSET[split], rir_example=e, rir_pos=p, t60=sc[e]['sound_decay_time'])
            for r, e, p in zip(rows, ex, pos)]


def reverb(file, rir_set, example, pos):
    x, sr = sf.read(SLURP / file, dtype='float64'); assert sr == 16000 and x.ndim == 1, (file, sr, x.shape)
    s = resample_poly(x, 1, 2)
    h, hs = sf.read(RIRS / rir_set / example / f'h_{pos}.wav', dtype='float64', always_2d=True); assert hs == 8000 and h.shape[1] == 6
    h = h.T; rs = rir_start(h)
    y8 = extract_piece(fftconvolve(s, h[0]), -rs, len(s))
    y = level(resample_poly(y8, 2, 1)[:2 * len(s)]); assert np.isfinite(y).all()
    return y.astype(np.float32), dict(rir_start=rs, n_16k_in=len(x), n_16k_out=len(y), peak=float(np.abs(y).max()), clipped=int((np.abs(y) >= 1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('devel', 'train', 'test'), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    A = assignment(a.split); n = len(A); sl = A[a.part * n // a.nparts:(a.part + 1) * n // a.nparts]
    a.out.mkdir(parents=True, exist_ok=True)
    recs, t0, nbytes = [], time.perf_counter(), 0
    for it in sl:
        y, info = reverb(it['file'], it['rir_set'], it['rir_example'], it['rir_pos'])
        r = dict(it, **info, wav=f"{Path(it['file']).stem}.wav"); nbytes += 2 * len(y) + 44
        if not a.dry_run:
            p = a.out / r['wav']; assert not p.exists(), p
            sf.write(p, y, 16000, subtype='PCM_16')
        recs.append(r)
    sec = time.perf_counter() - t0
    summ = dict(split=a.split, part=a.part, nparts=a.nparts, n_split=n, n_part=len(recs), seconds=sec, sec_per_item=sec / max(len(recs), 1),
                bytes_16bit=nbytes, audio_hours=sum(r['n_16k_out'] for r in recs) / 16000 / 3600, peak_max=max(r['peak'] for r in recs),
                clipped_total=sum(r['clipped'] for r in recs), n_distinct_examples=len({r['rir_example'] for r in A}),
                t60=dict(min=min(r['t60'] for r in recs), max=max(r['t60'] for r in recs)), dry_run=a.dry_run)
    name = f'dryrun_part{a.part}.json' if a.dry_run else f'manifest_part{a.part}.jsonl'
    if a.dry_run:
        (a.out / name).write_text(json.dumps(dict(summary=summ, first=recs[:3]), indent=1) + '\n')
    else:
        with open(a.out / name, 'w') as fh:
            for r in recs:
                fh.write(json.dumps(r) + '\n')
    print('SUMMARY', json.dumps(summ), flush=True); print('DONE', flush=True)


if __name__ == '__main__':
    main()
