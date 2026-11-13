"""Direct-path mic-0 references for mix_v1 (docs/frontend_mix_v1_eval_v2.md). Definition copied from
../speech_separation/multichannel/data/prepare.py (rir_generator 0.3.0, T60=0, order=0, c=343, 8 kHz, 8192 taps,
omnidirectional; extract_piece(offset - rir_start(full 6-ch RIR)); image gain). Run in ../speech_separation/envs/sms_wsj
(read-only use; the only env with rir_generator). Writes ONLY to OUT (outputs/...), never to data/.
Step 1: cv_dev93 validation on random.Random(2026).sample(sorted ids, 50) vs sms_wsj_direct_v1 files (max |diff| < 1e-6).
Step 2: mix_v1 (200 samples): s = resample_poly(slurp_16k, 1, 2); pos / rir_start / gain from the mix_v1 manifest;
  geometry from the cv_dev93 example. Also recomputes the mic-0 early image and checks it against the written file.
Usage: python -B scripts/sms_wsj/direct_refs_mix_v1.py MIX_DIR SLURP_AUDIO_DIR OUT_DIR
SLURP test (2026-11-12): env SLURP_EVAL_SPLIT=test -> test mix list and test_eval92 geometry for step 2 (step 1 validation
  stays cv_dev93). Devel default unchanged.
"""
import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import rir_generator  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import fftconvolve, resample_poly  # noqa: E402

sys.path.insert(0, '<SEPARATION_REPO>/repos/sms_wsj')
from sms_wsj.database.utils import extract_piece  # noqa: E402
from sms_wsj.reverb.reverb_utils import get_rir_start_sample  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJ / 'scripts/qwen'))
import split_cfg  # noqa: E402
DB = '<SEPARATION_DATA_ROOT>/data/sms_wsj/sms_wsj.json'
DIRECT = Path('<SEPARATION_REPO>/multichannel/data/sms_wsj_direct_v1')


def read(p):
    x, sr = sf.read(p, dtype='float64', always_2d=True); assert sr == 8000 or sr == 16000
    return x.T, sr


def direct_rir(ex, pos):
    return rir_generator.generate(
        c=343, fs=8000, r=np.ascontiguousarray(np.asarray(ex['sensor_position']).T[:1]),
        s=np.ascontiguousarray(np.asarray(ex['source_position'])[:, pos]),
        L=np.ascontiguousarray(np.asarray(ex['room_dimensions'])[:, 0]),
        reverberation_time=0, nsample=8192, order=0, mtype=rir_generator.mtype.omnidirectional)[:, 0]


def main(mix_dir, audio_dir, out):
    mix_dir, out = Path(mix_dir), Path(out); out.mkdir(parents=True, exist_ok=False); (out / 'direct_refs').mkdir()
    db = json.load(open(DB))['datasets']['cv_dev93']
    # ---------------- step 1: cv_dev93 validation
    ids = random.Random(2026).sample(sorted(db), 50)
    worst = 0.0
    for eid in ids:
        ex = db[eid]; ap = ex['audio_path']; n = ex['num_samples']['observation']
        for k in range(2):
            src = read(ap['original_source'][k])[0][0]
            h = read(ap['rir'][k])[0]
            start = int(get_rir_start_sample(h)); off = ex['offset'][k] - start
            raw = extract_piece(fftconvolve(src[None], h, axes=-1), off, n)
            gain = float(10 ** (ex['log_weights'][k] / 20) / max(raw.std(), np.finfo(float).tiny) / 71)
            y = extract_piece(fftconvolve(src, direct_rir(ex, k)), off, n) * gain
            ref = read(DIRECT / 'cv_dev93' / f'{eid}_{k}.wav')[0][0]
            worst = max(worst, float(np.abs(y - ref).max()))
    val = dict(n_examples=len(ids), n_refs=2 * len(ids), max_abs_diff=worst, passed=worst < 1e-6)
    print('VALIDATION', json.dumps(val), flush=True)
    if not val['passed']:
        (out / 'direct_refs_summary.json').write_text(json.dumps(dict(validation=val), indent=1) + '\n'); sys.exit(2)
    # ---------------- step 2: mix_v1
    L = {g['group']: g for g in json.load(open(split_cfg.LIST))['groups']}
    man = [json.loads(l) for l in open(mix_dir / 'manifest.jsonl')]
    if split_cfg.SPLIT != 'devel':
        db = json.load(open(DB))['datasets'][split_cfg.RIRSET]
    src_cache, early_worst, rows = {}, 0.0, []
    for m in man:
        g = L[m['group']]; key = f"g{m['group']:03d}_s{m['swap']}"; T = m['T_8k']; ex = db[g['rir']['example_id']]
        refs = {}
        for k in 'AB':
            f = g[k]['file']
            if f not in src_cache:
                x, sr = sf.read(Path(audio_dir) / f, dtype='float64'); assert sr == 16000; src_cache[f] = resample_poly(x, 1, 2)
            s = src_cache[f]; info = m['sources'][k]; pos = info['pos']
            h = read(Path(g['rir']['rir_dir']) / f'h_{pos}.wav')[0]
            assert int(get_rir_start_sample(h)) == info['rir_start']
            off = g[k]['offset_8k'] - info['rir_start']
            d = extract_piece(fftconvolve(s, direct_rir(ex, pos)), off, T) * info['gain']
            he = h[0].copy(); he[info['rir_start'] + 400:] = 0
            e = extract_piece(fftconvolve(s, he), off, T) * info['gain']
            e_file = read(mix_dir / key / f'early_{k}_6ch_8k.wav')[0][0]
            early_worst = max(early_worst, float(np.abs(e - e_file).max()))
            refs[k] = d
        np.savez(out / 'direct_refs' / f'{key}.npz', A=refs['A'], B=refs['B'])
        rows.append(dict(key=key, rms_A=float(np.sqrt(np.mean(refs['A'] ** 2))), rms_B=float(np.sqrt(np.mean(refs['B'] ** 2)))))
    summ = dict(validation=val, mix_v1_samples=len(rows), early_recompute_max_abs_diff=early_worst,
                early_recompute_ok=early_worst < 1e-6, rir_generator='0.3.0', definition='prepare.py (sms_wsj_direct_v1)')
    (out / 'direct_refs_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    print('SUMMARY', json.dumps(summ), flush=True)
    assert summ['early_recompute_ok'], early_worst


if __name__ == '__main__':
    main(*sys.argv[1:4])
